package app.presence.harness

import app.presence.audio.SAMPLE_RATE_HZ
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.FileSystems
import java.nio.file.Files
import java.nio.file.Path
import javax.sound.sampled.AudioFileFormat
import javax.sound.sampled.AudioFormat
import javax.sound.sampled.AudioInputStream
import javax.sound.sampled.AudioSystem
import kotlin.io.path.createDirectories
import kotlin.io.path.exists
import kotlin.io.path.extension
import kotlin.io.path.invariantSeparatorsPathString
import kotlin.io.path.isRegularFile
import kotlin.io.path.nameWithoutExtension
import kotlin.io.path.readText
import kotlin.io.path.writeText
import kotlin.system.exitProcess

private const val IMPORT_USAGE = """
Converts CHILDES (CHAT) transcripts and their media into a harness corpus.

Usage:
  ./gradlew :harness:importChildes --args="--transcripts DIR --media DIR --households FILE --out DIR"

  --transcripts DIR  root of the unzipped CHAT transcripts
  --media DIR        root of the downloaded media, laid out like the transcripts
  --households FILE  CSV with columns household_id,transcripts,target. "transcripts" is a glob
                     relative to --transcripts (e.g. MacWhinney/*.cha); "target" is the child's
                     participant name (e.g. Ross) or code (e.g. CHI)
  --out DIR          where wav/, recordings.csv and events.csv are written; pass it to the
                     harness as --manifest
  --cache DIR        where converted 16 kHz session audio is kept (default: OUT/cache)

Needs ffmpeg on the PATH. CHILDES data is CC BY-NC-SA: evaluate with it, never redistribute it
or train a shipped model on it.
"""

private val IMPORT_FLAGS = setOf("transcripts", "media", "households", "out", "cache", "help")

object ChildesImport {
    @JvmStatic
    fun main(argv: Array<String>) = importChildes(argv)
}

private fun importChildes(argv: Array<String>) {
    val args = try {
        parseFlags(argv, IMPORT_FLAGS)
    } catch (failure: IllegalArgumentException) {
        System.err.println("error: ${failure.message}")
        println(IMPORT_USAGE)
        exitProcess(2)
    }
    if (args.isEmpty() || "help" in args) {
        println(IMPORT_USAGE)
        return
    }
    for (required in listOf("transcripts", "media", "households", "out")) {
        if (required !in args) {
            System.err.println("error: --$required is required")
            println(IMPORT_USAGE)
            exitProcess(2)
        }
    }

    val transcriptsRoot = Path.of(args.getValue("transcripts"))
    val mediaRoot = Path.of(args.getValue("media"))
    val outDir = Path.of(args.getValue("out"))
    val cacheDir = args["cache"]?.let { Path.of(it) } ?: outDir.resolve("cache")

    val transcriptFiles = Files.walk(transcriptsRoot).use { paths ->
        paths.filter { it.isRegularFile() && it.extension.equals("cha", ignoreCase = true) }.toList()
    }
    val mediaFiles = Files.walk(mediaRoot).use { paths -> paths.filter { it.isRegularFile() }.toList() }

    val sessionDirectories = HashMap<String, Path>()
    val households = Csv.parse(Path.of(args.getValue("households")).readText()).map { row ->
        val householdId = row["household_id"] ?: error("households CSV needs a household_id column")
        val glob = row["transcripts"] ?: error("households CSV needs a transcripts column")
        val target = row["target"] ?: error("households CSV needs a target column")
        val matcher = FileSystems.getDefault().getPathMatcher("glob:$glob")
        val sessions = transcriptFiles
            .map { transcriptsRoot.relativize(it) }
            .filter { matcher.matches(Path.of(it.invariantSeparatorsPathString)) }
            .map { relative ->
                val id = ChildesPlanner.sanitize(relative.invariantSeparatorsPathString.removeSuffix(".cha"))
                sessionDirectories["$householdId/$id"] = relative.parent ?: Path.of("")
                ChildesSession(id, Chat.parse(transcriptsRoot.resolve(relative).readText()))
            }
        require(sessions.isNotEmpty()) { "household $householdId: no transcripts match '$glob'" }
        val unaligned = sessions.sumOf { it.transcript.unalignedUtterances }
        val aligned = sessions.sumOf { it.transcript.utterances.size }
        println("Household $householdId: ${sessions.size} transcripts, $aligned aligned / $unaligned unaligned utterances")
        HouseholdSource(householdId, target, sessions)
    }

    val plan = ChildesPlanner.plan(households)
    val warnings = plan.warnings.toMutableList()
    val written = ArrayList<PlannedFile>()
    val audio = SessionAudio(cacheDir)

    for (file in plan.files) {
        val household = file.recording.householdId
        val sessions = households.first { it.householdId == household }.sessions.associateBy { it.id }
        val samples = try {
            stitchAudio(file.pieces) { sessionId ->
                val mediaName = sessions.getValue(sessionId).transcript.mediaName!!
                val media = findMedia(mediaFiles, mediaRoot.resolve(sessionDirectories.getValue("$household/$sessionId")), mediaName)
                    ?: throw UnsupportedAudioException("no media file named '$mediaName' under $mediaRoot")
                audio.load("$household-$sessionId", media)
            }
        } catch (failure: UnsupportedAudioException) {
            warnings += "${file.recording.path}: ${failure.message}"
            continue
        }
        writeWav(outDir.resolve(file.recording.path), samples)
        written += file
    }

    outDir.createDirectories()
    outDir.resolve("recordings.csv").writeText(recordingsCsv(written.map { it.recording }))
    outDir.resolve("events.csv").writeText(eventsCsv(written.mapNotNull { it.event }))

    println()
    println("Wrote ${written.size} files to $outDir:")
    for ((role, files) in written.groupBy { "${it.recording.role} ${it.recording.contentType}" }.toSortedMap()) {
        println("  $role: ${files.size} files, ${"%.2f".format(files.sumOf { it.speechMs } / 3_600_000.0)} h of speech")
    }
    println("  call events: ${written.count { it.event != null }}")
    if (warnings.isNotEmpty()) {
        println()
        println("Warnings (${warnings.size}):")
        warnings.take(50).forEach { println("  - $it") }
        if (warnings.size > 50) println("  ... and ${warnings.size - 50} more")
    }
}

/**
 * Finds a session's media: first beside where its transcript sits in the mirrored layout, then
 * anywhere under the media root, but only if that name is unique there. Any audio or video
 * container ffmpeg reads is accepted.
 */
internal fun findMedia(all: List<Path>, expectedDir: Path, mediaName: String): Path? {
    fun named(path: Path) = path.nameWithoutExtension.equals(mediaName, ignoreCase = true)
    all.firstOrNull { named(it) && it.parent?.normalize() == expectedDir.normalize() }?.let { return it }
    return all.filter(::named).singleOrNull()
}

/** Converts session media to 16 kHz mono once, and keeps the most recent few sessions decoded. */
private class SessionAudio(private val cacheDir: Path) {
    private val decoded = object : LinkedHashMap<String, FloatArray>(4, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, FloatArray>?) = size > 3
    }

    fun load(key: String, media: Path): FloatArray = decoded.getOrPut(key) {
        val wav = cacheDir.resolve("$key.wav")
        if (!wav.exists()) {
            cacheDir.createDirectories()
            val process = ProcessBuilder(
                "ffmpeg", "-y", "-loglevel", "error", "-i", media.toString(),
                "-ar", SAMPLE_RATE_HZ.toString(), "-ac", "1", "-c:a", "pcm_s16le", wav.toString(),
            ).redirectErrorStream(true).start()
            val output = process.inputStream.bufferedReader().readText()
            if (process.waitFor() != 0) {
                Files.deleteIfExists(wav)
                throw UnsupportedAudioException("ffmpeg could not convert $media: ${output.trim()}")
            }
        }
        WavReader.read(wav)
    }
}

/**
 * Concatenates [pieces], each preceded by its gap of digital silence. A piece running past the
 * end of its media is clipped rather than refused: CHAT bullets are hand-placed and the last one
 * often overshoots by a few milliseconds.
 */
internal fun stitchAudio(pieces: List<Piece>, sessionSamples: (sessionId: String) -> FloatArray): FloatArray {
    val out = ArrayList<FloatArray>()
    for (piece in pieces) {
        val source = sessionSamples(piece.sessionId)
        val start = msToSamples(piece.startMs).coerceIn(0, source.size)
        val end = msToSamples(piece.endMs).coerceIn(start, source.size)
        out += FloatArray(msToSamples(piece.gapBeforeMs))
        out += source.copyOfRange(start, end)
    }
    val total = FloatArray(out.sumOf { it.size })
    var offset = 0
    for (chunk in out) {
        chunk.copyInto(total, offset)
        offset += chunk.size
    }
    return total
}

private fun msToSamples(ms: Long): Int = (ms * SAMPLE_RATE_HZ / 1_000).toInt()

internal fun writeWav(path: Path, samples: FloatArray) {
    path.parent?.createDirectories()
    val buffer = ByteBuffer.allocate(samples.size * 2).order(ByteOrder.LITTLE_ENDIAN)
    for (sample in samples) buffer.putShort((sample.coerceIn(-1f, 1f) * Short.MAX_VALUE).toInt().toShort())
    val format = AudioFormat(SAMPLE_RATE_HZ.toFloat(), 16, 1, true, false)
    AudioSystem.write(
        AudioInputStream(buffer.array().inputStream(), format, samples.size.toLong()),
        AudioFileFormat.Type.WAVE,
        path.toFile(),
    )
}

internal fun recordingsCsv(recordings: List<Recording>): String = buildString {
    appendLine("path,household_id,speaker_id,role,language,code_switched,content_type,distance_m,placement,background,room,date")
    for (r in recordings) {
        appendLine(
            listOf(
                r.path, r.householdId, r.speakerId, r.role.name.lowercase(), r.language?.name?.lowercase().orEmpty(),
                r.codeSwitched.toString(), r.contentType.name.lowercase(), r.distanceMeters?.toString().orEmpty(),
                r.placement, r.background, r.room, r.date,
            ).joinToString(","),
        )
    }
}

internal fun eventsCsv(events: List<CallEvent>): String = buildString {
    appendLine("recording_path,start_ms,target_speaker_id")
    for (e in events) appendLine("${e.recordingPath},${e.startMs},${e.targetSpeakerId}")
}
