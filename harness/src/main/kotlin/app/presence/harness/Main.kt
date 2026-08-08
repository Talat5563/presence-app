package app.presence.harness

import app.presence.audio.SAMPLE_RATE_HZ
import app.presence.domain.Language
import app.presence.domain.VoiceProfile
import app.presence.engine.DetectionEngine
import app.presence.engine.EvidenceAccumulator
import app.presence.engine.HouseholdMatcher
import app.presence.engine.SpeakerEmbedder
import app.presence.engine.VadGate
import app.presence.enrollment.EnrollmentClip
import app.presence.enrollment.EnrollmentPolicy
import app.presence.enrollment.EnrollmentResult
import app.presence.enrollment.EnrollmentService
import app.presence.onnx.OnnxModelSession
import app.presence.onnx.OnnxSpeakerEmbedder
import app.presence.onnx.SileroVad
import app.presence.audio.Framer
import java.nio.file.Path
import kotlin.io.path.createDirectories
import kotlin.io.path.exists
import kotlin.io.path.readText
import kotlin.io.path.writeText
import kotlin.system.exitProcess

private const val USAGE = """
Phase 0 accuracy harness.

Usage:
  ./gradlew :harness:run --args="--manifest DIR --models DIR --out DIR [--holdout h3,h4] [--evaluate dev|holdout] [--input-name N] [--output-name N]"

  --manifest DIR     directory containing recordings.csv and events.csv; audio paths are
                     resolved relative to it
  --models DIR       directory containing silero_vad.onnx and speaker_embedding.onnx
  --out DIR          where report.md and results.csv are written
  --holdout LIST     comma-separated household ids reserved for final measurement only
  --evaluate MODE    "dev" (default) sweeps and tunes on every household not in --holdout.
                     "holdout" measures once, only on --holdout households, and is the only
                     honest final number. A threshold chosen by looking at dev means nothing
                     until it is re-measured on holdout — dev numbers are for tuning, not
                     for deciding ship/iterate/fall-back.
  --input-name N     embedding model input tensor name (default: feats)
  --output-name N    embedding model output tensor name (default: embs)

Run ./tools/fetch-models.sh first. See spec §9.1 for the manifest format.
"""

fun main(argv: Array<String>) {
    val args = try {
        parseArgs(argv)
    } catch (failure: IllegalArgumentException) {
        System.err.println("error: ${failure.message}")
        println(USAGE)
        exitProcess(2)
    }
    if (args.isEmpty() || "help" in args) {
        println(USAGE)
        return
    }
    for (required in listOf("manifest", "models", "out")) {
        if (required !in args) {
            System.err.println("error: --$required is required")
            println(USAGE)
            exitProcess(2)
        }
    }

    val manifestDir = Path.of(args.getValue("manifest"))
    val modelsDir = Path.of(args.getValue("models"))
    val outDir = Path.of(args.getValue("out"))

    val recordings = Manifest.readRecordings(manifestDir.resolve("recordings.csv"))
    val events = Manifest.readEvents(manifestDir.resolve("events.csv"))
    validateEvents(recordings, events)
    println("Loaded ${recordings.size} recordings and ${events.size} call events.")

    val holdout = args["holdout"]?.split(",")?.map { it.trim() }?.filter { it.isNotEmpty() }?.toSet().orEmpty()
    val households = recordings.map { it.householdId }.toSet()
    val unknownHoldout = holdout - households
    require(unknownHoldout.isEmpty()) {
        "--holdout names households not in the corpus: ${unknownHoldout.joinToString()}. " +
            "Known: ${households.sorted().joinToString()}"
    }
    if (holdout.isNotEmpty()) println("Holding out households: ${holdout.joinToString()}")

    val mode = args["evaluate"] ?: "dev"
    require(mode == "dev" || mode == "holdout") { "--evaluate must be 'dev' or 'holdout', was '$mode'" }
    require(mode == "dev" || holdout.isNotEmpty()) { "--evaluate holdout needs --holdout" }

    val warnings = mutableListOf<String>()

    SileroVad(OnnxModelSession(modelsDir.resolve("silero_vad.onnx"))).use { vad ->
        OnnxSpeakerEmbedder(
            OnnxModelSession(modelsDir.resolve("speaker_embedding.onnx")),
            inputName = args["input-name"] ?: "feats",
            outputName = args["output-name"] ?: "embs",
        ).use { embedder ->
            val traces = ArrayList<Trace>()
            val totalToTrace = recordings.groupBy { it.householdId }.values.sumOf { householdRecordings ->
                val split = Evaluation.splitCorpus(householdRecordings)
                split.positives.size + split.negatives.size
            }
            var traced = 0

            for ((householdId, householdRecordings) in recordings.groupBy { it.householdId }) {
                val split = Evaluation.splitCorpus(householdRecordings)
                val profiles = enrolProfiles(split.enrollment, manifestDir, vad, embedder, warnings)
                val target = profiles.firstOrNull { it.role == VoiceProfile.Role.TARGET_CHILD }
                if (target == null) {
                    val message = "SKIP household $householdId: no target child enrolled"
                    println(message)
                    warnings += message
                    continue
                }
                val cohort = profiles - target
                if (cohort.isEmpty()) {
                    val message =
                        "SKIP household $householdId: no cohort — spec §5 requires the parent and siblings"
                    println(message)
                    warnings += message
                    continue
                }
                println("Household $householdId: target ${target.personId}, cohort ${cohort.map { it.personId }}")

                for (recording in split.positives + split.negatives) {
                    // One bad file must not abort a multi-hour run and discard every trace
                    // computed so far — WavReader.read is designed to be caught per file.
                    try {
                        traces += traceOf(recording, manifestDir, vad, embedder, target, cohort)
                    } catch (failure: UnsupportedAudioException) {
                        val message = "$householdId: skipped ${recording.path} — ${failure.message}"
                        println("  $message")
                        warnings += message
                    }
                    traced += 1
                    if (traced % 25 == 0) println("  ... $traced/$totalToTrace recordings traced")
                }
            }

            // dev sweeps and tunes; holdout measures once, on data never used for tuning.
            val evaluated = when (mode) {
                "holdout" -> traces.filter { it.recording.householdId in holdout }
                else -> traces.filter { it.recording.householdId !in holdout }
            }
            val thresholds = (2..20 step 2).map { it.toFloat() }
            val outcomes = buildList {
                addAll(Evaluation.sweep(evaluated, events, thresholds, "ALL"))
                addAll(Evaluation.sweep(evaluated, events, thresholds, "ENGLISH") {
                    it.language == Language.ENGLISH && !it.codeSwitched
                })
                addAll(Evaluation.sweep(evaluated, events, thresholds, "HEBREW") {
                    it.language == Language.HEBREW && !it.codeSwitched
                })
                addAll(Evaluation.sweep(evaluated, events, thresholds, "CODE_SWITCHED") { it.codeSwitched })
            }

            val provenanceFile = modelsDir.resolve("PROVENANCE.txt")
            val provenance = if (provenanceFile.exists()) provenanceFile.readText() else "unknown"

            outDir.createDirectories()
            outDir.resolve("report.md").writeText(Report.markdown(outcomes, provenance, warnings))
            outDir.resolve("results.csv").writeText(Report.csv(outcomes))
            println("Wrote ${outDir.resolve("report.md")} and ${outDir.resolve("results.csv")}")

            println()
            println(
                "Best verdict per slice (ENGLISH and HEBREW must each clear the bar " +
                    "independently — a system that passes ALL or one language and fails the other " +
                    "has not passed):",
            )
            for (sliceLabel in listOf("ALL", "ENGLISH", "HEBREW", "CODE_SWITCHED")) {
                val best = outcomes.filter { it.label == sliceLabel }.minByOrNull { it.verdict.rank }
                println("  $sliceLabel: ${best?.verdict ?: "none"}")
            }
        }
    }
}

/**
 * An event whose path does not match a manifest row would be silently dropped by the replay
 * filter, lowering totalEvents and raising the detection rate. One pointing at a non-target
 * recording would let a false alarm be counted as a hit. Both corrupt the headline number quietly,
 * so both are fatal at load time.
 */
private fun validateEvents(recordings: List<Recording>, events: List<CallEvent>) {
    val byPath = recordings.associateBy { it.path }
    val problems = events.mapNotNull { event ->
        val recording = byPath[event.recordingPath]
        when {
            recording == null -> "${event.recordingPath}: no such recording in the manifest"
            recording.role != RecordingRole.TARGET_CHILD ->
                "${event.recordingPath}: events may only be labelled on target-child recordings, " +
                    "but this is ${recording.role}"
            recording.speakerId != event.targetSpeakerId ->
                "${event.recordingPath}: event names speaker '${event.targetSpeakerId}' but the " +
                    "recording is by '${recording.speakerId}'"
            else -> null
        }
    }
    require(problems.isEmpty()) {
        "events.csv does not agree with recordings.csv:\n" + problems.joinToString("\n") { "  - $it" }
    }
}

private fun enrolProfiles(
    enrollment: List<Recording>,
    root: Path,
    vad: SileroVad,
    embedder: SpeakerEmbedder,
    warnings: MutableList<String>,
): List<VoiceProfile> =
    enrollment.groupBy { it.speakerId }.mapNotNull { (speakerId, group) ->
        // Runtime observations are VAD-trimmed windows of at most 3 s each; embedding a whole
        // unfiltered read passage in one shot would be a systematic mismatch that depresses
        // targetSimilarity against the absolute minTargetSimilarity gate. Running enrollment audio
        // through the same VadGate the runtime uses, and embedding per segment, keeps the two
        // comparable.
        val clips = group.flatMap { recording ->
            val language = recording.language ?: return@flatMap emptyList()
            val samples = try {
                WavReader.read(root.resolve(recording.path))
            } catch (failure: UnsupportedAudioException) {
                warnings += "$speakerId: skipped ${recording.path} — ${failure.message}"
                return@flatMap emptyList()
            }

            vad.reset()
            val vadGate = VadGate(vad)
            val segments = buildList {
                val framer = Framer(vad.frameSize)
                for (frame in framer.push(samples)) vadGate.accept(frame)?.let { add(it) }
                vadGate.flush()?.let { add(it) }
            }

            segments.filter { it.durationMs >= 800 }.map { segment ->
                EnrollmentClip(
                    language = language,
                    embedding = embedder.embed(segment.samples),
                    speechSeconds = segment.samples.size / SAMPLE_RATE_HZ.toDouble(),
                )
            }
        }
        if (clips.isEmpty()) {
            warnings += "$speakerId: no usable enrollment audio"
            return@mapNotNull null
        }
        // Require only the languages this person actually has material for. Demanding both would
        // flag every monolingual household, drowning real shortfalls in expected noise.
        val policy = EnrollmentPolicy(
            minSecondsPerLanguage = 20.0,
            requiredLanguages = clips.map { it.language }.toSet(),
        )
        when (val result = EnrollmentService(policy).enroll(speakerId, group.first().role.toProfileRole(), clips)) {
            is EnrollmentResult.Complete -> result.profile
            is EnrollmentResult.Incomplete -> {
                val message = "$speakerId: enrolled partially, short by ${result.shortfallSeconds}"
                println("  $message")
                warnings += message
                result.profile
            }
        }
    }

private fun traceOf(
    recording: Recording,
    root: Path,
    vad: SileroVad,
    embedder: SpeakerEmbedder,
    target: VoiceProfile,
    cohort: List<VoiceProfile>,
): Trace {
    val samples = WavReader.read(root.resolve(recording.path))
    val observations = ArrayList<TimedScore>()

    vad.reset()
    val engine = DetectionEngine(
        vadGate = VadGate(vad),
        embedder = embedder,
        matcher = HouseholdMatcher(target, cohort),
        accumulator = EvidenceAccumulator(),
        targetPersonId = target.personId,
        observationListener = { timestampMs, score -> observations += TimedScore(timestampMs, score) },
    )

    val framer = Framer(vad.frameSize)
    for (frame in framer.push(samples)) engine.accept(frame)
    engine.flush()

    return Trace(recording, observations, samples.size.toDouble() / SAMPLE_RATE_HZ)
}

private val KNOWN_FLAGS = setOf(
    "help", "manifest", "models", "out", "holdout", "evaluate", "input-name", "output-name",
)

internal fun parseArgs(argv: Array<String>): Map<String, String> {
    val args = mutableMapOf<String, String>()
    var index = 0
    while (index < argv.size) {
        val token = argv[index]
        require(token.startsWith("--")) { "unexpected argument: $token" }
        val key = token.removePrefix("--")
        require(key in KNOWN_FLAGS) {
            "unknown flag --$key. Known flags: ${KNOWN_FLAGS.sorted().joinToString { "--$it" }}"
        }
        if (key == "help") {
            args["help"] = "true"
            index += 1
            continue
        }
        require(index + 1 < argv.size) { "missing value for --$key" }
        val value = argv[index + 1]
        // A flag-shaped value almost always means the real value was forgotten.
        require(!value.startsWith("--")) { "--$key expects a value but was followed by $value" }
        args[key] = value
        index += 2
    }
    return args
}
