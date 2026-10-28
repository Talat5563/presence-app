package app.presence.harness

import app.presence.audio.Framer
import app.presence.engine.VoiceActivityDetector
import app.presence.onnx.OnnxModelSession
import app.presence.onnx.SileroVad
import java.io.BufferedOutputStream
import java.io.DataOutputStream
import java.nio.file.Path
import kotlin.io.path.bufferedWriter
import kotlin.io.path.createDirectories
import kotlin.io.path.outputStream
import kotlin.io.path.readBytes
import kotlin.io.path.readLines
import kotlin.system.exitProcess

private const val PROBES_USAGE = """
Records Silero's speech probability for every frame of every recording in a manifest, so VAD
gate settings can be swept offline (tools/experiments/vad_sweep.py) without re-running the model.

Usage:
  ./gradlew :harness:vadProbes --args="--manifest DIR --models DIR --out DIR"

  --manifest DIR   directory containing recordings.csv
  --models DIR     directory containing silero_vad.onnx
  --out DIR        receives probs.f32 (big-endian float32, every recording's frames back to back)
                   and index.csv (path, household, speaker, role, content type, first frame,
                   frame count)

Frames are 512 samples (32 ms) at 16 kHz, fed exactly as the engine feeds them: the VAD is reset
per recording and a trailing partial frame is dropped.
"""

object VadProbes {
    @JvmStatic
    fun main(argv: Array<String>) {
        val args = try {
            parseFlags(argv, setOf("manifest", "models", "out", "help"))
        } catch (failure: IllegalArgumentException) {
            System.err.println("error: ${failure.message}")
            println(PROBES_USAGE)
            exitProcess(2)
        }
        if (args.isEmpty() || "help" in args || listOf("manifest", "models", "out").any { it !in args }) {
            println(PROBES_USAGE)
            return
        }

        val manifestDir = Path.of(args.getValue("manifest"))
        val recordings = Manifest.readRecordings(manifestDir.resolve("recordings.csv"))
        val outDir = Path.of(args.getValue("out")).also { it.createDirectories() }

        var firstFrame = 0L
        var failed = 0
        SileroVad(OnnxModelSession(Path.of(args.getValue("models")).resolve("silero_vad.onnx"))).use { vad ->
            DataOutputStream(BufferedOutputStream(outDir.resolve("probs.f32").outputStream())).use { probs ->
                outDir.resolve("index.csv").bufferedWriter().use { index ->
                    index.appendLine("path,household_id,speaker_id,role,content_type,first_frame,frames")
                    recordings.forEachIndexed { position, recording ->
                        val samples = try {
                            WavReader.read(manifestDir.resolve(recording.path))
                        } catch (failure: UnsupportedAudioException) {
                            failed += 1
                            System.err.println("${recording.path}: ${failure.message}")
                            return@forEachIndexed
                        }
                        vad.reset()
                        var frames = 0
                        for (frame in Framer(vad.frameSize).push(samples)) {
                            probs.writeFloat(vad.speechProbability(frame))
                            frames += 1
                        }
                        index.appendLine(
                            listOf(
                                recording.path, recording.householdId, recording.speakerId,
                                recording.role.name.lowercase(), recording.contentType.name.lowercase(),
                                firstFrame, frames,
                            ).joinToString(","),
                        )
                        firstFrame += frames
                        if ((position + 1) % 100 == 0) println("  ... ${position + 1}/${recordings.size} recordings")
                    }
                }
            }
        }
        println("Wrote $firstFrame frames from ${recordings.size - failed} recordings to $outDir")
        if (failed > 0) exitProcess(1)
    }
}

/**
 * Speech probabilities recorded by [VadProbes], replayed in place of Silero.
 *
 * Silero costs more harness time than the embedder does, and its output does not change when only
 * the gate settings around it do, so a sweep of those settings can replay it instead.
 */
class RecordedProbes(dir: Path) {
    private val probabilities: FloatArray
    private val framesByPath: Map<String, Pair<Int, Int>>

    init {
        val bytes = dir.resolve("probs.f32").readBytes()
        val buffer = java.nio.ByteBuffer.wrap(bytes) // big-endian, as DataOutputStream wrote it
        probabilities = FloatArray(bytes.size / Float.SIZE_BYTES) { buffer.getFloat() }
        framesByPath = dir.resolve("index.csv").readLines().drop(1).filter { it.isNotBlank() }.associate { line ->
            val fields = line.split(",")
            fields[0] to (fields[5].toInt() to fields[6].toInt())
        }
    }

    /** A detector that replays [recording]'s frames in order, and fails if fed one too many. */
    fun forRecording(recording: Recording): VoiceActivityDetector {
        val (first, count) = framesByPath[recording.path]
            ?: error("${recording.path} has no recorded VAD probabilities; re-run vadProbes on this manifest")
        return object : VoiceActivityDetector {
            private var next = 0
            override val frameSize = 512

            override fun speechProbability(frame: FloatArray): Float {
                check(next < count) { "${recording.path}: fed more frames than the $count recorded" }
                return probabilities[first + next++]
            }

            override fun reset() {
                next = 0
            }
        }
    }
}
