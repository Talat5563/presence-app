package app.presence.harness

import app.presence.onnx.OnnxModelSession
import app.presence.onnx.OnnxSpeakerEmbedder
import java.nio.file.Path
import kotlin.io.path.bufferedWriter
import kotlin.io.path.createDirectories
import kotlin.io.path.readLines
import kotlin.system.exitProcess

private const val EMBED_USAGE = """
Embeds whole audio files with the shipping front end and speaker model, for verification
benchmarks such as Samrómur's trials.

Usage:
  ./gradlew :harness:embedFiles --args="--list FILE --models DIR --out FILE [--model NAME]"

  --list FILE    one WAV path per line (16 kHz mono 16-bit), relative to the list's directory
  --models DIR   directory holding the speaker model
  --model NAME   speaker model file in --models (default: speaker_embedding.onnx), so a
                 fine-tuned candidate can be scored beside the current one
  --out FILE     CSV: path, then one column per embedding dimension

Each file is embedded whole, without the VAD or windowing the engine applies: verification
benchmarks score utterance against utterance, and this keeps the numbers comparable with
published results while still using the exact Kotlin front end the app ships.
"""

object EmbedFiles {
    @JvmStatic
    fun main(argv: Array<String>) {
        val args = try {
            parseFlags(argv, setOf("list", "models", "model", "out", "help"))
        } catch (failure: IllegalArgumentException) {
            System.err.println("error: ${failure.message}")
            println(EMBED_USAGE)
            exitProcess(2)
        }
        if (args.isEmpty() || "help" in args || listOf("list", "models", "out").any { it !in args }) {
            println(EMBED_USAGE)
            return
        }

        val list = Path.of(args.getValue("list"))
        val root = list.toAbsolutePath().parent
        val paths = list.readLines().map { it.trim() }.filter { it.isNotEmpty() }
        val model = Path.of(args.getValue("models")).resolve(args["model"] ?: "speaker_embedding.onnx")
        val out = Path.of(args.getValue("out")).also { it.toAbsolutePath().parent?.createDirectories() }

        var failed = 0
        OnnxSpeakerEmbedder(OnnxModelSession(model)).use { embedder ->
            out.bufferedWriter().use { writer ->
                paths.forEachIndexed { index, relative ->
                    try {
                        val values = embedder.embed(WavReader.read(root.resolve(relative))).toFloatArray()
                        if (index == 0) writer.appendLine("path," + values.indices.joinToString(",") { "e$it" })
                        writer.appendLine("$relative," + values.joinToString(","))
                    } catch (failure: Exception) {
                        failed += 1
                        System.err.println("$relative: ${failure.message}")
                    }
                    if ((index + 1) % 250 == 0) println("  ... ${index + 1}/${paths.size} embedded")
                }
            }
        }
        println("Embedded ${paths.size - failed} of ${paths.size} files with ${model.fileName} into $out")
        if (failed > 0) exitProcess(1)
    }
}
