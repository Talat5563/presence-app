package app.presence.onnx

import app.presence.audio.SAMPLE_RATE_HZ
import java.nio.file.Path
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import org.junit.jupiter.api.Tag

/** Requires models/silero_vad.onnx. Run with ./gradlew :shared:integrationTest */
@Tag("model")
class SileroVadTest {

    private val modelPath = Path.of("../models/silero_vad.onnx")

    private fun withVad(block: (SileroVad) -> Unit) =
        SileroVad(OnnxModelSession(modelPath)).use(block)

    private fun silence() = FloatArray(512)

    private fun tone(frequencyHz: Double = 220.0) = FloatArray(512) {
        (0.3 * sin(2 * PI * frequencyHz * it / SAMPLE_RATE_HZ)).toFloat()
    }

    private fun noise(seed: Int) =
        Random(seed).let { rng -> FloatArray(512) { rng.nextFloat() * 0.02f - 0.01f } }

    /** Broadband noise at amplitude 0.5 — the one synthetic signal Silero reacts to. */
    private fun loudNoise(seed: Int) =
        Random(seed).let { rng -> FloatArray(512) { rng.nextFloat() - 0.5f } }

    @Test
    fun `reports the frame size the model requires`() {
        withVad { vad -> assertEquals(512, vad.frameSize) }
    }

    @Test
    fun `returns a probability in range for any input`() {
        withVad { vad ->
            listOf(silence(), tone(), noise(1)).forEach { frame ->
                val probability = vad.speechProbability(frame)
                assertTrue(probability in 0f..1f, "probability $probability outside [0,1]")
                assertTrue(probability.isFinite())
            }
        }
    }

    /**
     * Silero is not an energy detector — it responds to speech specifically. A pure tone has no
     * broadband excitation, no formant structure and no temporal dynamics, so the model rates it
     * no higher than silence (measured: tone 5.26e-4, silence 5.42e-4 — the tone marginally
     * *lower*). Broadband noise is the one synthetic signal it reacts to, at roughly 3.7x silence.
     *
     * So this is a smoke test that the model is genuinely a function of its input rather than
     * returning a constant. Real speech discrimination is measured on actual recordings in
     * Task 17; it cannot be established with synthetic audio.
     */
    @Test
    fun `distinguishes broadband noise from digital silence`() {
        withVad { vad ->
            // Averaged over several frames: the recurrent state settles after the first few.
            val silenceScore = (1..10).map { vad.speechProbability(silence()) }.average()
            vad.reset()
            val noiseScore = (1..10).map { vad.speechProbability(loudNoise(it)) }.average()

            assertTrue(
                noiseScore > silenceScore * 1.5,
                "broadband noise ($noiseScore) should score clearly above silence ($silenceScore)",
            )
        }
    }

    /**
     * Records what the model actually does with synthetic signals, for calibration. Asserts only
     * what must hold regardless — an ordering assertion here would be wishful thinking, since
     * none of these inputs is speech.
     */
    @Test
    fun `reports synthetic signal probabilities for inspection`() {
        withVad { vad ->
            val readings = linkedMapOf<String, Double>()
            listOf(
                "silence" to { silence() },
                "tone-150Hz" to { tone(150.0) },
                "tone-220Hz" to { tone(220.0) },
                "quiet-noise" to { noise(1) },
                "loud-noise" to { loudNoise(1) },
            ).forEach { (label, make) ->
                vad.reset()
                readings[label] = (1..10).map { vad.speechProbability(make()) }.average()
            }
            println("Silero synthetic-signal probabilities: $readings")

            readings.forEach { (label, value) ->
                assertTrue(value in 0.0..1.0 && value.isFinite(), "$label produced $value")
            }
        }
    }

    @Test
    fun `carries recurrent state between frames`() {
        withVad { vad ->
            val first = vad.speechProbability(tone())
            val tenth = (1..9).map { vad.speechProbability(tone()) }.last()

            assertTrue(
                abs(first - tenth) > 1e-6f,
                "state should evolve; got $first then $tenth",
            )
        }
    }

    @Test
    fun `rejects a frame of the wrong size`() {
        withVad { vad ->
            val failure = assertFailsWith<IllegalArgumentException> {
                vad.speechProbability(FloatArray(256))
            }
            assertTrue(failure.message!!.contains("512"))
        }
    }

    @Test
    fun `reset returns the detector to its initial response`() {
        withVad { vad ->
            val first = vad.speechProbability(tone())
            repeat(20) { vad.speechProbability(tone()) }
            vad.reset()
            val afterReset = vad.speechProbability(tone())

            assertTrue(
                abs(first - afterReset) < 1e-4f,
                "expected $first after reset, got $afterReset",
            )
        }
    }
}
