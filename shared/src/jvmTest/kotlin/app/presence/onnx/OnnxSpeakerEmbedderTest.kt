package app.presence.onnx

import app.presence.audio.SAMPLE_RATE_HZ
import app.presence.engine.cosine
import java.nio.file.Path
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import org.junit.jupiter.api.Tag

/** Requires models/speaker_embedding.onnx. Run with ./gradlew :shared:integrationTest */
@Tag("model")
class OnnxSpeakerEmbedderTest {

    private val modelPath = Path.of("../models/speaker_embedding.onnx")

    private fun withEmbedder(block: (OnnxSpeakerEmbedder) -> Unit) =
        OnnxSpeakerEmbedder(OnnxModelSession(modelPath)).use(block)

    /** A crude voice-like signal: a fundamental plus two harmonics. */
    private fun voice(fundamentalHz: Double, seconds: Double = 2.0): FloatArray {
        val count = (seconds * SAMPLE_RATE_HZ).toInt()
        return FloatArray(count) { i ->
            val t = i / SAMPLE_RATE_HZ.toDouble()
            (0.4 * sin(2 * PI * fundamentalHz * t) +
                0.2 * sin(2 * PI * fundamentalHz * 2 * t) +
                0.1 * sin(2 * PI * fundamentalHz * 3 * t)).toFloat()
        }
    }

    @Test
    fun `produces a unit-length embedding`() {
        withEmbedder { embedder ->
            val embedding = embedder.embed(voice(180.0))

            var sumOfSquares = 0.0
            for (v in embedding.values) sumOfSquares += v.toDouble() * v
            assertTrue(abs(1.0 - sqrt(sumOfSquares)) < 1e-4, "embedding was not unit length")
        }
    }

    @Test
    fun `produces the 256 dimensions the model declares`() {
        withEmbedder { embedder ->
            assertEquals(256, embedder.embed(voice(180.0)).dimension)
            assertEquals(256, embedder.dimension)
        }
    }

    @Test
    fun `every component is finite`() {
        withEmbedder { embedder ->
            embedder.embed(voice(180.0)).values.forEach {
                assertTrue(it.isFinite(), "non-finite component $it")
            }
        }
    }

    @Test
    fun `is deterministic for identical input`() {
        withEmbedder { embedder ->
            val samples = voice(180.0)

            val similarity = cosine(embedder.embed(samples), embedder.embed(samples.copyOf()))

            assertTrue(abs(1f - similarity) < 1e-4f, "expected 1.0, got $similarity")
        }
    }

    @Test
    fun `handles utterances of different lengths`() {
        withEmbedder { embedder ->
            // T varies with duration; the model takes a dynamic time axis.
            listOf(1.0, 2.0, 5.0).forEach { seconds ->
                assertEquals(256, embedder.embed(voice(180.0, seconds)).dimension)
            }
        }
    }

    @Test
    fun `reports synthetic voice separation for inspection`() {
        withEmbedder { embedder ->
            val speakerAFirst = embedder.embed(voice(120.0, seconds = 2.0))
            val speakerASecond = embedder.embed(voice(120.0, seconds = 3.0))
            val speakerB = embedder.embed(voice(260.0, seconds = 2.0))

            val sameSpeaker = cosine(speakerAFirst, speakerASecond)
            val differentSpeaker = cosine(speakerAFirst, speakerB)
            println("synthetic same-speaker=$sameSpeaker different-speaker=$differentSpeaker")

            // Only what must hold regardless of how the model treats synthetic tones. Real
            // separation is measured on actual corpora in Task 17 — asserting an ordering here
            // would be a coin flip, because these tones are far outside the training distribution.
            assertTrue(sameSpeaker.isFinite() && sameSpeaker in -1f..1f, "got $sameSpeaker")
            assertTrue(differentSpeaker.isFinite() && differentSpeaker in -1f..1f, "got $differentSpeaker")
        }
    }

    @Test
    fun `rejects audio too short to embed`() {
        withEmbedder { embedder ->
            assertFailsWith<IllegalArgumentException> { embedder.embed(FloatArray(100)) }
        }
    }
}
