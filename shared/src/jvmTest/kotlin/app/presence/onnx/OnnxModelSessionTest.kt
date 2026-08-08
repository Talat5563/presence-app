package app.presence.onnx

import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import org.junit.jupiter.api.Tag

/** Requires the files in models/. Run with ./gradlew :shared:integrationTest */
@Tag("model")
class OnnxModelSessionTest {

    private val vadModel = Path.of("../models/silero_vad.onnx")
    private val embeddingModel = Path.of("../models/speaker_embedding.onnx")

    @Test
    fun `reports a missing model file clearly`() {
        val failure = assertFailsWith<IllegalArgumentException> {
            OnnxModelSession(Path.of("../models/not-a-real-model.onnx"))
        }
        assertTrue(failure.message!!.contains("not-a-real-model.onnx"))
        assertTrue(failure.message!!.contains("fetch-models"), "should say how to fix it")
    }

    @Test
    fun `exposes the verified Silero VAD signature`() {
        OnnxModelSession(vadModel).use { session ->
            println("=== Silero VAD ===\n${session.describeSignature()}")

            assertTrue("input" in session.inputNames, "inputs were ${session.inputNames}")
            assertTrue("state" in session.inputNames, "expected v5 single state tensor")
            assertTrue("sr" in session.inputNames)
            assertTrue("output" in session.outputNames)
            assertTrue("stateN" in session.outputNames)
        }
    }

    @Test
    fun `exposes the verified embedding model signature`() {
        OnnxModelSession(embeddingModel).use { session ->
            println("=== Speaker embedding ===\n${session.describeSignature()}")

            assertTrue("feats" in session.inputNames, "inputs were ${session.inputNames}")
            assertTrue("embs" in session.outputNames, "outputs were ${session.outputNames}")
        }
    }
}
