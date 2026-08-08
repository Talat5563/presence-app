package app.presence.onnx

import ai.onnxruntime.TensorInfo
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
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

            val state = session.inputInfo.getValue("state").info as TensorInfo
            assertEquals(2L, state.shape[0], "Silero v5 carries 2 state layers")
            assertEquals(128L, state.shape[2], "Silero v5 state width is 128")
        }
    }

    @Test
    fun `exposes the verified embedding model signature`() {
        OnnxModelSession(embeddingModel).use { session ->
            println("=== Speaker embedding ===\n${session.describeSignature()}")

            assertTrue("feats" in session.inputNames, "inputs were ${session.inputNames}")
            assertTrue("embs" in session.outputNames, "outputs were ${session.outputNames}")

            val feats = session.inputInfo.getValue("feats").info as TensorInfo
            assertEquals(3, feats.shape.size, "feats should be [B, T, 80]")
            assertEquals(80L, feats.shape[2], "Task 13's filterbank produces 80 bins")

            val embs = session.outputInfo.getValue("embs").info as TensorInfo
            assertEquals(2, embs.shape.size, "embs should be [B, 256]")
            assertEquals(256L, embs.shape[1], "Task 14 expects 256 dimensions")
        }
    }
}
