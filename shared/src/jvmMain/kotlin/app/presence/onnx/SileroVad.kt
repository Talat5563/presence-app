package app.presence.onnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import app.presence.audio.SAMPLE_RATE_HZ
import app.presence.engine.VoiceActivityDetector

/**
 * Silero VAD v5 via ONNX Runtime.
 *
 * Verified signature: inputs `input [B, samples]`, `state [2, B, 128]`, `sr` (int64 scalar);
 * outputs `output [B, 1]`, `stateN`.
 *
 * The model is recurrent, so state carries between frames and [reset] matters between recordings.
 */
class SileroVad(
    private val session: OnnxModelSession,
    override val frameSize: Int = 512,
) : VoiceActivityDetector, AutoCloseable {

    private companion object {
        const val STATE_LAYERS = 2
        const val STATE_SIZE = 128
    }

    private val environment: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var state = FloatArray(STATE_LAYERS * STATE_SIZE)

    override fun speechProbability(frame: FloatArray): Float {
        require(frame.size == frameSize) {
            "Silero at $SAMPLE_RATE_HZ Hz needs exactly $frameSize samples, got ${frame.size} — use Framer"
        }

        val audioTensor = session.floatTensor(frame.copyOf(), longArrayOf(1, frameSize.toLong()))
        val stateTensor = session.floatTensor(
            state.copyOf(),
            longArrayOf(STATE_LAYERS.toLong(), 1, STATE_SIZE.toLong()),
        )
        val rateTensor = OnnxTensor.createTensor(environment, longArrayOf(SAMPLE_RATE_HZ.toLong()))

        return session.runTensors(
            mapOf("input" to audioTensor, "state" to stateTensor, "sr" to rateTensor),
        ) { result ->
            @Suppress("UNCHECKED_CAST")
            val probability = (result.get("output").get().value as Array<FloatArray>)[0][0]
            @Suppress("UNCHECKED_CAST")
            val nextState = result.get("stateN").get().value as Array<Array<FloatArray>>
            state = FloatArray(STATE_LAYERS * STATE_SIZE) { index ->
                nextState[index / STATE_SIZE][0][index % STATE_SIZE]
            }
            probability.coerceIn(0f, 1f)
        }
    }

    override fun reset() {
        state = FloatArray(STATE_LAYERS * STATE_SIZE)
    }

    override fun close() {
        session.close()
    }
}
