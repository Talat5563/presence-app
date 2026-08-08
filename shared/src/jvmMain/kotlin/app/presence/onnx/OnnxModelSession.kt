package app.presence.onnx

import ai.onnxruntime.NodeInfo
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.nio.file.Path
import kotlin.io.path.exists

/**
 * A loaded ONNX model.
 *
 * [describeSignature] exists so a swapped model can be checked before an adapter is written
 * against it — input and output names vary between models and between versions of the same model.
 */
class OnnxModelSession(modelPath: Path) : AutoCloseable {

    init {
        require(modelPath.exists()) {
            "ONNX model not found: ${modelPath.toAbsolutePath()} — run ./tools/fetch-models.sh"
        }
    }

    private val environment: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession =
        environment.createSession(modelPath.toAbsolutePath().toString(), OrtSession.SessionOptions())

    val inputNames: List<String> get() = session.inputNames.toList()
    val outputNames: List<String> get() = session.outputNames.toList()

    /** Per-tensor shape/type info, keyed by name — lets a test pin down a model's contract. */
    val inputInfo: Map<String, NodeInfo> get() = session.inputInfo
    val outputInfo: Map<String, NodeInfo> get() = session.outputInfo

    /** Human-readable dump of every input and output, with shapes. */
    fun describeSignature(): String = buildString {
        appendLine("inputs:")
        session.inputInfo.forEach { (name, info) -> appendLine("  $name: ${info.info}") }
        appendLine("outputs:")
        session.outputInfo.forEach { (name, info) -> appendLine("  $name: ${info.info}") }
    }

    /**
     * Builds a float tensor of the given shape. [runTensors] closes it for you.
     *
     * ONNX Runtime copies a non-direct buffer into its own direct allocation synchronously here,
     * so the caller's array is not aliased and may be reused or mutated once this returns.
     */
    fun floatTensor(data: FloatArray, shape: LongArray): OnnxTensor =
        OnnxTensor.createTensor(environment, FloatBuffer.wrap(data), shape)

    /** Runs the model, hands the result to [block], and closes every tensor afterwards. */
    fun <T> runTensors(inputs: Map<String, OnnxTensor>, block: (OrtSession.Result) -> T): T =
        try {
            session.run(inputs).use(block)
        } finally {
            inputs.values.forEach { it.close() }
        }

    override fun close() {
        session.close()
    }
}
