package app.presence.engine

/**
 * Turns speech audio into a speaker embedding.
 *
 * An interface rather than an `expect` declaration so tests can substitute a fake — the whole
 * point of keeping the engine in common code. Production implementations wrap ONNX Runtime.
 */
interface SpeakerEmbedder {

    /** Dimensionality of the embeddings this implementation produces. */
    val dimension: Int

    /** Embeds mono 16 kHz samples. The result is L2-normalized by [Embedding.ofNormalized]. */
    fun embed(samples: FloatArray): Embedding
}
