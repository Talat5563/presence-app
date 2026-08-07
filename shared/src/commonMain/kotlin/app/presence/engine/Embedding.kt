package app.presence.engine

import kotlin.math.sqrt

/**
 * A speaker embedding, always L2-normalized.
 *
 * The constructor is private and [ofNormalized] is the only way in, so every instance in the
 * system has unit magnitude. That makes [cosine] a plain dot product and removes the class of
 * bug where an unnormalized vector yields similarities above 1.0 that then distort the
 * evidence accumulator downstream.
 */
class Embedding private constructor(
    /**
     * The unit-length components.
     *
     * `internal` rather than public: a `FloatArray` is mutable, and one stray write from outside
     * would silently break the invariant that makes [cosine] a plain dot product. A defensive copy
     * on every read would be safer still, but [cosine] runs once per profile per observation and
     * copying the vector each time is a cost this hot path should not pay.
     */
    internal val values: FloatArray,
) {

    val dimension: Int get() = values.size

    companion object {
        /** Normalizes [values] to unit length. The input array is not modified. */
        fun ofNormalized(values: FloatArray): Embedding {
            require(values.isNotEmpty()) { "an embedding must have at least one dimension" }
            var sumOfSquares = 0.0
            for (v in values) sumOfSquares += v.toDouble() * v
            val magnitude = sqrt(sumOfSquares)
            require(magnitude > 0.0) { "an embedding must have non-zero magnitude" }
            return Embedding(FloatArray(values.size) { (values[it] / magnitude).toFloat() })
        }
    }
}

/**
 * Cosine similarity in [-1, 1]. Both operands are unit vectors by construction, so this is
 * their dot product.
 */
fun cosine(a: Embedding, b: Embedding): Float {
    require(a.dimension == b.dimension) {
        "dimension mismatch: ${a.dimension} vs ${b.dimension}"
    }
    var sum = 0.0
    for (i in a.values.indices) sum += a.values[i].toDouble() * b.values[i]
    return sum.toFloat()
}

/**
 * The mean of [embeddings], re-normalized to unit length.
 *
 * This is how a [app.presence.domain.VoiceProfile] centroid is built during enrollment: no
 * training, just arithmetic over embeddings, which is why enrollment completes in seconds and
 * works identically for live and imported audio (spec §4.1).
 */
fun centroidOf(embeddings: List<Embedding>): Embedding {
    require(embeddings.isNotEmpty()) { "cannot build a centroid from zero embeddings" }
    val dimension = embeddings.first().dimension
    val accumulator = DoubleArray(dimension)
    for (embedding in embeddings) {
        require(embedding.dimension == dimension) {
            "dimension mismatch in centroid input: expected $dimension, got ${embedding.dimension}"
        }
        for (i in 0 until dimension) accumulator[i] += embedding.values[i].toDouble()
    }
    return Embedding.ofNormalized(FloatArray(dimension) { accumulator[it].toFloat() })
}
