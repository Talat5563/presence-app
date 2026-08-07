package app.presence.engine

import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class EmbeddingTest {

    private fun assertClose(expected: Float, actual: Float, tolerance: Float = 1e-5f) {
        assertTrue(
            abs(expected - actual) <= tolerance,
            "expected $expected but was $actual (tolerance $tolerance)",
        )
    }

    private fun magnitude(embedding: Embedding): Float {
        var sum = 0.0
        for (v in embedding.values) sum += v.toDouble() * v
        return sqrt(sum).toFloat()
    }

    @Test
    fun `normalizes on construction`() {
        val embedding = Embedding.ofNormalized(floatArrayOf(3f, 4f))

        assertClose(1f, magnitude(embedding))
        assertClose(0.6f, embedding.values[0])
        assertClose(0.8f, embedding.values[1])
    }

    @Test
    fun `reports its dimension`() {
        assertClose(192f, Embedding.ofNormalized(FloatArray(192) { 1f }).dimension.toFloat())
    }

    @Test
    fun `rejects an empty vector`() {
        assertFailsWith<IllegalArgumentException> { Embedding.ofNormalized(FloatArray(0)) }
    }

    @Test
    fun `rejects a zero vector`() {
        assertFailsWith<IllegalArgumentException> { Embedding.ofNormalized(floatArrayOf(0f, 0f)) }
    }

    @Test
    fun `cosine of a vector with itself is one`() {
        val embedding = Embedding.ofNormalized(floatArrayOf(1f, 2f, 3f))

        assertClose(1f, cosine(embedding, embedding))
    }

    @Test
    fun `cosine of orthogonal vectors is zero`() {
        val a = Embedding.ofNormalized(floatArrayOf(1f, 0f))
        val b = Embedding.ofNormalized(floatArrayOf(0f, 1f))

        assertClose(0f, cosine(a, b))
    }

    @Test
    fun `cosine of opposed vectors is minus one`() {
        val a = Embedding.ofNormalized(floatArrayOf(1f, 0f))
        val b = Embedding.ofNormalized(floatArrayOf(-1f, 0f))

        assertClose(-1f, cosine(a, b))
    }

    @Test
    fun `cosine ignores magnitude because inputs are normalized`() {
        val small = Embedding.ofNormalized(floatArrayOf(1f, 1f))
        val large = Embedding.ofNormalized(floatArrayOf(100f, 100f))

        assertClose(1f, cosine(small, large))
    }

    @Test
    fun `cosine rejects a dimension mismatch`() {
        val a = Embedding.ofNormalized(floatArrayOf(1f, 0f))
        val b = Embedding.ofNormalized(floatArrayOf(1f, 0f, 0f))

        assertFailsWith<IllegalArgumentException> { cosine(a, b) }
    }

    @Test
    fun `centroid of one embedding is that embedding`() {
        val only = Embedding.ofNormalized(floatArrayOf(1f, 2f, 3f))

        assertClose(1f, cosine(only, centroidOf(listOf(only))))
    }

    @Test
    fun `centroid sits between its inputs and stays normalized`() {
        val a = Embedding.ofNormalized(floatArrayOf(1f, 0f))
        val b = Embedding.ofNormalized(floatArrayOf(0f, 1f))

        val centroid = centroidOf(listOf(a, b))

        assertClose(1f, magnitude(centroid))
        assertClose(cosine(centroid, a), cosine(centroid, b))
        assertTrue(cosine(centroid, a) > 0.7f, "centroid should resemble both inputs")
    }

    @Test
    fun `centroid rejects an empty list`() {
        assertFailsWith<IllegalArgumentException> { centroidOf(emptyList()) }
    }

    @Test
    fun `centroid rejects mixed dimensions`() {
        val a = Embedding.ofNormalized(floatArrayOf(1f, 0f))
        val b = Embedding.ofNormalized(floatArrayOf(1f, 0f, 0f))

        assertFailsWith<IllegalArgumentException> { centroidOf(listOf(a, b)) }
    }
}
