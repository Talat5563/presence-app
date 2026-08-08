package app.presence.audio

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class FftTest {

    /** The obvious, slow, independently-written transform the fast one must agree with. */
    private fun bruteForceDft(re: DoubleArray, im: DoubleArray): Pair<DoubleArray, DoubleArray> {
        val n = re.size
        val outRe = DoubleArray(n)
        val outIm = DoubleArray(n)
        for (k in 0 until n) {
            var sumRe = 0.0
            var sumIm = 0.0
            for (t in 0 until n) {
                val angle = -2.0 * PI * k * t / n
                sumRe += re[t] * cos(angle) - im[t] * sin(angle)
                sumIm += re[t] * sin(angle) + im[t] * cos(angle)
            }
            outRe[k] = sumRe
            outIm[k] = sumIm
        }
        return outRe to outIm
    }

    @Test
    fun `recognizes powers of two`() {
        listOf(1, 2, 4, 512).forEach { assertTrue(Fft.isPowerOfTwo(it), "$it is a power of two") }
        listOf(0, 3, 400, -8).forEach { assertTrue(!Fft.isPowerOfTwo(it), "$it is not") }
    }

    @Test
    fun `agrees with a brute-force DFT on random input`() {
        val random = Random(42)
        val n = 64
        val re = DoubleArray(n) { random.nextDouble(-1.0, 1.0) }
        val im = DoubleArray(n) { random.nextDouble(-1.0, 1.0) }
        val (expectedRe, expectedIm) = bruteForceDft(re, im)

        val actualRe = re.copyOf()
        val actualIm = im.copyOf()
        Fft.transform(actualRe, actualIm)

        for (k in 0 until n) {
            assertTrue(abs(expectedRe[k] - actualRe[k]) < 1e-9, "real part at $k")
            assertTrue(abs(expectedIm[k] - actualIm[k]) < 1e-9, "imaginary part at $k")
        }
    }

    @Test
    fun `transforms a constant signal to a single DC spike`() {
        val n = 8
        val re = DoubleArray(n) { 1.0 }
        val im = DoubleArray(n)

        Fft.transform(re, im)

        assertTrue(abs(8.0 - re[0]) < 1e-9, "DC bin should hold the sum, got ${re[0]}")
        for (k in 1 until n) assertTrue(abs(re[k]) < 1e-9 && abs(im[k]) < 1e-9, "bin $k should be zero")
    }

    @Test
    fun `power spectrum peaks at the frequency of a pure tone`() {
        val n = 512
        val binOfInterest = 40
        val frame = DoubleArray(n) { cos(2.0 * PI * binOfInterest * it / n) }

        val power = Fft.powerSpectrum(frame)

        assertEquals(n / 2 + 1, power.size)
        val peak = power.indices.maxBy { power[it] }
        assertEquals(binOfInterest, peak)
    }

    @Test
    fun `power spectrum is never negative`() {
        val random = Random(7)
        val power = Fft.powerSpectrum(DoubleArray(256) { random.nextDouble(-1.0, 1.0) })

        assertTrue(power.all { it >= 0.0 })
    }

    @Test
    fun `rejects a non-power-of-two size`() {
        assertFailsWith<IllegalArgumentException> { Fft.transform(DoubleArray(3), DoubleArray(3)) }
    }

    @Test
    fun `rejects mismatched real and imaginary lengths`() {
        assertFailsWith<IllegalArgumentException> { Fft.transform(DoubleArray(4), DoubleArray(8)) }
    }
}
