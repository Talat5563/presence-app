package app.presence.audio

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/** Iterative in-place radix-2 Cooley-Tukey FFT. */
object Fft {

    fun isPowerOfTwo(n: Int): Boolean = n > 0 && (n and (n - 1)) == 0

    /** Transforms [re] and [im] in place. Both must be the same power-of-two length. */
    fun transform(re: DoubleArray, im: DoubleArray) {
        require(re.size == im.size) {
            "real and imaginary parts must be the same length: ${re.size} vs ${im.size}"
        }
        val n = re.size
        require(isPowerOfTwo(n)) { "FFT size must be a power of two, was $n" }

        // Bit-reversal permutation.
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j xor bit
            if (i < j) {
                val tempRe = re[i]; re[i] = re[j]; re[j] = tempRe
                val tempIm = im[i]; im[i] = im[j]; im[j] = tempIm
            }
        }

        var length = 2
        while (length <= n) {
            val angle = -2.0 * PI / length
            val stepRe = cos(angle)
            val stepIm = sin(angle)
            var block = 0
            while (block < n) {
                var twiddleRe = 1.0
                var twiddleIm = 0.0
                val half = length / 2
                for (k in 0 until half) {
                    val upperRe = re[block + k]
                    val upperIm = im[block + k]
                    val lowerRe = re[block + k + half] * twiddleRe - im[block + k + half] * twiddleIm
                    val lowerIm = re[block + k + half] * twiddleIm + im[block + k + half] * twiddleRe
                    re[block + k] = upperRe + lowerRe
                    im[block + k] = upperIm + lowerIm
                    re[block + k + half] = upperRe - lowerRe
                    im[block + k + half] = upperIm - lowerIm
                    val nextTwiddleRe = twiddleRe * stepRe - twiddleIm * stepIm
                    twiddleIm = twiddleRe * stepIm + twiddleIm * stepRe
                    twiddleRe = nextTwiddleRe
                }
                block += length
            }
            length = length shl 1
        }
    }

    /** |X[k]|² for k in 0..n/2 inclusive. [frame] is not modified. */
    fun powerSpectrum(frame: DoubleArray): DoubleArray {
        val re = frame.copyOf()
        val im = DoubleArray(frame.size)
        transform(re, im)
        return DoubleArray(frame.size / 2 + 1) { k -> re[k] * re[k] + im[k] * im[k] }
    }
}
