package app.presence.audio

import kotlin.math.exp
import kotlin.math.ln

/**
 * Triangular mel filters, matching Kaldi's `MelBanks` so features line up with what the speaker
 * model was trained on.
 *
 * Kaldi uses `fftSize / 2` spectrum bins and excludes the Nyquist bin; reproducing that exactly
 * matters, because an off-by-one here shifts every filter slightly and quietly costs accuracy.
 */
class MelFilterBank(
    val numBins: Int = 80,
    val fftSize: Int = 512,
    val sampleRate: Int = SAMPLE_RATE_HZ,
    val lowFreqHz: Double = 20.0,
    val highFreqHz: Double = sampleRate / 2.0,
) {
    private val numFftBins = fftSize / 2
    private val weights: Array<DoubleArray>

    init {
        require(numBins > 0) { "numBins must be positive" }
        require(Fft.isPowerOfTwo(fftSize)) { "fftSize must be a power of two" }
        require(lowFreqHz < highFreqHz) { "lowFreqHz must be below highFreqHz" }

        val melLow = hzToMel(lowFreqHz)
        val melHigh = hzToMel(highFreqHz)
        val melDelta = (melHigh - melLow) / (numBins + 1)
        val hzPerBin = sampleRate.toDouble() / fftSize

        weights = Array(numBins) { bin ->
            val left = melLow + bin * melDelta
            val center = left + melDelta
            val right = center + melDelta
            DoubleArray(numFftBins) { k ->
                val mel = hzToMel(k * hzPerBin)
                when {
                    mel > left && mel <= center -> (mel - left) / (center - left)
                    mel > center && mel < right -> (right - mel) / (right - center)
                    else -> 0.0
                }
            }
        }
    }

    /** Sums [powerSpectrum] under each triangular filter. */
    fun apply(powerSpectrum: DoubleArray): DoubleArray {
        require(powerSpectrum.size >= numFftBins) {
            "power spectrum has ${powerSpectrum.size} bins, need at least $numFftBins"
        }
        return DoubleArray(numBins) { bin ->
            val filter = weights[bin]
            var energy = 0.0
            for (k in 0 until numFftBins) energy += filter[k] * powerSpectrum[k]
            energy
        }
    }

    companion object {
        fun hzToMel(hz: Double): Double = 1127.0 * ln(1.0 + hz / 700.0)
        fun melToHz(mel: Double): Double = 700.0 * (exp(mel / 1127.0) - 1.0)
    }
}
