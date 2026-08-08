package app.presence.audio

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.max

/**
 * 80-bin log-mel filterbank features, matching what the speaker model expects
 * (`feats [B, T, 80]`).
 *
 * Every constant here mirrors the configuration the model was trained with. A mismatch does not
 * throw — it silently degrades accuracy, which is the worst possible failure mode for this project
 * because it looks like a bad model rather than a bad front end.
 */
class FbankExtractor(
    private val numMelBins: Int = 80,
    frameLengthMs: Int = 25,
    frameShiftMs: Int = 10,
    private val sampleRate: Int = SAMPLE_RATE_HZ,
    private val preEmphasis: Double = 0.97,
) {
    private companion object {
        /** FLT_EPSILON, matching Kaldi's log floor. */
        const val LOG_FLOOR = 1.19209290e-7

        /** Kaldi operates on int16-range values; omitting this shifts every log energy. */
        const val INT16_SCALE = 32768.0
    }

    private val frameLength = sampleRate * frameLengthMs / 1_000
    private val frameShift = sampleRate * frameShiftMs / 1_000
    private val fftSize = generateSequence(1) { it * 2 }.first { it >= frameLength }

    /** Hamming. WeSpeaker trains with this, not Kaldi's default povey window. */
    private val window = DoubleArray(frameLength) {
        0.54 - 0.46 * cos(2.0 * PI * it / (frameLength - 1))
    }

    private val melBank = MelFilterBank(numMelBins, fftSize, sampleRate)

    /** Returns `[frames][numMelBins]`, mean-normalized per bin across time. */
    fun extract(samples: FloatArray): Array<FloatArray> {
        require(samples.size >= frameLength) {
            "need at least $frameLength samples for one frame, got ${samples.size}"
        }

        val frameCount = 1 + (samples.size - frameLength) / frameShift
        val features = Array(frameCount) { FloatArray(numMelBins) }
        val buffer = DoubleArray(fftSize)

        for (frame in 0 until frameCount) {
            val offset = frame * frameShift

            for (i in 0 until frameLength) buffer[i] = samples[offset + i] * INT16_SCALE
            for (i in frameLength until fftSize) buffer[i] = 0.0

            // Remove the frame's DC offset.
            var sum = 0.0
            for (i in 0 until frameLength) sum += buffer[i]
            val mean = sum / frameLength
            for (i in 0 until frameLength) buffer[i] -= mean

            // Pre-emphasis, applied descending so each sample sees its untouched predecessor.
            for (i in frameLength - 1 downTo 1) buffer[i] -= preEmphasis * buffer[i - 1]
            buffer[0] -= preEmphasis * buffer[0]

            for (i in 0 until frameLength) buffer[i] *= window[i]

            val energies = melBank.apply(Fft.powerSpectrum(buffer))
            for (bin in 0 until numMelBins) {
                features[frame][bin] = ln(max(energies[bin], LOG_FLOOR)).toFloat()
            }
        }

        subtractMeanOverTime(features)
        return features
    }

    /** Mean normalization per bin. Mean only — the model was trained without variance norm. */
    private fun subtractMeanOverTime(features: Array<FloatArray>) {
        for (bin in 0 until numMelBins) {
            var sum = 0.0
            for (frame in features) sum += frame[bin]
            val mean = (sum / features.size).toFloat()
            for (frame in features) frame[bin] -= mean
        }
    }
}
