package app.presence.audio

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class FbankExtractorTest {

    private val extractor = FbankExtractor()

    private fun tone(frequencyHz: Double, samples: Int) = FloatArray(samples) {
        (0.5 * sin(2.0 * PI * frequencyHz * it / SAMPLE_RATE_HZ)).toFloat()
    }

    // --- mel scale ---

    @Test
    fun `mel conversion round trips`() {
        listOf(20.0, 440.0, 1_000.0, 8_000.0).forEach { hz ->
            val roundTripped = MelFilterBank.melToHz(MelFilterBank.hzToMel(hz))
            assertTrue(abs(hz - roundTripped) < 1e-6, "expected $hz, got $roundTripped")
        }
    }

    @Test
    fun `mel scale is monotonic and compresses high frequencies`() {
        val lowSpan = MelFilterBank.hzToMel(1_100.0) - MelFilterBank.hzToMel(100.0)
        val highSpan = MelFilterBank.hzToMel(7_000.0) - MelFilterBank.hzToMel(6_000.0)

        assertTrue(MelFilterBank.hzToMel(100.0) < MelFilterBank.hzToMel(1_000.0))
        assertTrue(lowSpan > highSpan, "a low-frequency kHz should span more mels than a high one")
    }

    @Test
    fun `every mel filter has some non-zero weight`() {
        val bank = MelFilterBank()
        val allOnes = DoubleArray(512 / 2 + 1) { 1.0 }

        val energies = bank.apply(allOnes)

        assertEquals(80, energies.size)
        energies.forEachIndexed { bin, energy ->
            assertTrue(energy > 0.0, "mel bin $bin has no weight at all")
        }
    }

    @Test
    fun `mel filters are ordered so a higher tone excites a higher bin`() {
        val bank = MelFilterBank()
        val lowPeak = bank.apply(spikeAt(hz = 300.0)).let { it.indices.maxBy { i -> it[i] } }
        val highPeak = bank.apply(spikeAt(hz = 3_000.0)).let { it.indices.maxBy { i -> it[i] } }

        assertTrue(lowPeak < highPeak, "300 Hz peaked at $lowPeak, 3000 Hz at $highPeak")
    }

    private fun spikeAt(hz: Double): DoubleArray {
        val bin = (hz * 512 / SAMPLE_RATE_HZ).toInt()
        return DoubleArray(512 / 2 + 1) { if (it == bin) 1.0 else 0.0 }
    }

    // --- extraction ---

    @Test
    fun `produces 80 bins per frame`() {
        val features = extractor.extract(tone(440.0, SAMPLE_RATE_HZ))

        assertTrue(features.isNotEmpty())
        features.forEach { assertEquals(80, it.size) }
    }

    @Test
    fun `frame count follows the snip-edges formula`() {
        // 16000 samples: 1 + (16000 - 400) / 160 = 98 frames.
        assertEquals(98, extractor.extract(tone(440.0, 16_000)).size)
        // Exactly one frame's worth of samples yields exactly one frame.
        assertEquals(1, extractor.extract(tone(440.0, 400)).size)
        // One shift more yields one frame more.
        assertEquals(2, extractor.extract(tone(440.0, 560)).size)
    }

    @Test
    fun `rejects audio shorter than one frame`() {
        assertFailsWith<IllegalArgumentException> { extractor.extract(tone(440.0, 399)) }
    }

    @Test
    fun `is deterministic`() {
        val samples = tone(440.0, 8_000)

        val first = extractor.extract(samples)
        val second = extractor.extract(samples.copyOf())

        first.forEachIndexed { frame, row ->
            row.forEachIndexed { bin, value ->
                assertEquals(value, second[frame][bin], "frame $frame bin $bin differs")
            }
        }
    }

    @Test
    fun `produces no NaN or infinity, even for digital silence`() {
        val features = extractor.extract(FloatArray(8_000))

        features.forEach { row ->
            row.forEach { assertTrue(it.isFinite(), "non-finite value $it in silence") }
        }
    }

    @Test
    fun `mean normalization leaves each bin centred on zero`() {
        val features = extractor.extract(tone(440.0, 16_000))

        for (bin in 0 until 80) {
            var sum = 0.0
            for (frame in features) sum += frame[bin]
            val mean = sum / features.size
            assertTrue(abs(mean) < 1e-3, "bin $bin has mean $mean after normalization")
        }
    }

    /**
     * Mean normalization is per-utterance, so a stationary tone yields bit-identical frames whose
     * every bin normalizes to exactly zero — its "peak" would be an arbitrary tie on bin 0. This
     * signal changes pitch halfway through, so early frames sit above the utterance mean in the
     * low bins and late frames sit above it in the high bins. That is the property that matters,
     * and it exercises the whole extractor including normalization.
     *
     * Raw pre-normalization peaks, measured on this filterbank: 300 Hz → bin 10, 3000 Hz → bin 52.
     */
    @Test
    fun `energy moves to higher mel bins when the pitch rises mid-utterance`() {
        val lowThenHigh = tone(300.0, 8_000) + tone(3_000.0, 8_000)

        val features = extractor.extract(lowThenHigh)

        val early = features[features.size / 4]
        val late = features[3 * features.size / 4]
        val earlyPeak = early.indices.maxBy { early[it] }
        val latePeak = late.indices.maxBy { late[it] }

        assertTrue(
            earlyPeak < latePeak,
            "early frame peaked at bin $earlyPeak, late frame at bin $latePeak",
        )
    }

    /**
     * Guards the log floor against swallowing pathologically quiet input, and in doing so pins the
     * ×32768 int16 scaling — the one constant no other test can reach.
     *
     * Mean normalization cancels any uniform gain exactly: scaling the waveform shifts every log
     * energy by the same constant, which the per-bin mean then subtracts. So the scaling is
     * observable only where the floor clips. This amplitude (2e-7, well below one 16-bit LSB) is
     * chosen to sit that far down deliberately; at realistic levels the scaling provably has no
     * effect on the output, which is why nothing else here can pin it. Noise rather than a tone,
     * because a stationary signal normalizes to zero regardless.
     */
    @Test
    fun `quiet audio still carries information rather than clipping to the log floor`() {
        val quiet = Random(11).let { rng -> FloatArray(8_000) { (rng.nextFloat() - 0.5f) * 2e-7f } }

        val features = extractor.extract(quiet)

        val distinctValues = features.flatMap { row -> row.toList() }.distinct()
        assertTrue(
            distinctValues.size > 1,
            "every value collapsed to ${distinctValues.firstOrNull()} — the log floor swallowed the signal",
        )
    }
}
