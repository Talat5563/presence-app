package app.presence.audio

/**
 * Every sample that reaches the engine is mono PCM at this rate, as floats in [-1.0, 1.0].
 * Spec §9.1 requires corpora at 16 kHz mono; the harness rejects anything else rather than
 * silently resampling, because resampling artefacts corrupt the spectral detail that speaker
 * embeddings depend on.
 */
const val SAMPLE_RATE_HZ: Int = 16_000

/** Duration in milliseconds to a sample count at [SAMPLE_RATE_HZ]. */
fun millisToSamples(ms: Long): Int = (ms * SAMPLE_RATE_HZ / 1_000L).toInt()

/** Sample count at [SAMPLE_RATE_HZ] to a duration in milliseconds. */
fun samplesToMillis(sampleCount: Int): Long = sampleCount.toLong() * 1_000L / SAMPLE_RATE_HZ
