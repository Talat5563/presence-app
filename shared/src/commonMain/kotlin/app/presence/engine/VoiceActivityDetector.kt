package app.presence.engine

/**
 * Frame-level speech probability.
 *
 * Implementations are stateful — Silero carries recurrent state between frames — so a single
 * instance must not be shared across concurrent streams, and [reset] must be called between
 * independent recordings.
 */
interface VoiceActivityDetector {

    /** Exact number of samples each frame passed to [speechProbability] must contain. */
    val frameSize: Int

    /** Returns P(speech) in [0, 1] for one frame of exactly [frameSize] samples. */
    fun speechProbability(frame: FloatArray): Float

    /** Clears recurrent state. Call between independent recordings. */
    fun reset()
}
