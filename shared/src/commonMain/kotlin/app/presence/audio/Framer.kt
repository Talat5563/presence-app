package app.presence.audio

/**
 * Splits a continuous audio stream into fixed-size frames.
 *
 * Leftover samples are held between [push] calls, so the frames produced depend only on the
 * sample stream and never on how that stream happened to be chunked. The VAD model needs
 * exactly-sized frames; audio arrives in whatever sizes the capture layer feels like.
 *
 * @param frameSize samples per emitted frame
 * @param hopSize samples to advance between frames; equal to [frameSize] means no overlap
 */
class Framer(
    private val frameSize: Int,
    private val hopSize: Int = frameSize,
) {
    init {
        require(frameSize > 0) { "frameSize must be positive, was $frameSize" }
        require(hopSize in 1..frameSize) { "hopSize must be in 1..$frameSize, was $hopSize" }
    }

    private var buffer = FloatArray(0)

    /** Appends [samples] and returns every complete frame that is now available. */
    fun push(samples: FloatArray): List<FloatArray> {
        buffer = buffer + samples
        if (buffer.size < frameSize) return emptyList()

        val frames = ArrayList<FloatArray>()
        var offset = 0
        while (offset + frameSize <= buffer.size) {
            frames += buffer.copyOfRange(offset, offset + frameSize)
            offset += hopSize
        }
        buffer = buffer.copyOfRange(offset, buffer.size)
        return frames
    }

    /** Discards buffered samples. Call between independent recordings. */
    fun reset() {
        buffer = FloatArray(0)
    }
}
