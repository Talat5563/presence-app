package app.presence.engine

import app.presence.audio.samplesToMillis

/** A contiguous run of speech, with where it began in the stream. */
class SpeechSegment(
    val samples: FloatArray,
    val startSampleIndex: Long,
) {
    val durationMs: Long get() = samplesToMillis(samples.size)
}

/**
 * @param startThreshold probability at which a new utterance begins
 * @param endThreshold probability below which frames count toward closing it; deliberately
 *   lower than [startThreshold] so a brief dip mid-word does not chop an utterance in half
 * @param hangoverFrames consecutive sub-[endThreshold] frames required to close an utterance
 * @param minSegmentMs segments shorter than this are discarded as noise
 * @param maxSegmentMs an unbroken utterance is cut at this length so evidence flows steadily
 *   into the accumulator rather than arriving in one late lump
 */
class VadGateConfig(
    val startThreshold: Float = 0.5f,
    val endThreshold: Float = 0.35f,
    val hangoverFrames: Int = 8,
    val minSegmentMs: Long = 400,
    val maxSegmentMs: Long = 3_000,
) {
    init {
        require(startThreshold in 0f..1f) { "startThreshold must be in 0..1" }
        require(endThreshold in 0f..startThreshold) { "endThreshold must be in 0..startThreshold" }
        require(hangoverFrames > 0) { "hangoverFrames must be positive" }
        require(minSegmentMs > 0) { "minSegmentMs must be positive" }
        require(maxSegmentMs >= minSegmentMs) { "maxSegmentMs must be >= minSegmentMs" }
    }
}

/**
 * Turns a frame-by-frame speech probability into discrete speech segments.
 *
 * This is the gate that makes battery cost survivable: the expensive embedding model only wakes
 * for actual speech, roughly a 5% duty cycle in a normal room (spec §4).
 */
class VadGate(
    private val vad: VoiceActivityDetector,
    private val config: VadGateConfig = VadGateConfig(),
) {
    private val current = ArrayList<FloatArray>()
    private var currentSampleCount = 0
    private var inSpeech = false
    private var silentFrames = 0
    private var consumedSamples = 0L
    private var segmentStart = 0L

    /** Feeds one frame. Returns a segment if one completed on this frame. */
    fun accept(frame: FloatArray): SpeechSegment? {
        val probability = vad.speechProbability(frame)
        var completed: SpeechSegment? = null

        if (!inSpeech) {
            if (probability >= config.startThreshold) {
                inSpeech = true
                silentFrames = 0
                segmentStart = consumedSamples
                current.clear()
                currentSampleCount = 0
                append(frame)
            }
        } else {
            if (probability < config.endThreshold) {
                // Sub-threshold frames count toward the hangover close but must not be written
                // into the segment: padding the buffer with trailing silence would let hangover
                // alone push a too-short utterance over minSegmentMs.
                silentFrames++
            } else {
                silentFrames = 0
                append(frame)
            }
            val reachedCap = samplesToMillis(currentSampleCount) >= config.maxSegmentMs
            if (silentFrames >= config.hangoverFrames || reachedCap) {
                completed = close()
            }
        }

        consumedSamples += frame.size
        return completed
    }

    /** Emits an utterance still open at end of stream, if it is long enough. */
    fun flush(): SpeechSegment? = if (inSpeech) close() else null

    /** Discards any in-progress utterance and the stream position. */
    fun reset() {
        current.clear()
        currentSampleCount = 0
        inSpeech = false
        silentFrames = 0
        consumedSamples = 0L
        segmentStart = 0L
        vad.reset()
    }

    private fun append(frame: FloatArray) {
        current += frame
        currentSampleCount += frame.size
    }

    private fun close(): SpeechSegment? {
        val merged = FloatArray(currentSampleCount)
        var offset = 0
        for (chunk in current) {
            chunk.copyInto(merged, offset)
            offset += chunk.size
        }
        val start = segmentStart
        current.clear()
        currentSampleCount = 0
        inSpeech = false
        silentFrames = 0
        return if (samplesToMillis(merged.size) >= config.minSegmentMs) {
            SpeechSegment(merged, start)
        } else {
            null
        }
    }
}
