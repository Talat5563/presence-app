package app.presence.engine

import app.presence.audio.SAMPLE_RATE_HZ
import app.presence.audio.millisToSamples
import app.presence.domain.DetectionEvent

/**
 * @param minEmbedDurationMs segments shorter than this are dropped unembedded. Published work
 *   found utterances averaging 0.92 s too short to evaluate meaningfully (spec §4.1).
 * @param embedWindowMs length of each embedding window within a longer segment
 * @param embedHopMs advance between windows; overlap means more observations per segment
 */
class DetectionEngineConfig(
    val minEmbedDurationMs: Long = 800,
    val embedWindowMs: Long = 3_000,
    val embedHopMs: Long = 1_500,
) {
    init {
        require(minEmbedDurationMs > 0) { "minEmbedDurationMs must be positive" }
        require(embedWindowMs >= minEmbedDurationMs) { "embedWindowMs must be >= minEmbedDurationMs" }
        require(embedHopMs in 1..embedWindowMs) { "embedHopMs must be in 1..embedWindowMs" }
    }
}

/**
 * The complete detection pipeline: frames in, [DetectionEvent] out.
 *
 * Knows nothing about microphones, foreground services, or screens (spec §3), which is what
 * makes the whole recognizer testable on the JVM against recorded fixtures.
 */
class DetectionEngine(
    private val vadGate: VadGate,
    private val embedder: SpeakerEmbedder,
    private val matcher: HouseholdMatcher,
    private val accumulator: EvidenceAccumulator,
    private val targetPersonId: String,
    private val config: DetectionEngineConfig = DetectionEngineConfig(),
    /**
     * Receives every scored observation before it reaches the accumulator.
     *
     * The harness uses this to cache traces so an operating-point sweep can replay accumulators
     * without re-embedding audio. Also useful for on-device diagnostics later.
     */
    private val observationListener: ((Observation) -> Unit)? = null,
) {
    private class Window(val offsetSamples: Int, val samples: FloatArray)

    /** Feeds one VAD-sized frame. Returns an event if the threshold was crossed. */
    fun accept(frame: FloatArray): DetectionEvent? = vadGate.accept(frame)?.let(::process)

    /** Processes any utterance still open at end of stream. */
    fun flush(): DetectionEvent? = vadGate.flush()?.let(::process)

    /** Clears all state. Call between independent recordings. */
    fun reset() {
        vadGate.reset()
        accumulator.reset()
    }

    private fun process(segment: SpeechSegment): DetectionEvent? {
        if (segment.durationMs < config.minEmbedDurationMs) return null

        var event: DetectionEvent? = null
        for (window in windowsOf(segment.samples)) {
            val score = matcher.score(embedder.embed(window.samples))
            val timestampMs = toMillis(segment.startSampleIndex + window.offsetSamples + window.samples.size)
            val observation = Observation(timestampMs, score.logLikelihoodRatio)
            observationListener?.invoke(observation)
            val evidence = accumulator.accept(observation)
            // Only the first fire in a segment is returned, since accept() yields at most one
            // event. Unreachable under the default 60 s cooldown, which far exceeds any segment,
            // but a caller configuring a cooldown shorter than maxSegmentMs would have a second
            // within-segment detection coalesced into the first.
            if (evidence != null && event == null) {
                event = DetectionEvent(
                    personId = targetPersonId,
                    timestampMs = timestampMs,
                    accumulatedEvidence = evidence,
                    peakSimilarity = score.targetSimilarity,
                )
            }
        }
        return event
    }

    /**
     * Splits a segment into overlapping embedding windows.
     *
     * Each window becomes its own observation rather than being averaged into one: averaging
     * would discard exactly the independent evidence the accumulator exists to sum (spec §4.1).
     */
    private fun windowsOf(samples: FloatArray): List<Window> {
        val windowSamples = millisToSamples(config.embedWindowMs)
        val hopSamples = millisToSamples(config.embedHopMs)
        if (samples.size <= windowSamples) return listOf(Window(0, samples))

        val windows = ArrayList<Window>()
        var offset = 0
        while (offset + windowSamples <= samples.size) {
            windows += Window(offset, samples.copyOfRange(offset, offset + windowSamples))
            offset += hopSamples
        }
        // Cover the tail with a final window ending at the segment boundary — but only when the
        // striding loop missed at least one hop of audio. The threshold is the hop, not the
        // minimum embed duration: overlap with the previous window is `windowSamples - uncovered`,
        // so requiring a full hop bounds the tail's overlap at exactly the same 50% that every
        // regular striding pair already has. A smaller threshold would make the last pair of
        // observations markedly more correlated than the rest, and the accumulator sums them all
        // as though they were equally independent.
        val uncovered = samples.size - (offset - hopSamples + windowSamples)
        if (uncovered >= hopSamples) {
            val start = samples.size - windowSamples
            windows += Window(start, samples.copyOfRange(start, samples.size))
        }
        return windows
    }

    private fun toMillis(sampleIndex: Long): Long = sampleIndex * 1_000L / SAMPLE_RATE_HZ
}
