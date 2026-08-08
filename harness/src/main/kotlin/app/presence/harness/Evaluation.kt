package app.presence.harness

import app.presence.engine.AccumulatorConfig
import app.presence.engine.EvidenceAccumulator
import app.presence.engine.Observation

/**
 * One recording's scored observations, cached so a threshold sweep can replay accumulators
 * without re-embedding audio — the difference between a sweep taking seconds and taking hours.
 */
data class Trace(
    val recording: Recording,
    val observations: List<Observation>,
    val durationSeconds: Double,
)

/**
 * How the corpus divides. Stated explicitly because a leak between enrollment and evaluation
 * inflates results silently (spec §9.1).
 */
data class CorpusSplit(
    val enrollment: List<Recording>,
    val positives: List<Recording>,
    val negatives: List<Recording>,
)

object Evaluation {

    /**
     * Read passages enrol; the target's calling and spontaneous audio are positives; everything
     * else is a negative. Enrollment material never appears in evaluation.
     */
    fun splitCorpus(recordings: List<Recording>): CorpusSplit {
        val enrollment = recordings.filter {
            it.role.isHouseholdMember && it.contentType == ContentType.READ
        }
        val remainder = recordings - enrollment.toSet()
        return CorpusSplit(
            enrollment = enrollment,
            positives = remainder.filter { it.role == RecordingRole.TARGET_CHILD },
            negatives = remainder.filter { it.role != RecordingRole.TARGET_CHILD },
        )
    }

    /**
     * Replays a fresh accumulator over cached traces at one threshold.
     *
     * @param filter restricts which recordings count, for per-language slices. Spec §4.3 requires
     *   English, Hebrew and code-switched speech to be reported separately.
     */
    fun replay(
        traces: List<Trace>,
        events: List<CallEvent>,
        threshold: Float,
        label: String,
        filter: (Recording) -> Boolean = { true },
    ): Outcome {
        val selected = traces.filter { filter(it.recording) }
        val selectedPaths = selected.map { it.recording.path }.toSet()

        val detections = ArrayList<Detection>()
        var exposureSeconds = 0.0

        for (trace in selected) {
            val accumulator = EvidenceAccumulator(AccumulatorConfig(fireThreshold = threshold))
            for (observation in trace.observations) {
                if (accumulator.accept(observation) != null) {
                    detections += Detection(trace.recording.path, observation.timestampMs)
                }
            }
            exposureSeconds += trace.durationSeconds
        }

        val relevantEvents = events.filter { it.recordingPath in selectedPaths }
        val scoring = Metrics.scoreEvents(detections, relevantEvents)
        // Every detection that matched no labelled event is a false alarm, wherever it fired.
        // Exempting the target child's own recordings would hide the app interrupting a parent
        // while the child plays happily alone — which is exactly a false alarm — and would also
        // omit those hours from the denominator.
        val falseAlarms = detections.size - scoring.detected

        return Outcome(
            label = label,
            fireThreshold = threshold,
            falseAlarms = falseAlarms,
            negativeHours = exposureSeconds / 3_600.0,
            detectedEvents = scoring.detected,
            totalEvents = scoring.total,
            latenciesMs = scoring.latenciesMs,
        )
    }

    /** Replays every threshold. Cheap, because the embeddings are already computed. */
    fun sweep(
        traces: List<Trace>,
        events: List<CallEvent>,
        thresholds: List<Float>,
        label: String = "ALL",
        filter: (Recording) -> Boolean = { true },
    ): List<Outcome> = thresholds.map { replay(traces, events, it, label, filter) }
}
