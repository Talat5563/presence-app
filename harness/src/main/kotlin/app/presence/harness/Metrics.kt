package app.presence.harness

/** The engine fired at this position in this recording. */
data class Detection(val recordingPath: String, val timestampMs: Long)

/**
 * Spec §4.3's three outcomes.
 *
 * @param rank best-to-worst ordering, used to pick the best verdict across a sweep. Explicit
 *   rather than relying on declaration order (`ordinal`), so reordering the enum for readability
 *   can never silently change which verdict a sweep reports as best.
 */
enum class Verdict(val rank: Int) { SHIP(0), ITERATE(1), FALL_BACK(2) }

data class EventScoring(
    val detected: Int,
    val total: Int,
    val latenciesMs: List<Long>,
)

/**
 * Results at one operating point, for one slice of the corpus.
 *
 * @param label which slice: "ENGLISH", "HEBREW", "CODE_SWITCHED", or "ALL". Spec §4.3 requires
 *   every criterion to be reported separately per language, never pooled.
 * @param negativeHours hours of audio in which a firing detection would count as a false alarm.
 *   Since every detection that matches no labelled event is a false alarm regardless of whose
 *   recording it fired in, that is all of it — including the target child's own recordings, so an
 *   intervention while the child plays happily alone is not exempted.
 */
data class Outcome(
    val label: String,
    val fireThreshold: Float,
    val falseAlarms: Int,
    val negativeHours: Double,
    val detectedEvents: Int,
    val totalEvents: Int,
    val latenciesMs: List<Long>,
) {
    /**
     * False alarms per hour of [negativeHours] exposure.
     *
     * Zero when there is no negative audio at all — there is no rate to report.
     */
    val falseAlarmsPerHour: Double
        get() = if (negativeHours <= 0.0) 0.0 else falseAlarms / negativeHours

    /**
     * Fraction of labelled events that went undetected. Zero for a corpus with no events, since
     * nothing was there to miss.
     *
     * Deliberately NOT `1 - detectionRate` in that case: [detectionRate] returns zero for an empty
     * corpus as a safety default, so a misconfigured or empty run falls back rather than reporting
     * a perfect score. The two are inconsistent only when there are no events at all, and that
     * asymmetry is the point.
     */
    val missRate: Double
        get() = if (totalEvents == 0) 0.0 else (totalEvents - detectedEvents).toDouble() / totalEvents

    /** Fraction of labelled events detected. Zero for an empty corpus, so the verdict falls back. */
    val detectionRate: Double
        get() = if (totalEvents == 0) 0.0 else detectedEvents.toDouble() / totalEvents

    val medianLatencyMs: Long?
        get() {
            if (latenciesMs.isEmpty()) return null
            val sorted = latenciesMs.sorted()
            val middle = sorted.size / 2
            return if (sorted.size % 2 == 1) sorted[middle] else (sorted[middle - 1] + sorted[middle]) / 2
        }

    /**
     * Spec §4.3, encoded so no caller can quietly apply a friendlier rule.
     *
     * Ship: ≤ 1 false alarm per 2 hours AND ≥ 80% detection.
     * Iterate: up to 2 false alarms per hour AND ≥ 50% detection.
     * Fall back: anything worse.
     */
    val verdict: Verdict
        get() = when {
            falseAlarmsPerHour <= 0.5 && detectionRate >= 0.80 -> Verdict.SHIP
            falseAlarmsPerHour <= 2.0 && detectionRate >= 0.50 -> Verdict.ITERATE
            else -> Verdict.FALL_BACK
        }
}

object Metrics {

    /** Spec §4.3: a real event must be caught within ten seconds to count. */
    const val DEFAULT_LATENCY_BUDGET_MS = 10_000L

    /**
     * Matches detections to labelled call events.
     *
     * Each event is satisfied by at most one detection and each detection satisfies at most one
     * event, earliest first — otherwise a single lucky detection could "catch" several events
     * and flatter the result.
     */
    fun scoreEvents(
        detections: List<Detection>,
        events: List<CallEvent>,
        latencyBudgetMs: Long = DEFAULT_LATENCY_BUDGET_MS,
    ): EventScoring {
        val unusedByRecording = detections
            .groupBy { it.recordingPath }
            .mapValues { (_, group) -> group.sortedBy { it.timestampMs }.toMutableList() }

        val latencies = ArrayList<Long>()
        for (event in events.sortedBy { it.startMs }) {
            val candidates = unusedByRecording[event.recordingPath] ?: continue
            val match = candidates.firstOrNull {
                it.timestampMs >= event.startMs && it.timestampMs - event.startMs <= latencyBudgetMs
            } ?: continue
            candidates.remove(match)
            latencies += match.timestampMs - event.startMs
        }

        return EventScoring(detected = latencies.size, total = events.size, latenciesMs = latencies)
    }
}
