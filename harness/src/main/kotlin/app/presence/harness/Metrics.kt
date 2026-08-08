package app.presence.harness

/** The engine fired at this position in this recording. */
data class Detection(val recordingPath: String, val timestampMs: Long)

/** Spec §4.3's three outcomes. */
enum class Verdict { SHIP, ITERATE, FALL_BACK }

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
 * @param negativeHours hours of audio containing no target speech; the denominator for false
 *   alarms.
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
    val falseAlarmsPerHour: Double
        get() = if (negativeHours <= 0.0) 0.0 else falseAlarms / negativeHours

    val missRate: Double
        get() = if (totalEvents == 0) 0.0 else (totalEvents - detectedEvents).toDouble() / totalEvents

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
