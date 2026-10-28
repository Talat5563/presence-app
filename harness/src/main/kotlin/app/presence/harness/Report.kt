package app.presence.harness

object Report {

    fun markdown(outcomes: List<Outcome>, provenance: String, warnings: List<String> = emptyList()): String =
        buildString {
            appendLine("# Phase 0 accuracy report")
            appendLine()
            appendLine("Go / no-go criteria are spec §4.3. **Both languages must clear every criterion")
            appendLine("independently** — a system that passes in English and fails in Hebrew has not passed.")
            appendLine()
            appendLine("## Models")
            appendLine()
            appendLine("```")
            appendLine(provenance.trim())
            appendLine("```")
            appendLine()
            if (warnings.isNotEmpty()) {
                appendLine("## Warnings")
                appendLine()
                appendLine("These affect how much the numbers below can be trusted.")
                appendLine()
                for (warning in warnings) appendLine("- $warning")
                appendLine()
            }
            appendLine("## Results")
            appendLine()
            appendLine(
                "| Slice | Threshold | False alarms | Negative hours | FA/hour | Detection | " +
                    "Total events | Miss rate | Median latency | Verdict |",
            )
            appendLine("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
            for (outcome in outcomes) {
                appendLine(
                    "| ${outcome.label} | ${format(outcome.fireThreshold)} | ${outcome.falseAlarms} | " +
                        "${format(outcome.negativeHours)} | ${format(outcome.falseAlarmsPerHour)} | " +
                        "${percent(outcome.detectionRate)} | ${outcome.totalEvents} | " +
                        "${percent(outcome.missRate)} | " +
                        "${outcome.medianLatencyMs?.let { "$it ms" } ?: "—"} | ${outcome.verdict} |",
                )
            }
            appendLine()
            appendLine("## Reading this")
            appendLine()
            appendLine("- **SHIP** — ≤ 1 false alarm per 2 hours and ≥ 80% detection within 10 s.")
            appendLine("- **ITERATE** — up to 2 false alarms per hour and ≥ 50% detection.")
            appendLine("- **FALL_BACK** — worse. The fallback is the wake-phrase channel alone (spec §4.3).")
            appendLine()
            appendLine("EER is absent deliberately: error costs here are asymmetric, so the equal-error")
            appendLine("point is not the operating point we care about (spec §4.1).")
        }

    fun csv(outcomes: List<Outcome>): String = buildString {
        appendLine(
            "label,threshold,false_alarms,negative_hours,false_alarms_per_hour," +
                "detected_events,total_events,detection_rate,miss_rate,median_latency_ms,verdict",
        )
        for (o in outcomes) {
            appendLine(
                "${o.label},${o.fireThreshold},${o.falseAlarms},${format(o.negativeHours)}," +
                    "${format(o.falseAlarmsPerHour)},${o.detectedEvents},${o.totalEvents}," +
                    "${format(o.detectionRate)},${format(o.missRate)},${o.medianLatencyMs ?: ""},${o.verdict}",
            )
        }
    }

    /**
     * Every scored observation with the full [app.presence.engine.MatchScore] behind it, one row
     * each. A verdict says *that* the system fails; this says *where* — which speakers produce
     * positive evidence, how close the target sits to its own centroid, and which gate removed it.
     */
    fun observationsCsv(traces: List<Trace>): String = buildString {
        appendLine(
            "path,household_id,speaker_id,role,language,timestamp_ms,target_similarity," +
                "best_non_target_similarity,best_non_target_id,cohort_z,raw_llr,llr",
        )
        for (trace in traces) {
            val r = trace.recording
            for (timed in trace.observations) {
                val s = timed.score
                appendLine(
                    "${r.path},${r.householdId},${r.speakerId},${r.role.name.lowercase()}," +
                        "${r.language?.name?.lowercase().orEmpty()},${timed.timestampMs}," +
                        "${s.targetSimilarity},${s.bestNonTargetSimilarity},${s.bestNonTargetPersonId}," +
                        "${s.cohortZ},${s.rawLogLikelihoodRatio},${s.logLikelihoodRatio}",
                )
            }
        }
    }

    private fun format(value: Number): String {
        val rounded = kotlin.math.round(value.toDouble() * 1000) / 1000
        return rounded.toString()
    }

    private fun percent(value: Double): String {
        val rounded = kotlin.math.round(value * 1000) / 10
        return "$rounded%"
    }
}
