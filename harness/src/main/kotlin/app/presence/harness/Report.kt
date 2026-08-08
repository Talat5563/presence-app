package app.presence.harness

object Report {

    fun markdown(outcomes: List<Outcome>, provenance: String): String = buildString {
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
        appendLine("## Results")
        appendLine()
        appendLine("| Slice | Threshold | FA/hour | Detection | Miss rate | Median latency | Verdict |")
        appendLine("|---|---:|---:|---:|---:|---:|---|")
        for (outcome in outcomes) {
            appendLine(
                "| ${outcome.label} | ${format(outcome.fireThreshold)} | " +
                    "${format(outcome.falseAlarmsPerHour)} | " +
                    "${percent(outcome.detectionRate)} | ${percent(outcome.missRate)} | " +
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

    private fun format(value: Number): String {
        val rounded = kotlin.math.round(value.toDouble() * 1000) / 1000
        return rounded.toString()
    }

    private fun percent(value: Double): String {
        val rounded = kotlin.math.round(value * 1000) / 10
        return "$rounded%"
    }
}
