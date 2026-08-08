package app.presence.harness

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class MetricsTest {

    private fun event(startMs: Long, path: String = "a.wav") = CallEvent(path, startMs, "maya")
    private fun detection(timestampMs: Long, path: String = "a.wav") = Detection(path, timestampMs)

    private fun outcome(
        falseAlarms: Int = 0,
        negativeHours: Double = 1.0,
        detected: Int = 10,
        total: Int = 10,
        latencies: List<Long> = listOf(1_000),
    ) = Outcome("ALL", 6f, falseAlarms, negativeHours, detected, total, latencies)

    // --- event matching ---

    @Test
    fun `a detection shortly after an event counts as detecting it`() {
        val scoring = Metrics.scoreEvents(listOf(detection(5_000)), listOf(event(3_000)))

        assertEquals(1, scoring.detected)
        assertEquals(listOf(2_000L), scoring.latenciesMs)
    }

    @Test
    fun `a detection beyond the latency budget does not count`() {
        val scoring = Metrics.scoreEvents(listOf(detection(14_000)), listOf(event(3_000)))

        assertEquals(0, scoring.detected)
        assertTrue(scoring.latenciesMs.isEmpty())
    }

    @Test
    fun `a detection exactly at the latency budget counts`() {
        val scoring = Metrics.scoreEvents(listOf(detection(13_000)), listOf(event(3_000)))

        assertEquals(1, scoring.detected)
        assertEquals(listOf(10_000L), scoring.latenciesMs)
    }

    @Test
    fun `a detection one millisecond past the budget does not count`() {
        val scoring = Metrics.scoreEvents(listOf(detection(13_001)), listOf(event(3_000)))

        assertEquals(0, scoring.detected)
    }

    @Test
    fun `a detection before the event does not count`() {
        assertEquals(0, Metrics.scoreEvents(listOf(detection(1_000)), listOf(event(3_000))).detected)
    }

    @Test
    fun `a detection in a different recording does not count`() {
        val scoring = Metrics.scoreEvents(
            listOf(detection(5_000, "other.wav")),
            listOf(event(3_000, "a.wav")),
        )

        assertEquals(0, scoring.detected)
    }

    @Test
    fun `one detection cannot satisfy two events`() {
        val scoring = Metrics.scoreEvents(
            listOf(detection(4_000)),
            listOf(event(3_000), event(3_500)),
        )

        assertEquals(1, scoring.detected)
        assertEquals(2, scoring.total)
    }

    @Test
    fun `each event is matched by its earliest eligible detection`() {
        val scoring = Metrics.scoreEvents(
            listOf(detection(4_000), detection(5_000)),
            listOf(event(3_000), event(4_500)),
        )

        assertEquals(2, scoring.detected)
        assertEquals(listOf(1_000L, 500L), scoring.latenciesMs)
    }

    @Test
    fun `no events means nothing detected and nothing missed`() {
        val scoring = Metrics.scoreEvents(listOf(detection(1_000)), emptyList())

        assertEquals(0, scoring.detected)
        assertEquals(0, scoring.total)
    }

    // --- derived rates ---

    @Test
    fun `computes false alarms per hour`() {
        assertEquals(4.0, outcome(falseAlarms = 2, negativeHours = 0.5).falseAlarmsPerHour)
    }

    @Test
    fun `false alarms per hour is zero when there is no negative audio`() {
        assertEquals(0.0, outcome(falseAlarms = 0, negativeHours = 0.0).falseAlarmsPerHour)
    }

    @Test
    fun `computes miss rate`() {
        assertEquals(0.25, outcome(detected = 6, total = 8).missRate)
    }

    @Test
    fun `miss rate is zero when there are no events`() {
        assertEquals(0.0, outcome(detected = 0, total = 0).missRate)
    }

    @Test
    fun `median latency of an even count averages the middle two`() {
        assertEquals(1_500L, outcome(latencies = listOf(1_000, 2_000)).medianLatencyMs)
    }

    @Test
    fun `median latency is null with no detections`() {
        assertNull(outcome(latencies = emptyList()).medianLatencyMs)
    }

    // --- verdict, per spec 4.3 ---

    @Test
    fun `ships at one false alarm per two hours with 80 percent detection`() {
        val result = outcome(falseAlarms = 1, negativeHours = 2.0, detected = 8, total = 10)

        assertEquals(Verdict.SHIP, result.verdict)
    }

    @Test
    fun `iterates when false alarms are too frequent even with perfect detection`() {
        val result = outcome(falseAlarms = 1, negativeHours = 1.0, detected = 10, total = 10)

        assertEquals(Verdict.ITERATE, result.verdict)
    }

    @Test
    fun `iterates when detection is below eighty percent even with no false alarms`() {
        val result = outcome(falseAlarms = 0, negativeHours = 2.0, detected = 6, total = 10)

        assertEquals(Verdict.ITERATE, result.verdict)
    }

    @Test
    fun `falls back when false alarms exceed two per hour`() {
        val result = outcome(falseAlarms = 5, negativeHours = 1.0, detected = 10, total = 10)

        assertEquals(Verdict.FALL_BACK, result.verdict)
    }

    @Test
    fun `falls back when detection is below fifty percent`() {
        val result = outcome(falseAlarms = 0, negativeHours = 2.0, detected = 4, total = 10)

        assertEquals(Verdict.FALL_BACK, result.verdict)
    }

    @Test
    fun `iterates at exactly two false alarms per hour with exactly half detected`() {
        val result = outcome(falseAlarms = 2, negativeHours = 1.0, detected = 5, total = 10)

        assertEquals(2.0, result.falseAlarmsPerHour)
        assertEquals(0.5, result.detectionRate)
        assertEquals(Verdict.ITERATE, result.verdict, "both bounds are inclusive")
    }

    @Test
    fun `falls back just past either iterate bound`() {
        val tooManyAlarms = outcome(falseAlarms = 21, negativeHours = 10.0, detected = 10, total = 10)
        assertEquals(Verdict.FALL_BACK, tooManyAlarms.verdict, "2.1 per hour is past the bound")

        val tooFewDetected = outcome(falseAlarms = 0, negativeHours = 10.0, detected = 49, total = 100)
        assertEquals(Verdict.FALL_BACK, tooFewDetected.verdict, "49% is past the bound")
    }
}
