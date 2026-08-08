package app.presence.harness

import app.presence.domain.Language
import app.presence.engine.MatchScore
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class EvaluationTest {

    private fun recording(
        path: String,
        role: RecordingRole,
        contentType: ContentType,
        speakerId: String = "maya",
        language: Language? = Language.ENGLISH,
        codeSwitched: Boolean = false,
        household: String = "h1",
    ) = Recording(
        path = path,
        householdId = household,
        speakerId = speakerId,
        role = role,
        language = language,
        codeSwitched = codeSwitched,
        contentType = contentType,
        distanceMeters = 1.0,
        placement = "table",
        background = "quiet",
        room = "kitchen",
        date = "2026-08-01",
    )

    private fun matchScore(llr: Float) = MatchScore(
        targetSimilarity = 1f,
        bestNonTargetSimilarity = 0f,
        cohortZ = 10f,
        logLikelihoodRatio = llr,
        rawLogLikelihoodRatio = llr,
    )

    private fun trace(recording: Recording, vararg llr: Pair<Long, Float>) =
        Trace(recording, llr.map { (ts, value) -> TimedScore(ts, matchScore(value)) }, durationSeconds = 60.0)

    // --- corpus split ---

    @Test
    fun `read passages become enrollment material`() {
        val split = Evaluation.splitCorpus(
            listOf(
                recording("read.wav", RecordingRole.TARGET_CHILD, ContentType.READ),
                recording("call.wav", RecordingRole.TARGET_CHILD, ContentType.CALLING),
            ),
        )

        assertEquals(listOf("read.wav"), split.enrollment.map { it.path })
    }

    @Test
    fun `target calling and spontaneous audio become positives`() {
        val split = Evaluation.splitCorpus(
            listOf(
                recording("call.wav", RecordingRole.TARGET_CHILD, ContentType.CALLING),
                recording("chat.wav", RecordingRole.TARGET_CHILD, ContentType.SPONTANEOUS),
            ),
        )

        assertEquals(setOf("call.wav", "chat.wav"), split.positives.map { it.path }.toSet())
    }

    @Test
    fun `everything that is not the target becomes a negative`() {
        val split = Evaluation.splitCorpus(
            listOf(
                recording("sib.wav", RecordingRole.SIBLING, ContentType.SPONTANEOUS, "noam"),
                recording("tv.wav", RecordingRole.MEDIA, ContentType.SPONTANEOUS, "tv1"),
                recording("tone.wav", RecordingRole.ROOM_TONE, ContentType.ROOM_TONE, "room", null),
            ),
        )

        assertEquals(3, split.negatives.size)
        assertTrue(split.positives.isEmpty())
    }

    @Test
    fun `enrollment audio is never also evaluated`() {
        val split = Evaluation.splitCorpus(
            listOf(recording("read.wav", RecordingRole.SIBLING, ContentType.READ, "noam")),
        )

        assertTrue(split.negatives.isEmpty(), "enrollment material must not leak into evaluation")
    }

    // --- replay ---

    @Test
    fun `a positive trace above threshold detects its event`() {
        val positive = recording("call.wav", RecordingRole.TARGET_CHILD, ContentType.CALLING)
        val traces = listOf(trace(positive, 1_000L to 4f, 2_000L to 4f))
        val events = listOf(CallEvent("call.wav", 500, "maya"))

        val outcome = Evaluation.replay(traces, events, threshold = 6f, label = "ALL")

        assertEquals(1, outcome.detectedEvents)
        assertEquals(0, outcome.falseAlarms)
    }

    @Test
    fun `raising the threshold turns a detection into a miss`() {
        val positive = recording("call.wav", RecordingRole.TARGET_CHILD, ContentType.CALLING)
        val traces = listOf(trace(positive, 1_000L to 4f, 2_000L to 4f))
        val events = listOf(CallEvent("call.wav", 500, "maya"))

        val outcome = Evaluation.replay(traces, events, threshold = 20f, label = "ALL")

        assertEquals(0, outcome.detectedEvents)
        assertEquals(1, outcome.totalEvents)
    }

    @Test
    fun `a firing negative trace is a false alarm`() {
        val negative = recording("tv.wav", RecordingRole.MEDIA, ContentType.SPONTANEOUS, "tv1")
        val traces = listOf(trace(negative, 1_000L to 4f, 2_000L to 4f))

        val outcome = Evaluation.replay(traces, emptyList(), threshold = 6f, label = "ALL")

        assertEquals(1, outcome.falseAlarms)
        assertTrue(outcome.negativeHours > 0.0)
    }

    @Test
    fun `the language filter slices the corpus`() {
        val hebrew = recording("he.wav", RecordingRole.MEDIA, ContentType.SPONTANEOUS, "tv1", Language.HEBREW)
        val english = recording("en.wav", RecordingRole.MEDIA, ContentType.SPONTANEOUS, "tv2", Language.ENGLISH)
        val traces = listOf(
            trace(hebrew, 1_000L to 8f),
            trace(english, 1_000L to 8f),
        )

        val outcome = Evaluation.replay(traces, emptyList(), 6f, "HEBREW") {
            it.language == Language.HEBREW && !it.codeSwitched
        }

        assertEquals(1, outcome.falseAlarms, "only the Hebrew recording should count")
    }

    @Test
    fun `the sweep produces one outcome per threshold`() {
        val negative = recording("tv.wav", RecordingRole.MEDIA, ContentType.SPONTANEOUS, "tv1")
        val traces = listOf(trace(negative, 1_000L to 4f, 2_000L to 4f))

        val outcomes = Evaluation.sweep(traces, emptyList(), listOf(2f, 6f, 20f))

        assertEquals(3, outcomes.size)
        assertEquals(listOf(2f, 6f, 20f), outcomes.map { it.fireThreshold })
        assertTrue(
            outcomes.last().falseAlarms <= outcomes.first().falseAlarms,
            "false alarms must not increase as the threshold rises",
        )
    }

    // --- report ---

    @Test
    fun `the report names every language slice and the verdict`() {
        val outcomes = listOf(
            Outcome("ENGLISH", 6f, 1, 2.0, 8, 10, listOf(1_000)),
            Outcome("HEBREW", 6f, 4, 2.0, 4, 10, listOf(3_000)),
        )

        val markdown = Report.markdown(outcomes, provenance = "vad: x\nembedding: y")

        assertTrue(markdown.contains("ENGLISH"))
        assertTrue(markdown.contains("HEBREW"))
        assertTrue(markdown.contains("SHIP"))
        assertTrue(markdown.contains("FALL_BACK"))
        assertTrue(markdown.contains("embedding: y"), "provenance must appear in the report")
    }

    @Test
    fun `the csv has a header and one row per outcome`() {
        val csv = Report.csv(listOf(Outcome("ALL", 6f, 1, 2.0, 8, 10, listOf(1_000))))

        val lines = csv.trim().lines()
        assertEquals(2, lines.size)
        assertTrue(lines[0].startsWith("label,threshold"))
    }

    @Test
    fun `the report surfaces warnings that qualify the numbers`() {
        val outcomes = listOf(Outcome("ALL", 6f, 1, 2.0, 8, 10, listOf(1_000L)))

        val markdown = Report.markdown(outcomes, "vad: x", listOf("noam: enrolled partially"))

        assertTrue(markdown.contains("Warnings"))
        assertTrue(markdown.contains("noam: enrolled partially"))
    }
}
