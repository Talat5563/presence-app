package app.presence.harness

import java.nio.file.Files
import java.time.LocalDate
import kotlin.io.path.writeText
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ChildesPlanTest {

    private val participants = listOf(
        ChatParticipant("CHI", "Ross", "Target_Child"),
        ChatParticipant("MAR", "Mark", "Brother"),
        ChatParticipant("MOT", null, "Mother"),
        ChatParticipant("INV", "Kim", "Investigator"),
    ).associateBy { it.code }

    private fun session(id: String, day: Int, vararg utterances: ChatUtterance) = ChildesSession(
        id,
        ChatTranscript(
            mediaName = id,
            languages = listOf("eng"),
            date = LocalDate.of(1980, 1, day),
            participants = participants,
            utterances = utterances.toList(),
            unalignedUtterances = 0,
        ),
    )

    private fun u(code: String, startMs: Long, endMs: Long) = ChatUtterance(code, startMs, endMs)

    /** 70 s of Ross, plus his family, on day 1 — enough to enrol on that session alone. */
    private val day1 = session(
        "s1", 1,
        u("CHI", 0, 35_000), u("CHI", 36_000, 71_000),
        u("MOT", 80_000, 90_000), u("MAR", 91_000, 101_000), u("INV", 102_000, 110_000),
    )

    /** Two target episodes separated by 20 s of his silence, plus Mark and the mother. */
    private val day2 = session(
        "s2", 2,
        u("CHI", 0, 3_000), u("CHI", 3_500, 6_000),
        u("MOT", 6_500, 9_000),
        u("CHI", 26_000, 40_000),
        u("MAR", 41_000, 45_000),
    )

    private fun plan(vararg sessions: ChildesSession, options: ImportOptions = ImportOptions()) =
        ChildesPlanner.plan(listOf(HouseholdSource("h1", "Ross", sessions.toList())), options)

    @Test
    fun `the earliest sessions are reserved whole for enrollment`() {
        val files = plan(day2, day1).files

        val enrollment = files.filter { it.recording.contentType == ContentType.ENROLLMENT }
        assertEquals(setOf("h1-ross", "h1-mark", "h1-mot"), enrollment.map { it.recording.speakerId }.toSet())
        assertTrue(enrollment.all { file -> file.pieces.all { it.sessionId == "s1" } })
        assertTrue(files.filter { it.recording.contentType != ContentType.ENROLLMENT }.all { file -> file.pieces.all { it.sessionId == "s2" } })
    }

    @Test
    fun `strangers are never enrolled`() {
        val enrolled = plan(day1, day2).files.filter { it.recording.contentType == ContentType.ENROLLMENT }

        assertTrue(enrolled.none { it.recording.speakerId == "h1-kim" })
    }

    @Test
    fun `each target episode is its own file with one call event at its start`() {
        val episodes = plan(day1, day2).files.filter { it.event != null }

        assertEquals(2, episodes.size)
        assertTrue(episodes.all { it.recording.role == RecordingRole.TARGET_CHILD && it.event!!.startMs == 0L })
        assertTrue(episodes.all { it.event!!.recordingPath == it.recording.path })
        // The first episode stitches two utterances across a 500 ms pause.
        assertEquals(listOf(0L, 500L), episodes.first().pieces.map { it.gapBeforeMs })
    }

    @Test
    fun `an episode is cut to the latency budget`() {
        val second = plan(day1, day2).files.filter { it.event != null }[1]

        assertEquals(10_000L, second.speechMs, "a 14 s stretch keeps only its first 10 s")
    }

    @Test
    fun `everyone else becomes a negative file with no event`() {
        val negatives = plan(day1, day2).files.filter {
            it.recording.contentType == ContentType.SPONTANEOUS && it.event == null
        }

        assertEquals(
            mapOf("h1-mot" to RecordingRole.PARENT, "h1-mark" to RecordingRole.SIBLING),
            negatives.associate { it.recording.speakerId to it.recording.role },
        )
    }

    @Test
    fun `utterances overlapping another speaker are dropped`() {
        val clean = ChildesPlanner.withoutOverlaps(listOf(u("CHI", 0, 2_000), u("MOT", 1_500, 3_000), u("CHI", 4_000, 5_000)))

        assertEquals(listOf(u("CHI", 4_000, 5_000)), clean)
    }

    @Test
    fun `the target is matched by name, so a sibling coded CHI in his own session is still a sibling`() {
        val marksSession = ChildesSession(
            "s3",
            day2.transcript.copy(
                date = LocalDate.of(1980, 1, 3),
                participants = mapOf(
                    "CHI" to ChatParticipant("CHI", "Mark", "Target_Child"),
                    "ROS" to ChatParticipant("ROS", "Ross", "Brother"),
                ),
                utterances = listOf(u("CHI", 0, 5_000), u("ROS", 6_000, 9_000)),
            ),
        )

        val files = plan(day1, day2, marksSession).files.filter { file -> file.pieces.all { it.sessionId == "s3" } }

        assertEquals(
            mapOf("h1-mark" to RecordingRole.SIBLING, "h1-ross" to RecordingRole.TARGET_CHILD),
            files.associate { it.recording.speakerId to it.recording.role },
        )
    }

    @Test
    fun `a household whose sessions all go to enrollment is skipped with a warning`() {
        val plan = plan(day1)

        assertTrue(plan.files.isEmpty())
        assertTrue(plan.warnings.any { it.startsWith("SKIP household h1") })
    }

    @Test
    fun `stitching inserts silence for each gap`() {
        val session = FloatArray(16_000 * 3) { 0.5f }
        val pieces = listOf(Piece("s", 0, 1_000, 0), Piece("s", 2_000, 2_500, 250))

        val stitched = stitchAudio(pieces) { session }

        assertEquals(16_000 + 4_000 + 8_000, stitched.size)
        assertEquals(0f, stitched[16_000])
        assertEquals(0.5f, stitched.last())
    }

    @Test
    fun `stitching clips a bullet that overshoots the media`() {
        val stitched = stitchAudio(listOf(Piece("s", 500, 9_000, 0))) { FloatArray(16_000) }

        assertEquals(8_000, stitched.size)
    }

    @Test
    fun `the written manifest loads through the harness's strict reader`() {
        val plan = plan(day1, day2)
        val dir = Files.createTempDirectory("childes")
        dir.resolve("recordings.csv").writeText(recordingsCsv(plan.files.map { it.recording }))
        dir.resolve("events.csv").writeText(eventsCsv(plan.files.mapNotNull { it.event }))

        val recordings = Manifest.readRecordings(dir.resolve("recordings.csv"))
        val events = Manifest.readEvents(dir.resolve("events.csv"))
        validateEvents(recordings, events)

        assertEquals(plan.files.map { it.recording }, recordings)
        assertContentEquals(plan.files.mapNotNull { it.event }, events)
    }

    @Test
    fun `media is found beside its transcript before anywhere else`() {
        val root = Files.createTempDirectory("media")
        val beside = root.resolve("Ross/010106a.mp3")
        val elsewhere = root.resolve("Mark/010106a.mp3")

        assertEquals(beside, findMedia(listOf(elsewhere, beside), root.resolve("Ross"), "010106a"))
        assertEquals(null, findMedia(listOf(elsewhere, beside), root.resolve("Gone"), "010106a"), "ambiguous elsewhere")
    }

    @Test
    fun `speech by a speaker of unknown identity goes into no file`() {
        val withUnknown = day2.copy(
            transcript = day2.transcript.copy(
                participants = participants + ("PAR0" to ChatParticipant("PAR0", null, "Participant")),
                utterances = day2.transcript.utterances + u("PAR0", 50_000, 60_000),
            ),
        )

        val files = plan(day1, withUnknown).files

        assertTrue(files.none { it.recording.speakerId == "h1-par0" })
        assertTrue(files.flatMap { it.pieces }.none { it.startMs == 50_000L })
    }

    @Test
    fun `nameless children are told apart by code, not merged by role`() {
        val nameless = mapOf(
            "CHI" to ChatParticipant("CHI", null, "Target_Child"),
            "MAR" to ChatParticipant("MAR", null, "Target_Child"),
        )
        fun anon(id: String, day: Int, vararg utterances: ChatUtterance) =
            session(id, day, *utterances).let { it.copy(transcript = it.transcript.copy(participants = nameless)) }

        val files = ChildesPlanner.plan(
            listOf(
                HouseholdSource(
                    "h1", "CHI",
                    listOf(anon("s1", 1, u("CHI", 0, 70_000), u("MAR", 71_000, 80_000)), anon("s2", 2, u("CHI", 0, 5_000), u("MAR", 6_000, 9_000))),
                ),
            ),
        ).files

        assertEquals(
            mapOf("h1-chi" to RecordingRole.TARGET_CHILD, "h1-mar" to RecordingRole.SIBLING),
            files.associate { it.recording.speakerId to it.recording.role },
        )
    }

    @Test
    fun `enrollment continues until a parent missing from the first session is enrolled too`() {
        val dadFirst = session("s1", 1, u("CHI", 0, 70_000), u("FAT", 71_000, 140_000))
        val momLater = session("s2", 2, u("CHI", 0, 5_000), u("MOT", 6_000, 76_000))
        val rest = (3..6).map { session("s$it", it, u("CHI", 0, 5_000), u("MOT", 6_000, 9_000)) }
        val withFather = participants + ("FAT" to ChatParticipant("FAT", null, "Father"))
        val sessions = (listOf(dadFirst, momLater) + rest).map { it.copy(transcript = it.transcript.copy(participants = withFather)) }

        val enrolled = plan(*sessions.toTypedArray()).files.filter { it.recording.contentType == ContentType.ENROLLMENT }

        assertEquals(setOf("h1-ross", "h1-fat", "h1-mot"), enrolled.map { it.recording.speakerId }.toSet())
    }

    @Test
    fun `enrollment does not wait for a rarely seen grandparent`() {
        val withGrandma = participants + ("GRA" to ChatParticipant("GRA", null, "Grandmother"))
        val sessions = (1..6).map { day ->
            val grandmaVisits = if (day == 6) arrayOf(u("GRA", 80_000, 85_000)) else emptyArray()
            session("s$day", day, u("CHI", 0, 70_000), u("MOT", 71_000, 79_000 + 60_000), *grandmaVisits)
        }.map { it.copy(transcript = it.transcript.copy(participants = withGrandma)) }

        val plan = plan(*sessions.toTypedArray())
        val enrollmentSessions = plan.files.filter { it.recording.contentType == ContentType.ENROLLMENT }
            .flatMap { file -> file.pieces.map { it.sessionId } }.toSet()

        assertEquals(setOf("s1"), enrollmentSessions)
    }

    @Test
    fun `enrollment audio per speaker is capped`() {
        val talkative = (1..4).map { day -> session("s$day", day, *Array(10) { u("CHI", it * 70_000L, it * 70_000L + 60_000) }) }

        val enrolled = plan(*talkative.toTypedArray(), options = ImportOptions(enrollSeconds = 600.0, maxEnrollSecondsPerSpeaker = 120.0))
            .files.single { it.recording.contentType == ContentType.ENROLLMENT }

        assertTrue(enrolled.speechMs in 120_000L..180_000L, "got ${enrolled.speechMs}")
    }
}
