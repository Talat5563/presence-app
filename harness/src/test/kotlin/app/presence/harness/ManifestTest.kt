package app.presence.harness

import app.presence.domain.Language
import app.presence.domain.VoiceProfile
import java.nio.file.Files
import kotlin.io.path.writeText
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

class ManifestTest {

    private fun tempFile(name: String, content: String) =
        Files.createTempDirectory("manifest-test").resolve(name).also { it.writeText(content) }

    private val recordingsHeader =
        "path,household_id,speaker_id,role,language,code_switched,content_type," +
            "distance_m,placement,background,room,date"

    // --- Csv ---

    @Test
    fun `parses a header and rows into maps`() {
        val rows = Csv.parse("a,b\n1,2\n3,4\n")

        assertEquals(2, rows.size)
        assertEquals("1", rows[0]["a"])
        assertEquals("4", rows[1]["b"])
    }

    @Test
    fun `ignores blank lines and trims whitespace`() {
        val rows = Csv.parse("a, b \n\n 1 ,2\n\n")

        assertEquals(1, rows.size)
        assertEquals("1", rows[0]["a"])
        assertEquals("2", rows[0]["b"])
    }

    @Test
    fun `rejects a row with the wrong column count, naming the line`() {
        val failure = assertFailsWith<IllegalArgumentException> { Csv.parse("a,b\n1,2,3\n") }

        assertTrue(failure.message!!.contains("line 2"), failure.message!!)
        assertTrue(failure.message!!.contains("3"), failure.message!!)
    }

    @Test
    fun `rejects quoting because embedded commas are unsupported`() {
        val failure = assertFailsWith<IllegalArgumentException> { Csv.parse("a,b\n\"x,y\",2\n") }

        assertTrue(failure.message!!.contains("quote"), failure.message!!)
    }

    @Test
    fun `rejects an empty document`() {
        assertFailsWith<IllegalArgumentException> { Csv.parse("") }
    }

    @Test
    fun `rejects duplicate header columns rather than silently losing one`() {
        val failure = assertFailsWith<IllegalArgumentException> { Csv.parse("a,b,a\n1,2,3\n") }

        assertTrue(failure.message!!.contains("duplicate"), failure.message!!)
        assertTrue(failure.message!!.contains("a"))
    }

    // --- Recordings ---

    @Test
    fun `reads a target child recording`() {
        val file = tempFile(
            "recordings.csv",
            "$recordingsHeader\n" +
                "audio/maya-01.wav,h1,maya,target_child,hebrew,false,calling,1.0,table,quiet,kitchen,2026-08-01\n",
        )

        val recordings = Manifest.readRecordings(file)

        assertEquals(1, recordings.size)
        val recording = recordings[0]
        assertEquals("maya", recording.speakerId)
        assertEquals(RecordingRole.TARGET_CHILD, recording.role)
        assertEquals(Language.HEBREW, recording.language)
        assertEquals(ContentType.CALLING, recording.contentType)
        assertEquals(1.0, recording.distanceMeters)
        assertTrue(!recording.codeSwitched)
    }

    @Test
    fun `reads code-switched content`() {
        val file = tempFile(
            "recordings.csv",
            "$recordingsHeader\n" +
                "a.wav,h1,maya,target_child,hebrew,true,spontaneous,1.0,table,quiet,kitchen,2026-08-01\n",
        )

        assertTrue(Manifest.readRecordings(file)[0].codeSwitched)
    }

    @Test
    fun `reads room tone with no language`() {
        val file = tempFile(
            "recordings.csv",
            "$recordingsHeader\n" +
                "tone.wav,h1,room,room_tone,,false,room_tone,,table,quiet,kitchen,2026-08-01\n",
        )

        val recording = Manifest.readRecordings(file)[0]

        assertNull(recording.language)
        assertNull(recording.distanceMeters)
        assertEquals(RecordingRole.ROOM_TONE, recording.role)
    }

    @Test
    fun `maps household roles to profile roles`() {
        assertEquals(VoiceProfile.Role.TARGET_CHILD, RecordingRole.TARGET_CHILD.toProfileRole())
        assertEquals(VoiceProfile.Role.SIBLING, RecordingRole.SIBLING.toProfileRole())
        assertEquals(VoiceProfile.Role.PARENT, RecordingRole.PARENT.toProfileRole())
        assertEquals(VoiceProfile.Role.OTHER, RecordingRole.OTHER_HOUSEHOLD.toProfileRole())
    }

    @Test
    fun `only household members are enrollable`() {
        assertTrue(RecordingRole.TARGET_CHILD.isHouseholdMember)
        assertTrue(RecordingRole.SIBLING.isHouseholdMember)
        assertTrue(RecordingRole.PARENT.isHouseholdMember)
        assertTrue(!RecordingRole.MEDIA.isHouseholdMember)
        assertTrue(!RecordingRole.STRANGER.isHouseholdMember)
        assertTrue(!RecordingRole.ROOM_TONE.isHouseholdMember)
    }

    @Test
    fun `rejects an unknown role, listing the valid ones`() {
        val file = tempFile(
            "recordings.csv",
            "$recordingsHeader\n" +
                "a.wav,h1,x,grandparent,hebrew,false,spontaneous,1.0,table,quiet,kitchen,2026-08-01\n",
        )

        val failure = assertFailsWith<IllegalArgumentException> { Manifest.readRecordings(file) }

        assertTrue(failure.message!!.contains("grandparent"))
        assertTrue(failure.message!!.contains("target_child"), "should list valid roles")
    }

    @Test
    fun `rejects a missing required column`() {
        val file = tempFile("recordings.csv", "path,household_id\na.wav,h1\n")

        val failure = assertFailsWith<IllegalArgumentException> { Manifest.readRecordings(file) }

        assertTrue(failure.message!!.contains("speaker_id"))
    }

    @Test
    fun `rejects a manifest with a header but no rows`() {
        val file = tempFile("recordings.csv", "$recordingsHeader\n")

        val failure = assertFailsWith<IllegalArgumentException> { Manifest.readRecordings(file) }

        assertTrue(failure.message!!.contains("no data rows"), failure.message!!)
    }

    @Test
    fun `rejects a manifest listing the same audio path twice`() {
        val row = "dup.wav,h1,maya,target_child,hebrew,false,spontaneous,1.0,table,quiet,kitchen,2026-08-01"
        val file = tempFile("recordings.csv", "$recordingsHeader\n$row\n$row\n")

        val failure = assertFailsWith<IllegalArgumentException> { Manifest.readRecordings(file) }

        assertTrue(failure.message!!.contains("dup.wav"), failure.message!!)
    }

    @Test
    fun `a malformed number names the file and line`() {
        val file = tempFile(
            "recordings.csv",
            "$recordingsHeader\n" +
                "a.wav,h1,maya,target_child,hebrew,false,spontaneous,abc,table,quiet,kitchen,2026-08-01\n",
        )

        val failure = assertFailsWith<IllegalArgumentException> { Manifest.readRecordings(file) }

        assertTrue(failure.message!!.contains("recordings.csv"), failure.message!!)
        assertTrue(failure.message!!.contains("line 2"), failure.message!!)
    }

    // --- Events ---

    @Test
    fun `reads call events`() {
        val file = tempFile(
            "events.csv",
            "recording_path,start_ms,target_speaker_id\naudio/maya-01.wav,4200,maya\n",
        )

        val events = Manifest.readEvents(file)

        assertEquals(1, events.size)
        assertEquals("audio/maya-01.wav", events[0].recordingPath)
        assertEquals(4200L, events[0].startMs)
        assertEquals("maya", events[0].targetSpeakerId)
    }

    @Test
    fun `rejects a negative event start`() {
        val file = tempFile(
            "events.csv",
            "recording_path,start_ms,target_speaker_id\na.wav,-1,maya\n",
        )

        assertFailsWith<IllegalArgumentException> { Manifest.readEvents(file) }
    }

    @Test
    fun `rejects an event with a blank speaker id`() {
        val file = tempFile("events.csv", "recording_path,start_ms,target_speaker_id\na.wav,100,\n")

        assertFailsWith<IllegalArgumentException> { Manifest.readEvents(file) }
    }
}
