package app.presence.harness

import app.presence.domain.Language
import java.io.DataOutputStream
import java.nio.file.Files
import kotlin.io.path.outputStream
import kotlin.io.path.writeText
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class RecordedProbesTest {

    private fun recording(path: String) = Recording(
        path = path,
        householdId = "h1",
        speakerId = "maya",
        role = RecordingRole.TARGET_CHILD,
        language = Language.ENGLISH,
        codeSwitched = false,
        contentType = ContentType.SPONTANEOUS,
        distanceMeters = null,
        placement = "table",
        background = "quiet",
        room = "kitchen",
        date = "2026-08-01",
    )

    private fun probes(): RecordedProbes {
        val dir = Files.createTempDirectory("probes")
        DataOutputStream(dir.resolve("probs.f32").outputStream()).use { out ->
            listOf(0.1f, 0.9f, 0.8f, 0.2f, 0.3f).forEach(out::writeFloat)
        }
        dir.resolve("index.csv").writeText(
            "path,household_id,speaker_id,role,content_type,first_frame,frames\n" +
                "a.wav,h1,maya,target_child,spontaneous,0,3\n" +
                "b.wav,h1,maya,target_child,spontaneous,3,2\n",
        )
        return RecordedProbes(dir)
    }

    private val frame = FloatArray(512)

    @Test
    fun `each recording replays its own frames, from the start after a reset`() {
        val vad = probes().forRecording(recording("b.wav"))

        assertEquals(listOf(0.2f, 0.3f), List(2) { vad.speechProbability(frame) })
        vad.reset()
        assertEquals(0.2f, vad.speechProbability(frame))
    }

    @Test
    fun `feeding more frames than were recorded fails rather than reading the next recording`() {
        val vad = probes().forRecording(recording("a.wav"))
        repeat(3) { vad.speechProbability(frame) }

        assertFailsWith<IllegalStateException> { vad.speechProbability(frame) }
    }

    @Test
    fun `a recording missing from the dump is refused`() {
        assertFailsWith<IllegalStateException> { probes().forRecording(recording("c.wav")) }
    }
}
