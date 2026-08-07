package app.presence.engine

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** Replays a fixed list of probabilities, one per frame. */
private class ScriptedVad(
    private val probabilities: List<Float>,
    override val frameSize: Int = 512,
) : VoiceActivityDetector {
    private var index = 0
    override fun speechProbability(frame: FloatArray): Float =
        probabilities.getOrElse(index++) { 0f }
    override fun reset() { index = 0 }
}

class VadGateTest {

    /** 512 samples at 16 kHz is 32 ms, so ~31 frames per second. */
    private val frame = FloatArray(512) { 0.1f }

    private fun gate(
        script: List<Float>,
        config: VadGateConfig = VadGateConfig(),
    ): Pair<VadGate, List<Float>> = VadGate(ScriptedVad(script), config) to script

    private fun runAll(script: List<Float>, config: VadGateConfig = VadGateConfig()): List<SpeechSegment> {
        val (vadGate, _) = gate(script, config)
        val segments = mutableListOf<SpeechSegment>()
        repeat(script.size) { vadGate.accept(frame)?.let { segments += it } }
        vadGate.flush()?.let { segments += it }
        return segments
    }

    @Test
    fun `emits nothing for pure silence`() {
        assertTrue(runAll(List(50) { 0.05f }).isEmpty())
    }

    @Test
    fun `emits one segment for one continuous utterance`() {
        // 40 speech frames (~1.28 s), then enough silence to close the segment.
        val script = List(40) { 0.9f } + List(20) { 0.02f }

        val segments = runAll(script)

        assertEquals(1, segments.size)
        assertTrue(segments[0].durationMs >= 1_000, "expected ~1.28 s, got ${segments[0].durationMs} ms")
    }

    @Test
    fun `does not split on a brief dip below the start threshold`() {
        // Dips to 0.4 — under the 0.5 start threshold but over the 0.35 end threshold.
        val script = List(20) { 0.9f } + List(3) { 0.4f } + List(20) { 0.9f } + List(20) { 0.02f }

        val segments = runAll(script)

        assertEquals(1, segments.size, "a brief dip must not chop the utterance in two")
    }

    @Test
    fun `splits two utterances separated by real silence`() {
        val script = List(20) { 0.9f } + List(20) { 0.01f } + List(20) { 0.9f } + List(20) { 0.01f }

        val segments = runAll(script)

        assertEquals(2, segments.size)
    }

    @Test
    fun `discards a segment shorter than the minimum`() {
        // 5 frames is 160 ms, well under the 400 ms floor.
        val script = List(5) { 0.9f } + List(20) { 0.01f }

        assertTrue(runAll(script).isEmpty())
    }

    @Test
    fun `caps a very long utterance at the maximum segment length`() {
        val config = VadGateConfig(maxSegmentMs = 1_000)
        // 100 frames is ~3.2 s of unbroken speech.
        val script = List(100) { 0.9f } + List(20) { 0.01f }

        val segments = runAll(script, config)

        assertTrue(segments.size >= 3, "expected the 3.2 s utterance to be cut into ~1 s pieces")
        segments.forEach {
            assertTrue(it.durationMs <= 1_100, "segment of ${it.durationMs} ms exceeded the cap")
        }
    }

    @Test
    fun `flush emits an utterance still open at end of stream`() {
        val vadGate = VadGate(ScriptedVad(List(40) { 0.9f }))
        repeat(40) { vadGate.accept(frame) }

        val flushed = vadGate.flush()

        assertNotNull(flushed)
        assertTrue(flushed.durationMs >= 1_000)
    }

    @Test
    fun `flush returns null when no utterance is open`() {
        val vadGate = VadGate(ScriptedVad(List(10) { 0.01f }))
        repeat(10) { vadGate.accept(frame) }

        assertNull(vadGate.flush())
    }

    @Test
    fun `reports where in the stream each segment started`() {
        val script = List(20) { 0.01f } + List(40) { 0.9f } + List(20) { 0.01f }

        val segments = runAll(script)

        assertEquals(1, segments.size)
        // Speech begins at frame 20 => 20 * 512 = 10240 samples => 640 ms.
        assertEquals(10_240L, segments[0].startSampleIndex)
    }

    @Test
    fun `reset clears an in-progress utterance`() {
        val vadGate = VadGate(ScriptedVad(List(60) { 0.9f }))
        repeat(30) { vadGate.accept(frame) }

        vadGate.reset()

        assertNull(vadGate.flush())
    }
}
