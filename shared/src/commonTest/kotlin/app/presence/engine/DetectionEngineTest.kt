package app.presence.engine

import app.presence.domain.DetectionEvent
import app.presence.domain.Language
import app.presence.domain.VoiceProfile
import kotlin.test.Test
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** Always reports speech, so tests control the audio content rather than the gating. */
private class AlwaysSpeech(override val frameSize: Int = 512) : VoiceActivityDetector {
    override fun speechProbability(frame: FloatArray) = 0.99f
    override fun reset() = Unit
}

private class NeverSpeech(override val frameSize: Int = 512) : VoiceActivityDetector {
    override fun speechProbability(frame: FloatArray) = 0.01f
    override fun reset() = Unit
}

/** Maps loud audio to the target's voice and quiet audio to the parent's. */
private class FakeEmbedder : SpeakerEmbedder {
    override val dimension = 3
    override fun embed(samples: FloatArray): Embedding {
        val mean = samples.average()
        return if (mean > 0.5) {
            Embedding.ofNormalized(floatArrayOf(1f, 0f, 0f))
        } else {
            Embedding.ofNormalized(floatArrayOf(0f, 1f, 0f))
        }
    }
}

class DetectionEngineTest {

    private fun profile(id: String, role: VoiceProfile.Role, vararg values: Float) = VoiceProfile(
        personId = id,
        role = role,
        centroids = mapOf(Language.ENGLISH to Embedding.ofNormalized(floatArrayOf(*values))),
    )

    private val maya = profile("maya", VoiceProfile.Role.TARGET_CHILD, 1f, 0f, 0f)
    private val parent = profile("parent", VoiceProfile.Role.PARENT, 0f, 1f, 0f)
    private val sibling = profile("sibling", VoiceProfile.Role.SIBLING, 0f, 0f, 1f)

    private fun engine(vad: VoiceActivityDetector = AlwaysSpeech()) = DetectionEngine(
        vadGate = VadGate(vad),
        embedder = FakeEmbedder(),
        matcher = HouseholdMatcher(maya, listOf(parent, sibling)),
        accumulator = EvidenceAccumulator(),
        targetPersonId = "maya",
    )

    private fun targetFrame() = FloatArray(512) { 1.0f }
    private fun parentFrame() = FloatArray(512) { 0.1f }

    /** 200 frames is ~6.4 s, enough for two capped 3 s segments. */
    private fun feed(subject: DetectionEngine, frame: () -> FloatArray, count: Int = 200): DetectionEvent? {
        var event: DetectionEvent? = null
        repeat(count) { if (event == null) event = subject.accept(frame()) else subject.accept(frame()) }
        return event
    }

    @Test
    fun `fires once enough evidence for the target accumulates`() {
        val event = feed(engine(), ::targetFrame)

        val fired = assertNotNull(event, "expected a detection after ~6 s of the target's voice")
        assertTrue(fired.personId == "maya")
        assertTrue(fired.accumulatedEvidence >= 6f)
    }

    @Test
    fun `does not fire for a cohort member's voice`() {
        assertNull(feed(engine(), ::parentFrame))
    }

    @Test
    fun `does not fire when there is no speech`() {
        assertNull(feed(engine(NeverSpeech()), ::targetFrame))
    }

    @Test
    fun `ignores a segment too short to embed`() {
        val subject = engine()

        // 20 frames is 640 ms — above the 400 ms VAD floor, below the 800 ms embed floor.
        repeat(20) { subject.accept(targetFrame()) }

        assertNull(subject.flush())
    }

    @Test
    fun `flush emits a detection completed by the final segment`() {
        val subject = engine()
        var duringStream: DetectionEvent? = null

        // 140 frames is ~4.5 s: one capped 3 s segment plus a 1.5 s tail left open.
        repeat(140) { if (duringStream == null) duringStream = subject.accept(targetFrame()) }

        val flushed = subject.flush()
        assertTrue(duringStream != null || flushed != null, "the stream should have produced a detection")
    }

    @Test
    fun `reset clears accumulated evidence`() {
        val subject = engine()

        repeat(100) { subject.accept(targetFrame()) }
        subject.reset()

        // A single 3 s segment on its own is not enough to reach the threshold.
        assertNull(feed(subject, ::targetFrame, count = 94))
    }

    @Test
    fun `a long segment yields several observations rather than one average`() {
        // With a 6 s cap, one segment spans two 3 s embedding windows, so the threshold is
        // reachable inside a single segment. Averaging the windows would prevent that.
        val subject = DetectionEngine(
            vadGate = VadGate(AlwaysSpeech(), VadGateConfig(maxSegmentMs = 6_000)),
            embedder = FakeEmbedder(),
            matcher = HouseholdMatcher(maya, listOf(parent, sibling)),
            accumulator = EvidenceAccumulator(),
            targetPersonId = "maya",
        )

        var event: DetectionEvent? = null
        repeat(190) { if (event == null) event = subject.accept(targetFrame()) }

        assertNotNull(event, "a single long segment should contribute multiple observations")
    }

    @Test
    fun `reports every observation to the listener`() {
        val seen = mutableListOf<MatchScore>()
        val subject = DetectionEngine(
            vadGate = VadGate(AlwaysSpeech()),
            embedder = FakeEmbedder(),
            matcher = HouseholdMatcher(maya, listOf(parent, sibling)),
            accumulator = EvidenceAccumulator(),
            targetPersonId = "maya",
            observationListener = { _, score -> seen += score },
        )

        repeat(200) { subject.accept(targetFrame()) }
        subject.flush()

        assertTrue(seen.size >= 2, "expected several observations, got ${seen.size}")
        assertTrue(seen.all { it.logLikelihoodRatio > 0f }, "target audio should score positive")
    }
}
