package app.presence.engine

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class EvidenceAccumulatorTest {

    private fun accumulator(
        windowMs: Long = 20_000,
        fireThreshold: Float = 6.0f,
        cooldownMs: Long = 60_000,
    ) = EvidenceAccumulator(AccumulatorConfig(windowMs, fireThreshold, cooldownMs))

    private fun assertClose(expected: Float, actual: Float) {
        assertTrue(abs(expected - actual) < 1e-4f, "expected $expected but was $actual")
    }

    @Test
    fun `one weak observation does not fire`() {
        assertNull(accumulator().accept(Observation(0, 2.0f)))
    }

    @Test
    fun `evidence accumulates across observations until it fires`() {
        val subject = accumulator()

        assertNull(subject.accept(Observation(0, 2.0f)))
        assertNull(subject.accept(Observation(1_000, 2.0f)))
        val fired = subject.accept(Observation(2_000, 2.0f))

        assertClose(6.0f, assertNotNull(fired))
    }

    @Test
    fun `one very strong observation can fire alone`() {
        assertClose(7.0f, assertNotNull(accumulator().accept(Observation(0, 7.0f))))
    }

    @Test
    fun `observations older than the window stop counting`() {
        val subject = accumulator()

        subject.accept(Observation(0, 2.0f))
        subject.accept(Observation(1_000, 2.0f))
        // 25 s later both earlier observations are outside the 20 s window.
        val fired = subject.accept(Observation(25_000, 2.0f))

        assertNull(fired)
    }

    @Test
    fun `negative evidence cancels positive evidence`() {
        val subject = accumulator()

        subject.accept(Observation(0, 4.0f))
        subject.accept(Observation(1_000, -3.0f))
        assertNull(subject.accept(Observation(2_000, 4.0f)))

        assertClose(7.0f, assertNotNull(subject.accept(Observation(3_000, 2.0f))))
    }

    @Test
    fun `the window is cleared after firing`() {
        val subject = accumulator(cooldownMs = 0)

        subject.accept(Observation(0, 7.0f))

        assertNull(subject.accept(Observation(1_000, 2.0f)))
    }

    @Test
    fun `cooldown suppresses a second detection`() {
        val subject = accumulator()

        assertNotNull(subject.accept(Observation(0, 7.0f)))
        assertNull(subject.accept(Observation(1_000, 7.0f)))
        assertNull(subject.accept(Observation(30_000, 7.0f)))
    }

    @Test
    fun `firing resumes once the cooldown has elapsed`() {
        val subject = accumulator()

        subject.accept(Observation(0, 7.0f))

        assertNotNull(subject.accept(Observation(61_000, 7.0f)))
    }

    @Test
    fun `reset clears both the window and the cooldown`() {
        val subject = accumulator()

        subject.accept(Observation(0, 7.0f))
        subject.reset()

        assertNotNull(subject.accept(Observation(1_000, 7.0f)))
    }

    @Test
    fun `rejects a non-positive window`() {
        kotlin.test.assertFailsWith<IllegalArgumentException> {
            AccumulatorConfig(windowMs = 0)
        }
    }
}
