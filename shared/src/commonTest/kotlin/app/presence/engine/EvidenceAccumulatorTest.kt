package app.presence.engine

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
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

    private fun assertClose(expected: Float, actual: Firing) = assertClose(expected, actual.evidence)

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

    @Test
    fun `rejects an observation that goes backwards in time`() {
        val subject = accumulator()
        subject.accept(Observation(5_000, 1.0f))

        assertFailsWith<IllegalArgumentException> { subject.accept(Observation(4_999, 1.0f)) }
    }

    @Test
    fun `an observation exactly one window old still counts`() {
        val subject = accumulator()

        subject.accept(Observation(0, 4.0f))

        assertClose(6.0f, assertNotNull(subject.accept(Observation(20_000, 2.0f))))
    }

    @Test
    fun `an observation one millisecond past the window does not count`() {
        val subject = accumulator()

        subject.accept(Observation(0, 4.0f))

        assertNull(subject.accept(Observation(20_001, 2.0f)))
    }

    @Test
    fun `evidence arriving during cooldown does not carry across the boundary`() {
        val subject = accumulator()
        assertNotNull(subject.accept(Observation(0, 7.0f)))

        // Suppressed, and deliberately not accumulated.
        assertNull(subject.accept(Observation(30_000, 3.0f)))
        assertNull(subject.accept(Observation(50_000, 3.0f)))

        // Cooldown has exactly elapsed, so this is allowed to fire — but on its own evidence,
        // which is below the threshold. Had the suppressed observations accumulated, this
        // would have fired instead.
        assertNull(subject.accept(Observation(60_000, 3.0f)))
    }

    @Test
    fun `firing is allowed the moment the cooldown has exactly elapsed`() {
        val subject = accumulator()
        assertNotNull(subject.accept(Observation(0, 7.0f)))

        assertClose(7.0f, assertNotNull(subject.accept(Observation(60_000, 7.0f))))
    }

    @Test
    fun `the default cooldown matches the design's 45 seconds`() {
        assertEquals(45_000, AccumulatorConfig().cooldownMs)
    }

    @Test
    fun `a firing reports the observations that summed to it, oldest first`() {
        val subject = accumulator()
        val first = Observation(0, 2.0f)
        val second = Observation(1_000, 2.0f)
        val third = Observation(2_000, 2.0f)

        subject.accept(first)
        subject.accept(second)
        val fired = assertNotNull(subject.accept(third))

        assertEquals(listOf(first, second, third), fired.contributors)
    }

    @Test
    fun `contributors exclude observations that aged out of the window`() {
        val subject = accumulator()
        val stale = Observation(0, 2.0f)
        val recent = Observation(25_000, 7.0f)

        subject.accept(stale)
        val fired = assertNotNull(subject.accept(recent))

        assertEquals(listOf(recent), fired.contributors)
    }

    @Test
    fun `restarting the cooldown suppresses firing for a full cooldown from that point`() {
        val subject = accumulator()
        assertNotNull(subject.accept(Observation(0, 7.0f)))

        // Dismissed at 40 s: the cooldown that would have lapsed at 60 s now runs to 100 s.
        subject.restartCooldown(atMs = 40_000)

        assertNull(subject.accept(Observation(61_000, 7.0f)))
        assertNull(subject.accept(Observation(99_999, 7.0f)))
        assertNotNull(subject.accept(Observation(100_000, 7.0f)))
    }

    @Test
    fun `restarting the cooldown discards evidence gathered before it`() {
        val subject = accumulator(cooldownMs = 0)
        subject.accept(Observation(0, 4.0f))

        subject.restartCooldown(atMs = 1_000)

        assertNull(subject.accept(Observation(2_000, 4.0f)))
    }

    @Test
    fun `restarting the cooldown never shortens one already running`() {
        val subject = accumulator()
        assertNotNull(subject.accept(Observation(50_000, 7.0f)))

        subject.restartCooldown(atMs = 10_000)

        assertNull(subject.accept(Observation(80_000, 7.0f)))
    }
}
