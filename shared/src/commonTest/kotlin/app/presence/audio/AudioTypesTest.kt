package app.presence.audio

import kotlin.test.Test
import kotlin.test.assertEquals

class AudioTypesTest {

    @Test
    fun `sample rate is 16 kHz`() {
        assertEquals(16_000, SAMPLE_RATE_HZ)
    }

    @Test
    fun `converts milliseconds to samples`() {
        assertEquals(16_000, millisToSamples(1_000))
        assertEquals(512, millisToSamples(32))
        assertEquals(0, millisToSamples(0))
    }

    @Test
    fun `converts samples to milliseconds`() {
        assertEquals(1_000L, samplesToMillis(16_000))
        assertEquals(32L, samplesToMillis(512))
    }

    @Test
    fun `round trips a three second duration`() {
        assertEquals(3_000L, samplesToMillis(millisToSamples(3_000)))
    }
}
