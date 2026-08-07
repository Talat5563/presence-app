package app.presence.audio

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class FramerTest {

    private fun ramp(n: Int, from: Int = 0) = FloatArray(n) { (from + it).toFloat() }

    @Test
    fun `emits whole frames and keeps the remainder`() {
        val framer = Framer(frameSize = 4)

        val frames = framer.push(ramp(10))

        assertEquals(2, frames.size)
        assertContentEquals(floatArrayOf(0f, 1f, 2f, 3f), frames[0])
        assertContentEquals(floatArrayOf(4f, 5f, 6f, 7f), frames[1])
    }

    @Test
    fun `remainder joins the next push so boundaries do not shift`() {
        val framer = Framer(frameSize = 4)

        framer.push(ramp(10))
        val frames = framer.push(ramp(4, from = 10))

        assertEquals(1, frames.size)
        assertContentEquals(floatArrayOf(8f, 9f, 10f, 11f), frames[0])
    }

    @Test
    fun `chunking of the input does not change the frames produced`() {
        val whole = Framer(frameSize = 4).push(ramp(16))

        val piecemeal = Framer(frameSize = 4).let { framer ->
            ramp(16).toList().chunked(3).flatMap { framer.push(it.toFloatArray()) }
        }

        assertEquals(whole.size, piecemeal.size)
        whole.forEachIndexed { i, expected -> assertContentEquals(expected, piecemeal[i]) }
    }

    @Test
    fun `overlapping hop advances by the hop size`() {
        val framer = Framer(frameSize = 4, hopSize = 2)

        val frames = framer.push(ramp(8))

        assertEquals(3, frames.size)
        assertContentEquals(floatArrayOf(0f, 1f, 2f, 3f), frames[0])
        assertContentEquals(floatArrayOf(2f, 3f, 4f, 5f), frames[1])
        assertContentEquals(floatArrayOf(4f, 5f, 6f, 7f), frames[2])
    }

    @Test
    fun `emits nothing until a full frame is available`() {
        val framer = Framer(frameSize = 4)

        assertTrue(framer.push(ramp(3)).isEmpty())
        assertEquals(1, framer.push(ramp(1, from = 3)).size)
    }

    @Test
    fun `reset discards buffered samples`() {
        val framer = Framer(frameSize = 4)

        framer.push(ramp(3))
        framer.reset()

        assertTrue(framer.push(ramp(3)).isEmpty())
    }

    @Test
    fun `rejects a non-positive frame size`() {
        assertFailsWith<IllegalArgumentException> { Framer(frameSize = 0) }
    }

    @Test
    fun `rejects a hop larger than the frame`() {
        assertFailsWith<IllegalArgumentException> { Framer(frameSize = 4, hopSize = 5) }
    }
}
