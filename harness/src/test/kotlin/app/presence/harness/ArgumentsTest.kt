package app.presence.harness

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class ArgumentsTest {

    @Test
    fun `parses flags and values`() {
        val args = parseArgs(arrayOf("--manifest", "corpus", "--models", "models"))

        assertEquals("corpus", args["manifest"])
        assertEquals("models", args["models"])
    }

    @Test
    fun `help needs no value`() {
        assertEquals("true", parseArgs(arrayOf("--help"))["help"])
    }

    @Test
    fun `rejects an unknown flag and lists the known ones`() {
        val failure = assertFailsWith<IllegalArgumentException> { parseArgs(arrayOf("--holdouts", "h3")) }

        assertTrue(failure.message!!.contains("--holdouts"), failure.message!!)
        assertTrue(failure.message!!.contains("--holdout"), "should list the real flag")
    }

    @Test
    fun `rejects a flag used as another flag's value`() {
        val failure = assertFailsWith<IllegalArgumentException> {
            parseArgs(arrayOf("--holdout", "--manifest", "corpus"))
        }

        assertTrue(failure.message!!.contains("--holdout"), failure.message!!)
    }

    @Test
    fun `rejects a missing value`() {
        assertFailsWith<IllegalArgumentException> { parseArgs(arrayOf("--manifest")) }
    }

    @Test
    fun `rejects a bare argument`() {
        assertFailsWith<IllegalArgumentException> { parseArgs(arrayOf("corpus")) }
    }
}
