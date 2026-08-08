package app.presence.domain

import app.presence.engine.Embedding
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class VoiceProfileTest {

    private fun embedding(vararg values: Float) = Embedding.ofNormalized(floatArrayOf(*values))

    @Test
    fun `exposes the languages it was enrolled in`() {
        val profile = VoiceProfile(
            personId = "maya",
            role = VoiceProfile.Role.TARGET_CHILD,
            centroids = mapOf(
                Language.ENGLISH to embedding(1f, 0f),
                Language.HEBREW to embedding(0f, 1f),
            ),
        )

        assertEquals(setOf(Language.ENGLISH, Language.HEBREW), profile.languages)
    }

    @Test
    fun `allows a single-language profile`() {
        val profile = VoiceProfile(
            personId = "guest",
            role = VoiceProfile.Role.OTHER,
            centroids = mapOf(Language.ENGLISH to embedding(1f, 0f)),
        )

        assertEquals(setOf(Language.ENGLISH), profile.languages)
    }

    @Test
    fun `rejects a profile with no centroids`() {
        val failure = assertFailsWith<IllegalArgumentException> {
            VoiceProfile(
                personId = "nobody",
                role = VoiceProfile.Role.OTHER,
                centroids = emptyMap(),
            )
        }
        assertTrue(failure.message!!.contains("nobody"))
    }

    @Test
    fun `rejects a blank person id`() {
        assertFailsWith<IllegalArgumentException> {
            VoiceProfile(
                personId = "  ",
                role = VoiceProfile.Role.OTHER,
                centroids = mapOf(Language.ENGLISH to embedding(1f, 0f)),
            )
        }
    }

    @Test
    fun `rejects centroids of differing dimensions`() {
        assertFailsWith<IllegalArgumentException> {
            VoiceProfile(
                personId = "maya",
                role = VoiceProfile.Role.TARGET_CHILD,
                centroids = mapOf(
                    Language.ENGLISH to embedding(1f, 0f),
                    Language.HEBREW to embedding(0f, 1f, 0f),
                ),
            )
        }
    }

    @Test
    fun `defaults to no negatives and no recorded durations`() {
        val profile = VoiceProfile(
            personId = "maya",
            role = VoiceProfile.Role.TARGET_CHILD,
            centroids = mapOf(Language.ENGLISH to embedding(1f, 0f)),
        )

        assertTrue(profile.negatives.isEmpty())
        assertTrue(profile.enrolledSecondsByLanguage.isEmpty())
    }

    @Test
    fun `rejects a negative example of a different dimension`() {
        val failure = assertFailsWith<IllegalArgumentException> {
            VoiceProfile(
                personId = "maya",
                role = VoiceProfile.Role.TARGET_CHILD,
                centroids = mapOf(Language.ENGLISH to embedding(1f, 0f)),
                negatives = listOf(embedding(0f, 1f, 0f)),
            )
        }
        assertTrue(failure.message!!.contains("maya"))
    }
}
