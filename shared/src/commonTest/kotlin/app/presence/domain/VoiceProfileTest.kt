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

    @Test
    fun `rejected embeddings are appended to the negatives`() {
        val existing = embedding(0f, 1f)
        val profile = VoiceProfile(
            personId = "maya",
            role = VoiceProfile.Role.TARGET_CHILD,
            centroids = mapOf(Language.ENGLISH to embedding(1f, 0f)),
            negatives = listOf(existing),
        )
        val rejected = listOf(embedding(1f, 1f), embedding(1f, 2f))

        assertEquals(listOf(existing) + rejected, profile.withNegatives(rejected).negatives)
    }

    @Test
    fun `negatives are capped by dropping the oldest`() {
        val oldest = embedding(0f, 1f)
        val middle = embedding(1f, 1f)
        val newest = embedding(1f, 2f)
        val profile = VoiceProfile(
            personId = "maya",
            role = VoiceProfile.Role.TARGET_CHILD,
            centroids = mapOf(Language.ENGLISH to embedding(1f, 0f)),
            negatives = listOf(oldest),
        )

        val capped = profile.withNegatives(listOf(middle, newest), maxNegatives = 2)

        assertEquals(listOf(middle, newest), capped.negatives)
    }

    @Test
    fun `rejects a non-positive negatives cap`() {
        val profile = VoiceProfile(
            personId = "maya",
            role = VoiceProfile.Role.TARGET_CHILD,
            centroids = mapOf(Language.ENGLISH to embedding(1f, 0f)),
        )

        assertFailsWith<IllegalArgumentException> { profile.withNegatives(emptyList(), maxNegatives = 0) }
    }
}
