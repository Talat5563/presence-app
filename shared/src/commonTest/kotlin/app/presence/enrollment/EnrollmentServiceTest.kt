package app.presence.enrollment

import app.presence.domain.Language
import app.presence.domain.VoiceProfile
import app.presence.engine.Embedding
import app.presence.engine.cosine
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class EnrollmentServiceTest {

    private val service = EnrollmentService()

    private fun clip(language: Language, seconds: Double, vararg values: Float) =
        EnrollmentClip(language, Embedding.ofNormalized(floatArrayOf(*values)), seconds)

    private fun enough(language: Language, vararg values: Float) =
        clip(language, 50.0, *values)

    @Test
    fun `completes when both languages meet the minimum`() {
        val result = service.enroll(
            personId = "maya",
            role = VoiceProfile.Role.TARGET_CHILD,
            clips = listOf(
                enough(Language.ENGLISH, 1f, 0f),
                enough(Language.HEBREW, 0f, 1f),
            ),
        )

        val complete = assertIs<EnrollmentResult.Complete>(result)
        assertEquals(setOf(Language.ENGLISH, Language.HEBREW), complete.profile.languages)
    }

    @Test
    fun `is incomplete when Hebrew is missing entirely`() {
        val result = service.enroll(
            personId = "maya",
            role = VoiceProfile.Role.TARGET_CHILD,
            clips = listOf(enough(Language.ENGLISH, 1f, 0f)),
        )

        val incomplete = assertIs<EnrollmentResult.Incomplete>(result)
        assertEquals(45.0, incomplete.shortfallSeconds[Language.HEBREW])
        assertNull(incomplete.shortfallSeconds[Language.ENGLISH])
    }

    @Test
    fun `reports the exact remaining seconds per language`() {
        val result = service.enroll(
            personId = "maya",
            role = VoiceProfile.Role.TARGET_CHILD,
            clips = listOf(
                clip(Language.ENGLISH, 30.0, 1f, 0f),
                clip(Language.HEBREW, 5.0, 0f, 1f),
            ),
        )

        val incomplete = assertIs<EnrollmentResult.Incomplete>(result)
        assertEquals(15.0, incomplete.shortfallSeconds[Language.ENGLISH])
        assertEquals(40.0, incomplete.shortfallSeconds[Language.HEBREW])
    }

    @Test
    fun `still returns a usable partial profile when incomplete`() {
        val result = service.enroll(
            personId = "maya",
            role = VoiceProfile.Role.TARGET_CHILD,
            clips = listOf(clip(Language.ENGLISH, 10.0, 1f, 0f)),
        )

        val incomplete = assertIs<EnrollmentResult.Incomplete>(result)
        val profile = assertNotNull(incomplete.profile)
        assertEquals(setOf(Language.ENGLISH), profile.languages)
    }

    @Test
    fun `returns no profile at all when there are no clips`() {
        val result = service.enroll(
            personId = "maya",
            role = VoiceProfile.Role.TARGET_CHILD,
            clips = emptyList(),
        )

        val incomplete = assertIs<EnrollmentResult.Incomplete>(result)
        assertNull(incomplete.profile)
    }

    @Test
    fun `averages clips within a language into one centroid`() {
        val result = service.enroll(
            personId = "maya",
            role = VoiceProfile.Role.TARGET_CHILD,
            clips = listOf(
                clip(Language.ENGLISH, 25.0, 1f, 0f),
                clip(Language.ENGLISH, 25.0, 0f, 1f),
                enough(Language.HEBREW, 0f, 1f),
            ),
        )

        val complete = assertIs<EnrollmentResult.Complete>(result)
        val english = complete.profile.centroids[Language.ENGLISH]!!
        val first = Embedding.ofNormalized(floatArrayOf(1f, 0f))
        val second = Embedding.ofNormalized(floatArrayOf(0f, 1f))
        assertTrue(cosine(english, first) > 0.7f)
        assertTrue(cosine(english, second) > 0.7f)
    }

    @Test
    fun `records enrolled seconds per language on the profile`() {
        val result = service.enroll(
            personId = "maya",
            role = VoiceProfile.Role.TARGET_CHILD,
            clips = listOf(
                clip(Language.ENGLISH, 25.0, 1f, 0f),
                clip(Language.ENGLISH, 30.0, 1f, 0f),
                enough(Language.HEBREW, 0f, 1f),
            ),
        )

        val complete = assertIs<EnrollmentResult.Complete>(result)
        assertEquals(55.0, complete.profile.enrolledSecondsByLanguage[Language.ENGLISH])
        assertEquals(50.0, complete.profile.enrolledSecondsByLanguage[Language.HEBREW])
    }

    @Test
    fun `a single-language policy suits a monolingual household`() {
        val monolingual = EnrollmentService(
            EnrollmentPolicy(requiredLanguages = setOf(Language.HEBREW)),
        )

        val result = monolingual.enroll(
            personId = "savta",
            role = VoiceProfile.Role.OTHER,
            clips = listOf(enough(Language.HEBREW, 0f, 1f)),
        )

        assertIs<EnrollmentResult.Complete>(result)
    }

    @Test
    fun `rejects a policy with no required languages`() {
        kotlin.test.assertFailsWith<IllegalArgumentException> {
            EnrollmentPolicy(requiredLanguages = emptySet())
        }
    }
}
