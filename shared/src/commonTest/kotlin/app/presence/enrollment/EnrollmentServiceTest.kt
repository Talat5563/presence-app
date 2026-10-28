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

    private fun enrolled(): VoiceProfile {
        val result = service.enroll(
            personId = "maya",
            role = VoiceProfile.Role.TARGET_CHILD,
            clips = listOf(
                clip(Language.ENGLISH, 150.0, 1f, 0f),
                enough(Language.HEBREW, 0f, 1f),
            ),
        )
        return assertIs<EnrollmentResult.Complete>(result).profile
    }

    @Test
    fun `a top-up merges into the existing centroid weighted by speech seconds`() {
        // 150 s of (1, 0) already enrolled; a 50 s session of (0, 1) pulls it a quarter of the way.
        val result = service.topUp(enrolled(), listOf(clip(Language.ENGLISH, 50.0, 0f, 1f)))

        val merged = assertIs<EnrollmentResult.Complete>(result).profile
        val expected = Embedding.ofNormalized(floatArrayOf(3f, 1f))
        assertTrue(cosine(expected, merged.centroids.getValue(Language.ENGLISH)) > 0.9999f)
        assertEquals(200.0, merged.enrolledSecondsByLanguage[Language.ENGLISH])
    }

    @Test
    fun `a top-up leaves languages it has no clips for untouched`() {
        val existing = enrolled()

        val result = service.topUp(existing, listOf(clip(Language.ENGLISH, 10.0, 0f, 1f)))

        val merged = assertIs<EnrollmentResult.Complete>(result).profile
        assertTrue(merged.centroids[Language.HEBREW] === existing.centroids[Language.HEBREW])
        assertEquals(50.0, merged.enrolledSecondsByLanguage[Language.HEBREW])
    }

    @Test
    fun `a top-up keeps every not-them correction`() {
        val corrected = enrolled().withNegatives(listOf(Embedding.ofNormalized(floatArrayOf(1f, 1f))))

        val result = service.topUp(corrected, listOf(enough(Language.ENGLISH, 1f, 0f)))

        val merged = assertIs<EnrollmentResult.Complete>(result).profile
        assertEquals(corrected.negatives, merged.negatives)
        assertEquals(VoiceProfile.Role.TARGET_CHILD, merged.role)
    }

    @Test
    fun `a top-up can complete a language the first session left short`() {
        val partial = assertIs<EnrollmentResult.Incomplete>(
            service.enroll(
                personId = "maya",
                role = VoiceProfile.Role.TARGET_CHILD,
                clips = listOf(enough(Language.ENGLISH, 1f, 0f), clip(Language.HEBREW, 30.0, 0f, 1f)),
            ),
        )

        val result = service.topUp(assertNotNull(partial.profile), listOf(clip(Language.HEBREW, 20.0, 0f, 1f)))

        assertIs<EnrollmentResult.Complete>(result)
    }

    @Test
    fun `a top-up can add a language the profile never had`() {
        val englishOnly = assertNotNull(
            assertIs<EnrollmentResult.Incomplete>(
                service.enroll("maya", VoiceProfile.Role.TARGET_CHILD, listOf(enough(Language.ENGLISH, 1f, 0f))),
            ).profile,
        )

        val result = service.topUp(englishOnly, listOf(enough(Language.HEBREW, 0f, 1f)))

        val merged = assertIs<EnrollmentResult.Complete>(result).profile
        assertEquals(setOf(Language.ENGLISH, Language.HEBREW), merged.languages)
    }

    @Test
    fun `a top-up refuses a profile whose sessions cannot be weighted`() {
        val noSeconds = VoiceProfile(
            personId = "maya",
            role = VoiceProfile.Role.TARGET_CHILD,
            centroids = mapOf(Language.ENGLISH to Embedding.ofNormalized(floatArrayOf(1f, 0f))),
        )

        kotlin.test.assertFailsWith<IllegalArgumentException> {
            service.topUp(noSeconds, listOf(enough(Language.ENGLISH, 0f, 1f)))
        }
    }
}
