package app.presence.engine

import app.presence.domain.Language
import app.presence.domain.VoiceProfile
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class HouseholdMatcherTest {

    private fun profile(
        id: String,
        role: VoiceProfile.Role,
        vararg centroids: Pair<Language, FloatArray>,
    ) = VoiceProfile(
        personId = id,
        role = role,
        centroids = centroids.associate { (language, values) ->
            language to Embedding.ofNormalized(values)
        },
    )

    private fun observation(vararg values: Float) = Embedding.ofNormalized(floatArrayOf(*values))

    private val maya = profile(
        "maya", VoiceProfile.Role.TARGET_CHILD,
        Language.ENGLISH to floatArrayOf(1f, 0f, 0f),
    )
    private val parent = profile(
        "parent", VoiceProfile.Role.PARENT,
        Language.ENGLISH to floatArrayOf(0f, 1f, 0f),
    )
    private val sibling = profile(
        "sibling", VoiceProfile.Role.SIBLING,
        Language.ENGLISH to floatArrayOf(0f, 0f, 1f),
    )

    private fun matcher(
        target: VoiceProfile = maya,
        cohort: List<VoiceProfile> = listOf(parent, sibling),
        config: MatcherConfig = MatcherConfig(),
    ) = HouseholdMatcher(target, cohort, config)

    @Test
    fun `the target's own voice produces strong positive evidence`() {
        val score = matcher().score(observation(1f, 0f, 0f))

        assertTrue(score.logLikelihoodRatio > 3f, "expected strong evidence, got ${score.logLikelihoodRatio}")
        assertTrue(abs(1f - score.targetSimilarity) < 1e-5f)
    }

    @Test
    fun `a cohort member's voice produces negative evidence`() {
        val score = matcher().score(observation(0f, 1f, 0f))

        assertTrue(score.logLikelihoodRatio < 0f, "expected negative evidence, got ${score.logLikelihoodRatio}")
    }

    @Test
    fun `a sibling is not mistaken for the target`() {
        val score = matcher().score(observation(0f, 0f, 1f))

        assertTrue(score.logLikelihoodRatio < 0f)
        assertTrue(score.bestNonTargetSimilarity > score.targetSimilarity)
    }

    @Test
    fun `a stranger resembling everyone equally produces no positive evidence`() {
        // Equidistant from all three profiles: the softmax alone would still pick a winner,
        // so this is the case the z gate exists for.
        val score = matcher().score(observation(0.577f, 0.577f, 0.577f))

        assertTrue(score.logLikelihoodRatio <= 0f, "got ${score.logLikelihoodRatio}")
    }

    @Test
    fun `the z gate suppresses a weak win over a small cohort`() {
        // One cohort member means no usable cohort statistics, so the fallback mean and
        // standard deviation apply: z = (0.18 - 0.15) / 0.10 = 0.30, below the 0.5 floor.
        val single = matcher(cohort = listOf(parent))

        val score = single.score(observation(0.18f, 0.05f, 0.9823f))

        assertTrue(score.targetSimilarity > score.bestNonTargetSimilarity, "target should nominally win")
        assertTrue(score.cohortZ < 0.5f, "z was ${score.cohortZ}")
        assertEquals(0f, score.logLikelihoodRatio, "a weak win must not become positive evidence")
    }

    @Test
    fun `a negative example vetoes a match`() {
        val withNegative = maya.copy(negatives = listOf(observation(1f, 0f, 0f)))

        val score = matcher(target = withNegative).score(observation(1f, 0f, 0f))

        assertEquals(0f, score.logLikelihoodRatio)
    }

    @Test
    fun `a negative example unlike the observation does not veto`() {
        val withNegative = maya.copy(negatives = listOf(observation(0f, 0f, 1f)))

        val score = matcher(target = withNegative).score(observation(1f, 0f, 0f))

        assertTrue(score.logLikelihoodRatio > 3f)
    }

    @Test
    fun `the best-matching language sub-centroid is used`() {
        val bilingual = profile(
            "maya", VoiceProfile.Role.TARGET_CHILD,
            Language.ENGLISH to floatArrayOf(1f, 0f, 0f),
            Language.HEBREW to floatArrayOf(0f, 1f, 0f),
        )
        // The observation matches the Hebrew centroid; scoring must not average the two.
        val score = HouseholdMatcher(bilingual, listOf(sibling)).score(observation(0f, 1f, 0f))

        assertTrue(abs(1f - score.targetSimilarity) < 1e-5f, "got ${score.targetSimilarity}")
    }

    @Test
    fun `evidence is clamped to the configured maximum`() {
        val score = matcher(config = MatcherConfig(maxAbsLlr = 2f)).score(observation(1f, 0f, 0f))

        assertTrue(score.logLikelihoodRatio <= 2f)
    }

    @Test
    fun `reports the best non-target similarity`() {
        val score = matcher().score(observation(0f, 0.6f, 0.8f))

        assertTrue(abs(0.8f - score.bestNonTargetSimilarity) < 1e-5f, "got ${score.bestNonTargetSimilarity}")
    }

    /**
     * The test the z gate cannot satisfy. A winning target always sits far above the cohort mean
     * in standard-deviation terms, so only the absolute floor can reject a voice that beats the
     * household without resembling anyone much.
     */
    @Test
    fun `a weak absolute match produces no positive evidence even when it beats the cohort`() {
        val target = profile("maya", VoiceProfile.Role.TARGET_CHILD, Language.ENGLISH to floatArrayOf(1f, 0f, 0f, 0f))
        val parent = profile("parent", VoiceProfile.Role.PARENT, Language.ENGLISH to floatArrayOf(0f, 1f, 0f, 0f))
        val sibling = profile("sibling", VoiceProfile.Role.SIBLING, Language.ENGLISH to floatArrayOf(0f, 0f, 1f, 0f))
        // Similarity 0.30 to the target, 0.05 to each cohort member, the rest matching nobody.
        val stranger = observation(0.30f, 0.05f, 0.05f, 0.9513f)

        val score = HouseholdMatcher(target, listOf(parent, sibling)).score(stranger)

        assertTrue(score.targetSimilarity > score.bestNonTargetSimilarity, "target should nominally win")
        assertTrue(score.cohortZ >= 0.5f, "the z gate does not bind here — that is the point")
        assertEquals(0f, score.logLikelihoodRatio, "the absolute floor must suppress this")
    }

    @Test
    fun `a negative example close to the target also vetoes`() {
        val nearMiss = maya.copy(negatives = listOf(observation(0.97f, 0.2431f, 0f)))

        val score = matcher(target = nearMiss).score(observation(1f, 0f, 0f))

        assertEquals(0f, score.logLikelihoodRatio)
    }

    @Test
    fun `an empty cohort is rejected because household enrollment is required`() {
        val failure = assertFailsWith<IllegalArgumentException> {
            HouseholdMatcher(maya, emptyList())
        }
        assertTrue(failure.message!!.contains("cohort"))
    }

    @Test
    fun `a second target child in the cohort is rejected`() {
        val noa = profile("noa", VoiceProfile.Role.TARGET_CHILD, Language.ENGLISH to floatArrayOf(0f, 0f, 1f))

        val failure = assertFailsWith<IllegalArgumentException> {
            HouseholdMatcher(maya, listOf(parent, noa))
        }
        assertTrue(failure.message!!.contains("noa"))
    }

    @Test
    fun `a target without the target role is rejected`() {
        assertFailsWith<IllegalArgumentException> {
            HouseholdMatcher(sibling, listOf(parent, maya.copy(role = VoiceProfile.Role.SIBLING)))
        }
    }

    @Test
    fun `switching children means re-roling the previous target as a sibling`() {
        val noa = profile("noa", VoiceProfile.Role.TARGET_CHILD, Language.ENGLISH to floatArrayOf(0f, 0f, 1f))
        val switched = HouseholdMatcher(noa, listOf(parent, maya.copy(role = VoiceProfile.Role.SIBLING)))

        // Maya now argues against Noa rather than for herself.
        assertTrue(switched.score(observation(1f, 0f, 0f)).logLikelihoodRatio < 0f)
        assertTrue(switched.score(observation(0f, 0f, 1f)).logLikelihoodRatio > 3f)
    }

    @Test
    fun `reports which household member the observation most resembled`() {
        assertEquals("sibling", matcher().score(observation(0f, 0.6f, 0.8f)).bestNonTargetPersonId)
        assertEquals("parent", matcher().score(observation(0f, 0.8f, 0.6f)).bestNonTargetPersonId)
    }
}
