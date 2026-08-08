package app.presence.engine

import app.presence.domain.VoiceProfile
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * @param cohortZ how far the target's similarity stands above the rest of the household, in
 *   standard deviations. This is what distinguishes "it is the target" from "it is a stranger
 *   who happens to resemble the target slightly more than the others".
 * @param logLikelihoodRatio the evidence this observation contributes. Positive favours the
 *   target; the accumulator sums these across segments.
 */
data class MatchScore(
    val targetSimilarity: Float,
    val bestNonTargetSimilarity: Float,
    val cohortZ: Float,
    val logLikelihoodRatio: Float,
)

/**
 * @param softmaxTemperature lower values sharpen the closed-set posterior. Tuned in Phase 0.
 * @param minCohortZ evidence floor for the open-set case; below this, no positive evidence passes
 * @param fallbackCohortMean used when the cohort is too small for real statistics
 * @param fallbackCohortStdDev likewise
 * @param maxAbsLlr keeps one confident observation from dominating an entire window
 */
data class MatcherConfig(
    val softmaxTemperature: Float = 0.1f,
    val minCohortZ: Float = 0.5f,
    val fallbackCohortMean: Float = 0.15f,
    val fallbackCohortStdDev: Float = 0.10f,
    val maxAbsLlr: Float = 4.0f,
) {
    init {
        require(softmaxTemperature > 0f) { "softmaxTemperature must be positive" }
        require(fallbackCohortStdDev > 0f) { "fallbackCohortStdDev must be positive" }
        require(maxAbsLlr > 0f) { "maxAbsLlr must be positive" }
    }
}

/**
 * Scores an observation against the whole household rather than against an absolute threshold.
 *
 * Rung L1 of the accuracy ladder (spec §4.1). Published EERs measure open-set verification with
 * a fixed threshold; we have a known roster, which turns the question into "which of these
 * people does this most resemble" and attacks sibling confusion directly.
 *
 * @param cohort every other enrolled household member. Must not be empty: enrolling only the
 *   target child is a design error (spec §5), and this constructor is where that rule is enforced.
 */
class HouseholdMatcher(
    private val target: VoiceProfile,
    private val cohort: List<VoiceProfile>,
    private val config: MatcherConfig = MatcherConfig(),
) {
    init {
        require(cohort.isNotEmpty()) {
            "cohort must not be empty: the parent and any siblings must be enrolled (spec §5)"
        }
    }

    fun score(observation: Embedding): MatchScore {
        val targetSimilarity = bestSimilarity(target, observation)
        val cohortSimilarities = cohort.map { bestSimilarity(it, observation) }
        val bestNonTarget = cohortSimilarities.max()

        val z = cohortZScore(targetSimilarity, cohortSimilarities)
        var llr = closedSetLogLikelihoodRatio(targetSimilarity, cohortSimilarities)

        // Open-set gate. A stranger or the television resembles nobody in particular, but the
        // posterior alone would still crown a winner among the enrolled.
        if (z < config.minCohortZ) llr = minOf(llr, 0f)

        // "Not them" vetoes. A drifting profile should decay toward silence, never toward
        // firing on a sibling.
        val closestNegative = target.negatives.maxOfOrNull { cosine(observation, it) }
        if (closestNegative != null && closestNegative >= targetSimilarity) llr = minOf(llr, 0f)

        return MatchScore(
            targetSimilarity = targetSimilarity,
            bestNonTargetSimilarity = bestNonTarget,
            cohortZ = z,
            logLikelihoodRatio = llr.coerceIn(-config.maxAbsLlr, config.maxAbsLlr),
        )
    }

    /** The best-matching language sub-centroid, never an average across languages (spec §4.2). */
    private fun bestSimilarity(profile: VoiceProfile, observation: Embedding): Float =
        profile.centroids.values.maxOf { cosine(observation, it) }

    private fun cohortZScore(targetSimilarity: Float, cohortSimilarities: List<Float>): Float {
        if (cohortSimilarities.size < 2) {
            return (targetSimilarity - config.fallbackCohortMean) / config.fallbackCohortStdDev
        }
        val mean = cohortSimilarities.average().toFloat()
        var sumSquaredDeviation = 0.0
        for (similarity in cohortSimilarities) {
            val deviation = (similarity - mean).toDouble()
            sumSquaredDeviation += deviation * deviation
        }
        val stdDev = sqrt(sumSquaredDeviation / cohortSimilarities.size)
            .toFloat()
            .coerceAtLeast(1e-4f)
        return (targetSimilarity - mean) / stdDev
    }

    private fun closedSetLogLikelihoodRatio(
        targetSimilarity: Float,
        cohortSimilarities: List<Float>,
    ): Float {
        val all = ArrayList<Float>(cohortSimilarities.size + 1)
        all += targetSimilarity
        all += cohortSimilarities

        // Subtract the maximum before exponentiating, for numerical stability.
        val maximum = all.max()
        var denominator = 0.0
        for (similarity in all) {
            denominator += exp(((similarity - maximum) / config.softmaxTemperature).toDouble())
        }
        val numerator = exp(((targetSimilarity - maximum) / config.softmaxTemperature).toDouble())
        val posterior = (numerator / denominator).coerceIn(1e-6, 1.0 - 1e-6)
        return ln(posterior / (1.0 - posterior)).toFloat()
    }
}
