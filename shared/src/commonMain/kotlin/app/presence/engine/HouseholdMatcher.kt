package app.presence.engine

import app.presence.domain.VoiceProfile
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * @param cohortZ how far the target's similarity stands above the rest of the household, in
 *   standard deviations. This is what distinguishes "it is the target" from "it is a stranger
 *   who happens to resemble the target slightly more than the others".
 * @param bestNonTargetPersonId which household member [bestNonTargetSimilarity] belongs to: the
 *   person this observation most resembled other than the target, for diagnosing confusions
 * @param logLikelihoodRatio the evidence this observation contributes. Positive favours the
 *   target; the accumulator sums these across segments.
 * @param rawLogLikelihoodRatio the closed-set LLR before the open-set gates and negative veto are
 *   applied. Caching this alongside the gated value is what lets a threshold sweep replay
 *   [MatcherConfig.applyGates] at different gate settings without re-embedding audio — only
 *   [logLikelihoodRatio] itself is free to sweep otherwise.
 */
data class MatchScore(
    val targetSimilarity: Float,
    val bestNonTargetSimilarity: Float,
    val bestNonTargetPersonId: String,
    val cohortZ: Float,
    val logLikelihoodRatio: Float,
    val rawLogLikelihoodRatio: Float,
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
    /**
     * Absolute similarity the target must reach before any positive evidence is allowed.
     *
     * The z-score alone cannot do this job: a winning target necessarily sits far above the
     * cohort mean in standard-deviation terms, so the z floor never binds for a normal-sized
     * household. A stranger who resembles the target more than the family does would otherwise
     * pass. Calibrate this against real embeddings in Phase 0 — it is the single most important
     * open-set knob.
     */
    val minTargetSimilarity: Float = 0.35f,
    /** Floor on the cohort standard deviation, in cosine units — keeps z finite and meaningful. */
    val minCohortStdDev: Float = 0.02f,
    /** How close a negative example may come to the target before it vetoes the match. */
    val negativeVetoMargin: Float = 0.05f,
) {
    init {
        require(softmaxTemperature > 0f) { "softmaxTemperature must be positive" }
        require(fallbackCohortStdDev > 0f) { "fallbackCohortStdDev must be positive" }
        require(maxAbsLlr > 0f) { "maxAbsLlr must be positive" }
    }

    /**
     * Applies the open-set gates and the "not them" veto to a raw closed-set LLR.
     *
     * The single implementation [HouseholdMatcher.score] uses, and the one a future replay must
     * use too if it recomputes evidence at different gate settings from a cached
     * [MatchScore.rawLogLikelihoodRatio] — duplicating this logic anywhere else would let the two
     * silently drift apart.
     */
    fun applyGates(raw: Float, targetSimilarity: Float, cohortZ: Float, closestNegative: Float?): Float {
        var llr = raw
        // Open-set gates. A stranger or the television resembles nobody in particular, but the
        // posterior alone would still crown a winner among the enrolled.
        if (cohortZ < minCohortZ) llr = minOf(llr, 0f)
        if (targetSimilarity < minTargetSimilarity) llr = minOf(llr, 0f)

        // "Not them" veto. A drifting profile should decay toward silence, never toward firing on
        // a sibling.
        if (closestNegative != null && closestNegative >= targetSimilarity - negativeVetoMargin) {
            llr = minOf(llr, 0f)
        }

        return llr.coerceIn(-maxAbsLlr, maxAbsLlr)
    }
}

/**
 * Scores an observation against the whole household rather than against an absolute threshold.
 *
 * Rung L1 of the accuracy ladder (spec §4.1). Published EERs measure open-set verification with
 * a fixed threshold; we have a known roster, which turns the question into "which of these
 * people does this most resemble" and attacks sibling confusion directly.
 *
 * @param target the one child being listened for, whose role must be
 *   [VoiceProfile.Role.TARGET_CHILD]. v1 listens for a single child at a time (design decision D1);
 *   switching children means re-roling the previous target as a [VoiceProfile.Role.SIBLING].
 * @param cohort every other enrolled household member. Must not be empty: enrolling only the
 *   target child is a design error (spec §5), and this constructor is where that rule is enforced.
 *   Must not contain a second [VoiceProfile.Role.TARGET_CHILD] either: in the cohort, that child
 *   would suppress the first child's evidence while never producing any of their own, so a
 *   household asking for two targets would silently get worse at detecting both.
 */
class HouseholdMatcher(
    private val target: VoiceProfile,
    private val cohort: List<VoiceProfile>,
    private val config: MatcherConfig = MatcherConfig(),
) {
    init {
        require(target.role == VoiceProfile.Role.TARGET_CHILD) {
            "target '${target.personId}' has role ${target.role}, not TARGET_CHILD"
        }
        require(cohort.isNotEmpty()) {
            "cohort must not be empty: the parent and any siblings must be enrolled (spec §5)"
        }
        val otherTargets = cohort.filter { it.role == VoiceProfile.Role.TARGET_CHILD }.map { it.personId }
        require(otherTargets.isEmpty()) {
            "only one target child at a time: $otherTargets must be re-roled as SIBLING while " +
                "'${target.personId}' is the target"
        }
    }

    fun score(observation: Embedding): MatchScore {
        val targetSimilarity = bestSimilarity(target, observation)
        val cohortSimilarities = cohort.map { bestSimilarity(it, observation) }
        val bestIndex = cohortSimilarities.indices.maxBy { cohortSimilarities[it] }
        val bestNonTarget = cohortSimilarities[bestIndex]

        val z = cohortZScore(targetSimilarity, cohortSimilarities)
        val raw = closedSetLogLikelihoodRatio(targetSimilarity, cohortSimilarities)
        val closestNegative = target.negatives.maxOfOrNull { cosine(observation, it) }
        val llr = config.applyGates(raw, targetSimilarity, z, closestNegative)

        return MatchScore(
            targetSimilarity = targetSimilarity,
            bestNonTargetSimilarity = bestNonTarget,
            bestNonTargetPersonId = cohort[bestIndex].personId,
            cohortZ = z,
            logLikelihoodRatio = llr,
            rawLogLikelihoodRatio = raw,
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
            .coerceAtLeast(config.minCohortStdDev)
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
