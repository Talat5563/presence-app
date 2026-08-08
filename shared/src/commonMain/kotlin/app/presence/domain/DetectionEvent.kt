package app.presence.domain

/**
 * The engine's output: the target was heard, with enough confidence to interrupt.
 *
 * Carries no audio — spec §11.4 permits logging timestamp, person and confidence, never audio.
 *
 * @param timestampMs position in the audio stream where the threshold was crossed
 * @param accumulatedEvidence summed log-likelihood ratio at the moment of firing
 * @param peakSimilarity strongest raw cosine similarity seen, for diagnostics only
 */
data class DetectionEvent(
    val personId: String,
    val timestampMs: Long,
    val accumulatedEvidence: Float,
    val peakSimilarity: Float,
)
