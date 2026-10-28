package app.presence.domain

import app.presence.engine.Embedding

/**
 * The engine's output: the target was heard, with enough confidence to interrupt.
 *
 * Carries no audio — spec §11.4 permits logging timestamp, person and confidence, never audio.
 *
 * @param timestampMs position in the audio stream where the threshold was crossed
 * @param accumulatedEvidence summed log-likelihood ratio at the moment of firing
 * @param peakSimilarity strongest raw cosine similarity seen, for diagnostics only
 * @param evidenceEmbeddings the voices that pushed evidence over the threshold — every
 *   positive-evidence observation in the window that fired. This is the "not them" correction's
 *   data path: if the parent rejects the detection, these become the profile's negatives (see
 *   [VoiceProfile.withNegatives]). They are biometric, so they go into the profile or nowhere —
 *   never into the event log.
 */
data class DetectionEvent(
    val personId: String,
    val timestampMs: Long,
    val accumulatedEvidence: Float,
    val peakSimilarity: Float,
    val evidenceEmbeddings: List<Embedding>,
)
