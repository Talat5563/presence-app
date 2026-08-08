package app.presence.enrollment

import app.presence.domain.Language
import app.presence.domain.VoiceProfile
import app.presence.engine.Embedding
import app.presence.engine.centroidOf

/**
 * One accepted piece of enrollment audio, already embedded.
 *
 * @param speechSeconds seconds of *speech* in the clip, not wall-clock length. Progress is
 *   measured in usable speech because clips under about a second are near-worthless to the
 *   recognizer (spec §5), so a clip count would mislead.
 */
data class EnrollmentClip(
    val language: Language,
    val embedding: Embedding,
    val speechSeconds: Double,
)

/**
 * @param minSecondsPerLanguage usable speech required in each language before a profile counts
 *   as enrolled
 * @param requiredLanguages defaults to both supported languages, because bilingual enrollment is
 *   mandatory (spec §4.2). A genuinely monolingual household can narrow this, accepting that
 *   detection in the other language will be unreliable.
 */
data class EnrollmentPolicy(
    val minSecondsPerLanguage: Double = 45.0,
    val requiredLanguages: Set<Language> = setOf(Language.ENGLISH, Language.HEBREW),
) {
    init {
        require(minSecondsPerLanguage > 0) { "minSecondsPerLanguage must be positive" }
        require(requiredLanguages.isNotEmpty()) { "at least one language must be required" }
    }
}

sealed interface EnrollmentResult {

    /** Every required language met its minimum. */
    data class Complete(val profile: VoiceProfile) : EnrollmentResult

    /**
     * @param profile a usable partial profile, or null if no clips were supplied at all
     * @param shortfallSeconds remaining seconds needed, per language, for languages still short.
     *   This is what the enrollment UI shows the parent.
     */
    data class Incomplete(
        val profile: VoiceProfile?,
        val shortfallSeconds: Map<Language, Double>,
    ) : EnrollmentResult
}

/**
 * Builds a [VoiceProfile] from embedded clips.
 *
 * No training happens here — a centroid is the mean of the clip embeddings. That is why
 * enrollment completes in seconds and why imported recordings work exactly as well as live ones
 * (spec §4.1).
 */
class EnrollmentService(
    private val policy: EnrollmentPolicy = EnrollmentPolicy(),
) {
    fun enroll(
        personId: String,
        role: VoiceProfile.Role,
        clips: List<EnrollmentClip>,
    ): EnrollmentResult {
        val byLanguage = clips.groupBy { it.language }
        val secondsByLanguage = byLanguage.mapValues { (_, group) -> group.sumOf { it.speechSeconds } }

        val shortfall = policy.requiredLanguages
            .associateWith { language ->
                policy.minSecondsPerLanguage - (secondsByLanguage[language] ?: 0.0)
            }
            .filterValues { it > 0.0 }

        val profile = if (byLanguage.isEmpty()) {
            null
        } else {
            VoiceProfile(
                personId = personId,
                role = role,
                centroids = byLanguage.mapValues { (_, group) -> centroidOf(group.map { it.embedding }) },
                negatives = emptyList(),
                enrolledSecondsByLanguage = secondsByLanguage,
            )
        }

        return if (shortfall.isEmpty() && profile != null) {
            EnrollmentResult.Complete(profile)
        } else {
            EnrollmentResult.Incomplete(profile, shortfall)
        }
    }
}
