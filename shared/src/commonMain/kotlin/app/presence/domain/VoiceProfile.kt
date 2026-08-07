package app.presence.domain

import app.presence.engine.Embedding

/**
 * An enrolled person's voice.
 *
 * Centroids are stored **per language**, never pooled (spec §4.2). There is deliberately no way
 * to express a single cross-language centroid: pooling across a domain shift that large blurs
 * the profile instead of broadening it.
 *
 * @param role why this person is enrolled. Non-target roles are the cohort that makes relative
 *   scoring possible (spec §5); enrolling only the target child is a design error.
 * @param negatives embeddings the parent rejected via the "not them" button. A drifting profile
 *   should decay toward silence, not toward firing on a sibling.
 *
 * Treated as immutable once constructed. `init` validates at construction only, so callers must
 * not retain a mutable alias to any collection passed in — pass a fresh map or list. This is a
 * contract rather than an enforced guarantee: defensively copying would mean surrendering the
 * generated `copy()`, which enrollment and the matcher both use.
 */
data class VoiceProfile(
    val personId: String,
    val role: Role,
    val centroids: Map<Language, Embedding>,
    val negatives: List<Embedding> = emptyList(),
    val enrolledSecondsByLanguage: Map<Language, Double> = emptyMap(),
) {
    enum class Role {
        TARGET_CHILD,
        SIBLING,
        PARENT,
        OTHER,
    }

    init {
        require(personId.isNotBlank()) { "personId must not be blank" }
        require(centroids.isNotEmpty()) { "profile '$personId' has no centroids" }
        val dimensions = (centroids.values + negatives).map { it.dimension }.distinct()
        require(dimensions.size == 1) {
            "profile '$personId' mixes embedding dimensions: $dimensions"
        }
    }

    val languages: Set<Language> get() = centroids.keys
}
