package app.presence.domain

/**
 * Languages the app supports at first release (spec §4.2).
 *
 * Language is a first-class dimension of a voice profile rather than a UI setting, because
 * cross-lingual mismatch degrades speaker verification by 40–49% — a child enrolled in Hebrew
 * and heard speaking English is close to a different-domain trial.
 */
enum class Language {
    ENGLISH,
    HEBREW,
}
