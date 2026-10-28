package app.presence.harness

import app.presence.domain.Language
import java.time.LocalDate
import java.time.format.DateTimeFormatter
import java.time.format.DateTimeFormatterBuilder
import java.util.Locale

/**
 * One speaker declared in a CHAT `@Participants` header.
 *
 * @param name the participant's name if the transcript gives one. CHAT allows it to be omitted.
 * @param role the CHAT role, e.g. `Target_Child`, `Mother`, `Brother`, `Investigator`
 */
data class ChatParticipant(val code: String, val name: String?, val role: String)

/** One time-aligned utterance: who spoke, and where in the session's media. */
data class ChatUtterance(val speakerCode: String, val startMs: Long, val endMs: Long) {
    val durationMs: Long get() = endMs - startMs
}

/**
 * The parts of a CHILDES transcript the importer needs. Words are deliberately dropped: only
 * who spoke, and when, matters for speaker recognition.
 *
 * @param mediaName the `@Media` base name, or null when the transcript has no linked media
 * @param unalignedUtterances main-tier lines with no time bullet, which cannot be cut from the
 *   audio. Reported so a corpus that is mostly unaligned is noticed rather than silently thin.
 */
data class ChatTranscript(
    val mediaName: String?,
    val languages: List<String>,
    val date: LocalDate?,
    val participants: Map<String, ChatParticipant>,
    val utterances: List<ChatUtterance>,
    val unalignedUtterances: Int,
) {
    /** The first declared language this harness supports, or null if none is. */
    val primaryLanguage: Language?
        get() = languages.firstNotNullOfOrNull { CHAT_LANGUAGES[it] }
}

private val CHAT_LANGUAGES = mapOf("eng" to Language.ENGLISH, "heb" to Language.HEBREW)

/**
 * A reader for the CHAT transcription format used by CHILDES and TalkBank.
 *
 * Only headers and main tiers are read. Utterance timing comes from the media bullet CHAT puts
 * at the end of an aligned utterance: `\u0015start_end\u0015`, in milliseconds.
 */
object Chat {

    private val BULLET = Regex("\u0015(\\d+)_(\\d+)\u0015")
    private val MAIN_TIER = Regex("^\\*([A-Za-z0-9]+):")

    private val DATE_FORMAT: DateTimeFormatter = DateTimeFormatterBuilder()
        .parseCaseInsensitive()
        .appendPattern("d-MMM-yyyy")
        .toFormatter(Locale.ENGLISH)

    fun parse(text: String): ChatTranscript {
        var mediaName: String? = null
        var languages = emptyList<String>()
        var date: LocalDate? = null
        val participants = LinkedHashMap<String, ChatParticipant>()
        val utterances = ArrayList<ChatUtterance>()
        var unaligned = 0

        for (line in logicalLines(text.removePrefix("﻿"))) {
            when {
                line.startsWith("@Media:") -> mediaName = parseMedia(headerValue(line))
                line.startsWith("@Languages:") -> languages = headerValue(line)
                    .split(',', ' ')
                    .map { it.trim() }
                    .filter { it.isNotEmpty() }
                line.startsWith("@Date:") -> date = parseDate(headerValue(line))
                line.startsWith("@Participants:") -> parseParticipants(headerValue(line))
                    .forEach { participants[it.code] = it }
                line.startsWith("*") -> {
                    val speaker = MAIN_TIER.find(line)?.groupValues?.get(1) ?: continue
                    val bullet = BULLET.find(line)
                    if (bullet == null) {
                        unaligned += 1
                        continue
                    }
                    val start = bullet.groupValues[1].toLong()
                    val end = bullet.groupValues[2].toLong()
                    // A zero-length or reversed bullet is a transcription slip; it cannot be cut.
                    if (end > start) utterances += ChatUtterance(speaker, start, end) else unaligned += 1
                }
            }
        }

        return ChatTranscript(mediaName, languages, date, participants, utterances, unaligned)
    }

    /** Joins tab-indented continuation lines onto the line they continue. */
    private fun logicalLines(text: String): List<String> {
        val lines = ArrayList<String>()
        for (raw in text.lineSequence()) {
            val line = raw.trimEnd('\r')
            if (line.startsWith("\t") && lines.isNotEmpty()) {
                lines[lines.size - 1] = lines.last() + " " + line.trim()
            } else {
                lines += line
            }
        }
        return lines
    }

    private fun headerValue(line: String): String = line.substringAfter(':').trim()

    /** `@Media: 010106a, audio` → `010106a`. Unlinked or missing media cannot be cut. */
    private fun parseMedia(value: String): String? {
        val fields = value.split(',').map { it.trim().lowercase() }
        if (fields.any { it == "unlinked" || it == "missing" || it == "notrans" }) return null
        return value.split(',').first().trim().ifEmpty { null }
    }

    private fun parseDate(value: String): LocalDate? =
        try {
            LocalDate.parse(value, DATE_FORMAT)
        } catch (_: Exception) {
            null
        }

    /** `CHI Ross Target_Child, FAT Father` — the name is optional, the role is always last. */
    private fun parseParticipants(value: String): List<ChatParticipant> =
        value.split(',').mapNotNull { entry ->
            val tokens = entry.trim().split(Regex("\\s+")).filter { it.isNotEmpty() }
            when {
                tokens.size >= 3 -> ChatParticipant(
                    code = tokens.first(),
                    name = tokens.subList(1, tokens.size - 1).joinToString(" "),
                    role = tokens.last(),
                )
                tokens.size == 2 -> ChatParticipant(code = tokens[0], name = null, role = tokens[1])
                else -> null
            }
        }
}
