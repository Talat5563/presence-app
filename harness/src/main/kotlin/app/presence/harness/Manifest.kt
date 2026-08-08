package app.presence.harness

import app.presence.domain.Language
import app.presence.domain.VoiceProfile
import java.nio.file.Path
import kotlin.io.path.readText

/** Why a speaker appears in the corpus. Only household members can be enrolled. */
enum class RecordingRole {
    TARGET_CHILD,
    SIBLING,
    PARENT,
    OTHER_HOUSEHOLD,

    /** Television, streaming, children's programming — the worst adversary (spec §5). */
    MEDIA,

    /** Visitors, cousins, a playground: open-set impostors (spec §9.1). */
    STRANGER,

    /** Silence and room tone, for calibrating the VAD. */
    ROOM_TONE,
    ;

    val isHouseholdMember: Boolean
        get() = this == TARGET_CHILD || this == SIBLING || this == PARENT || this == OTHER_HOUSEHOLD

    fun toProfileRole(): VoiceProfile.Role = when (this) {
        TARGET_CHILD -> VoiceProfile.Role.TARGET_CHILD
        SIBLING -> VoiceProfile.Role.SIBLING
        PARENT -> VoiceProfile.Role.PARENT
        else -> VoiceProfile.Role.OTHER
    }
}

enum class ContentType { SPONTANEOUS, READ, CALLING, ROOM_TONE }

/** One audio file plus the §9.1 metadata that makes it interpretable. */
data class Recording(
    val path: String,
    val householdId: String,
    val speakerId: String,
    val role: RecordingRole,
    val language: Language?,
    val codeSwitched: Boolean,
    val contentType: ContentType,
    val distanceMeters: Double?,
    val placement: String,
    val background: String,
    val room: String,
    val date: String,
)

/** Where in a recording the child actually called for the parent — the event we must catch. */
data class CallEvent(
    val recordingPath: String,
    val startMs: Long,
    val targetSpeakerId: String,
) {
    init {
        require(recordingPath.isNotBlank()) { "recordingPath must not be blank" }
        require(targetSpeakerId.isNotBlank()) { "targetSpeakerId must not be blank" }
        require(startMs >= 0) { "event start must not be negative, was $startMs" }
    }
}

object Manifest {

    private val REQUIRED_RECORDING_COLUMNS = listOf(
        "path", "household_id", "speaker_id", "role", "language", "code_switched",
        "content_type", "distance_m", "placement", "background", "room", "date",
    )

    private val REQUIRED_EVENT_COLUMNS = listOf("recording_path", "start_ms", "target_speaker_id")

    fun readRecordings(path: Path): List<Recording> {
        val rows = Csv.parseWithLineNumbers(path.readText())
        require(rows.isNotEmpty()) { "$path has a header but no data rows" }
        requireColumns(rows.first().second.keys, REQUIRED_RECORDING_COLUMNS, path)

        return rows.map { (lineNumber, row) ->
            withRowContext(path, lineNumber) {
                Recording(
                    path = row.getValue("path"),
                    householdId = row.getValue("household_id"),
                    speakerId = row.getValue("speaker_id"),
                    role = parseEnum(row.getValue("role"), RecordingRole.entries, "role"),
                    language = row.getValue("language").ifBlank { null }
                        ?.let { parseEnum(it, Language.entries, "language") },
                    codeSwitched = row.getValue("code_switched").equals("true", ignoreCase = true),
                    contentType = parseEnum(row.getValue("content_type"), ContentType.entries, "content_type"),
                    distanceMeters = row.getValue("distance_m").ifBlank { null }?.toDouble(),
                    placement = row.getValue("placement"),
                    background = row.getValue("background"),
                    room = row.getValue("room"),
                    date = row.getValue("date"),
                )
            }
        }
    }

    fun readEvents(path: Path): List<CallEvent> {
        val rows = Csv.parseWithLineNumbers(path.readText())
        require(rows.isNotEmpty()) { "$path has a header but no data rows" }
        requireColumns(rows.first().second.keys, REQUIRED_EVENT_COLUMNS, path)

        return rows.map { (lineNumber, row) ->
            withRowContext(path, lineNumber) {
                CallEvent(
                    recordingPath = row.getValue("recording_path"),
                    startMs = row.getValue("start_ms").toLong(),
                    targetSpeakerId = row.getValue("target_speaker_id"),
                )
            }
        }
    }

    private fun requireColumns(present: Set<String>, required: List<String>, path: Path) {
        val missing = required - present
        require(missing.isEmpty()) { "$path is missing required columns: ${missing.joinToString()}" }
    }

    /** Adds file and line context to any validation failure, so a bad row is findable. */
    private inline fun <T> withRowContext(path: Path, lineNumber: Int, build: () -> T): T =
        try {
            build()
        } catch (failure: IllegalArgumentException) {
            throw IllegalArgumentException(
                "${path.fileName}, line $lineNumber: ${failure.message}",
                failure,
            )
        }

    private fun <T : Enum<T>> parseEnum(raw: String, values: List<T>, column: String): T =
        values.firstOrNull { it.name.equals(raw.replace('-', '_'), ignoreCase = true) }
            ?: throw IllegalArgumentException(
                "unknown $column '$raw'. Valid values: " +
                    values.joinToString { it.name.lowercase() },
            )
}
