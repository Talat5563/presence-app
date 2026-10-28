package app.presence.harness

import app.presence.domain.Language

/** One CHAT transcript and the session it describes. [id] must be unique within its household. */
data class ChildesSession(val id: String, val transcript: ChatTranscript)

/**
 * A household to import.
 *
 * @param target the child listened for, matched against each session's participant name or code.
 *   Matching by name is what keeps one child the target across sessions in which CHAT gives the
 *   code `CHI` to whichever sibling the session focused on.
 */
data class HouseholdSource(val householdId: String, val target: String, val sessions: List<ChildesSession>)

/**
 * @param enrollSeconds target-child speech to set aside for enrollment before evaluation starts
 * @param maxEnrollSecondsPerSpeaker enrollment audio kept per speaker and language; more adds
 *   size, not accuracy, to a centroid
 * @param maxEnrollSessionFraction enrollment never takes more than this share of a household's
 *   sessions, so a quiet child cannot consume the whole evaluation set
 * @param episodeQuietMs target-child silence that ends one episode and lets the next begin. An
 *   episode stands for "the child started talking to the parent".
 * @param episodeMaxMs an episode is cut to the latency budget: a detection later than that is a
 *   miss anyway, and audio past it would only add exposure in which a late fire counts as a false
 *   alarm
 * @param minSpeechMs files with less speech than this are not written
 * @param maxStitchGapMs silence inserted between stitched utterances is the real gap, capped at
 *   this. The real gap often contains other speakers, which must not leak into a single-speaker
 *   file.
 */
data class ImportOptions(
    val enrollSeconds: Double = 60.0,
    val maxEnrollSecondsPerSpeaker: Double = 300.0,
    val maxEnrollSessionFraction: Double = 0.5,
    val episodeQuietMs: Long = 10_000,
    val episodeMaxMs: Long = Metrics.DEFAULT_LATENCY_BUDGET_MS,
    val minSpeechMs: Long = 1_000,
    val maxStitchGapMs: Long = 1_000,
) {
    init {
        require(enrollSeconds > 0) { "enrollSeconds must be positive" }
        require(maxEnrollSessionFraction > 0 && maxEnrollSessionFraction < 1) {
            "maxEnrollSessionFraction must be in (0, 1)"
        }
        require(episodeQuietMs > 0 && episodeMaxMs > 0 && minSpeechMs > 0 && maxStitchGapMs >= 0) {
            "durations must be positive"
        }
    }
}

/** A stretch of one session's media to copy into an output file, after [gapBeforeMs] of silence. */
data class Piece(val sessionId: String, val startMs: Long, val endMs: Long, val gapBeforeMs: Long) {
    val durationMs: Long get() = endMs - startMs
}

/** One audio file to write, the manifest row describing it, and its call event if it has one. */
data class PlannedFile(val recording: Recording, val pieces: List<Piece>, val event: CallEvent?) {
    val speechMs: Long get() = pieces.sumOf { it.durationMs }
}

data class ImportPlan(val files: List<PlannedFile>, val warnings: List<String>)

/**
 * Turns CHILDES sessions into a harness corpus, as speaker episodes:
 *
 * - **Enrollment:** a household's earliest sessions are set aside whole, and each household
 *   member's speech in them becomes one enrollment file per language. Whole sessions rather than
 *   individual utterances, so the same day's room and microphone never sit on both sides of the
 *   split.
 * - **Positives:** every later stretch of the target child talking becomes its own file, cut to
 *   the latency budget, with one call event at its start.
 * - **Negatives:** every other speaker's speech in a later session becomes one file per session,
 *   in which any detection is a false alarm.
 *
 * Utterances overlapping another speaker are dropped, so every file holds one voice. Stitching
 * removes the pauses in which others spoke, so false alarms are counted per hour of *speech*,
 * which is pessimistic against a real home's mostly-quiet hours.
 */
object ChildesPlanner {

    fun plan(households: List<HouseholdSource>, options: ImportOptions = ImportOptions()): ImportPlan {
        val files = ArrayList<PlannedFile>()
        val warnings = ArrayList<String>()
        for (household in households) planHousehold(household, options, files, warnings)
        return ImportPlan(files, warnings)
    }

    private data class Speaker(val id: String, val role: RecordingRole)

    private fun planHousehold(
        household: HouseholdSource,
        options: ImportOptions,
        files: MutableList<PlannedFile>,
        warnings: MutableList<String>,
    ) {
        val id = household.householdId
        val duplicateSessions = household.sessions.groupingBy { it.id }.eachCount().filterValues { it > 1 }.keys
        require(duplicateSessions.isEmpty()) { "household $id repeats session ids: $duplicateSessions" }

        val usable = household.sessions
            .filter { session ->
                val transcript = session.transcript
                when {
                    transcript.mediaName == null -> {
                        warnings += "$id/${session.id}: no linked media, skipped"
                        false
                    }
                    transcript.primaryLanguage == null -> {
                        warnings += "$id/${session.id}: languages ${transcript.languages} unsupported, skipped"
                        false
                    }
                    transcript.utterances.isEmpty() -> {
                        warnings += "$id/${session.id}: no time-aligned utterances, skipped"
                        false
                    }
                    else -> true
                }
            }
            .sortedWith(compareBy({ it.transcript.date == null }, { it.transcript.date }, { it.id }))

        val clean = usable.associate { it.id to withoutOverlaps(it.transcript.utterances) }
        fun isTarget(session: ChildesSession, code: String): Boolean {
            val participant = session.transcript.participants[code] ?: return code.equals(household.target, true)
            return participant.name.equals(household.target, true) || participant.code.equals(household.target, true)
        }
        fun targetSpeechMs(session: ChildesSession) =
            clean.getValue(session.id).filter { isTarget(session, it.speakerCode) }.sumOf { it.durationMs }

        if (usable.none { targetSpeechMs(it) > 0 }) {
            warnings += "SKIP household $id: target '${household.target}' never speaks in a usable session"
            return
        }

        /** Null for a speaker of unknown identity, whose speech goes into no file. */
        fun speaker(session: ChildesSession, code: String): Speaker? {
            val participant = session.transcript.participants[code]
            // The code, never the role, when there is no name: MacWhinney lists both brothers as
            // nameless `Target_Child`s, and only their codes (CHI, MAR) tell them apart.
            val key = participant?.name ?: code
            val role = if (isTarget(session, code)) RecordingRole.TARGET_CHILD else roleOf(participant?.role) ?: return null
            return Speaker("$id-${sanitize(key)}".lowercase(), role)
        }
        fun householdSpeechMs(session: ChildesSession): Map<Speaker, Long> = clean.getValue(session.id)
            .mapNotNull { utterance ->
                speaker(session, utterance.speakerCode)?.takeIf { it.role.isHouseholdMember }?.let { it to utterance.durationMs }
            }
            .groupBy({ it.first }, { it.second })
            .mapValues { (_, durations) -> durations.sum() }

        // Enrollment takes the earliest sessions until the core household — target, siblings and
        // parents — each has enough speech. Stopping once the target alone has enough would leave a
        // parent who missed the first session unenrolled, and spec §5 requires the whole household
        // in the cohort. Grandparents and other relatives are enrolled if they happen to speak in
        // those sessions but are not waited for: in MacWhinney they appear twice in 157 sessions,
        // and waiting would hand half the corpus to enrollment.
        val core = setOf(RecordingRole.TARGET_CHILD, RecordingRole.SIBLING, RecordingRole.PARENT)
        val members = usable.flatMap { session -> householdSpeechMs(session).keys.filter { it.role in core }.map { it.id } }.toSet()
        val wantedMs = (options.enrollSeconds * 1_000).toLong()
        val maxEnrollSessions = maxOf(1, (usable.size * options.maxEnrollSessionFraction).toInt())
        val enrollment = ArrayList<ChildesSession>()
        val enrolledMs = HashMap<String, Long>()
        for (session in usable) {
            if (members.all { (enrolledMs[it] ?: 0L) >= wantedMs } || enrollment.size >= maxEnrollSessions) break
            enrollment += session
            for ((member, ms) in householdSpeechMs(session)) enrolledMs[member.id] = (enrolledMs[member.id] ?: 0L) + ms
        }
        for (member in members.sorted()) {
            val ms = enrolledMs[member] ?: 0L
            if (ms < wantedMs) {
                warnings += "$id: $member enrolled on only ${ms / 1_000} s of speech " +
                    "(wanted ${options.enrollSeconds.toInt()} s) from ${enrollment.size} session(s)"
            }
        }
        val evaluation = usable - enrollment.toSet()
        if (evaluation.isEmpty()) {
            warnings += "SKIP household $id: every usable session was needed for enrollment"
            return
        }

        // Enrollment: one file per household member and language, across all enrollment sessions.
        val enrollPieces = LinkedHashMap<Pair<String, Language>, MutableList<Pair<ChildesSession, ChatUtterance>>>()
        val enrollRoles = HashMap<String, RecordingRole>()
        for (session in enrollment) {
            val language = session.transcript.primaryLanguage!!
            for (utterance in clean.getValue(session.id)) {
                val who = speaker(session, utterance.speakerCode) ?: continue
                if (!who.role.isHouseholdMember) continue
                enrollRoles[who.id] = who.role
                enrollPieces.getOrPut(who.id to language) { ArrayList() } += session to utterance
            }
        }
        for ((key, spoken) in enrollPieces) {
            val (speakerId, language) = key
            var keptMs = 0L
            val pieces = spoken.groupBy({ it.first }, { it.second }).entries
                .flatMapIndexed { sessionIndex, (session, utterances) ->
                    val stitched = stitch(session.id, utterances, options)
                    // A session boundary gets the full gap, as though the speaker paused.
                    if (sessionIndex == 0) stitched else listOf(stitched.first().copy(gapBeforeMs = options.maxStitchGapMs)) + stitched.drop(1)
                }
                // A centroid needs a minute or two; hours of a talkative parent add nothing but size.
                .takeWhile { piece -> (keptMs < options.maxEnrollSecondsPerSpeaker * 1_000).also { keptMs += piece.durationMs } }
            val file = PlannedFile(
                recording = recording(
                    path = "wav/$id/enroll/${speakerId.removePrefix("$id-")}-${language.name.lowercase()}.wav",
                    householdId = id,
                    speakerId = speakerId,
                    role = enrollRoles.getValue(speakerId),
                    language = language,
                    contentType = ContentType.ENROLLMENT,
                    date = enrollment.first().transcript.date?.toString().orEmpty(),
                ),
                pieces = pieces,
                event = null,
            )
            if (file.speechMs >= options.minSpeechMs) files += file
        }

        for (session in evaluation) {
            val language = session.transcript.primaryLanguage!!
            val date = session.transcript.date?.toString().orEmpty()
            val bySpeaker = clean.getValue(session.id)
                .mapNotNull { utterance -> speaker(session, utterance.speakerCode)?.let { it to utterance } }
                .groupBy({ it.first }, { it.second })
            val directory = "wav/$id/${sanitize(session.id)}"

            for ((who, utterances) in bySpeaker) {
                val speakerId = who.id
                val name = speakerId.removePrefix("$id-")
                if (who.role == RecordingRole.TARGET_CHILD) {
                    episodes(session.id, utterances, options).forEachIndexed { index, pieces ->
                        val path = "$directory/$name-ep${(index + 1).toString().padStart(3, '0')}.wav"
                        files += PlannedFile(
                            recording = recording(path, id, speakerId, who.role, language, ContentType.SPONTANEOUS, date),
                            pieces = pieces,
                            event = CallEvent(path, startMs = 0, targetSpeakerId = speakerId),
                        )
                    }
                } else {
                    val file = PlannedFile(
                        recording = recording("$directory/$name.wav", id, speakerId, who.role, language, ContentType.SPONTANEOUS, date),
                        pieces = stitch(session.id, utterances, options),
                        event = null,
                    )
                    if (file.speechMs >= options.minSpeechMs) files += file
                }
            }
        }
    }

    /**
     * Splits one session's target-child utterances into episodes. An episode starts after at least
     * [ImportOptions.episodeQuietMs] without the target speaking, and is cut at
     * [ImportOptions.episodeMaxMs] of stitched audio.
     */
    internal fun episodes(sessionId: String, utterances: List<ChatUtterance>, options: ImportOptions): List<List<Piece>> {
        val groups = ArrayList<MutableList<ChatUtterance>>()
        var previousEnd: Long? = null
        for (utterance in utterances.sortedBy { it.startMs }) {
            val end = previousEnd
            if (end == null || utterance.startMs - end >= options.episodeQuietMs) groups += ArrayList<ChatUtterance>()
            groups.last() += utterance
            previousEnd = maxOf(end ?: 0L, utterance.endMs)
        }

        return groups.mapNotNull { group ->
            val pieces = ArrayList<Piece>()
            var lengthMs = 0L
            for (piece in stitch(sessionId, group, options)) {
                val room = options.episodeMaxMs - lengthMs - piece.gapBeforeMs
                if (room <= 0) break
                val kept = if (piece.durationMs > room) piece.copy(endMs = piece.startMs + room) else piece
                pieces += kept
                lengthMs += kept.gapBeforeMs + kept.durationMs
            }
            pieces.takeIf { kept -> kept.sumOf { it.durationMs } >= options.minSpeechMs }
        }
    }

    /** Orders utterances into pieces, each preceded by the real gap capped at the stitch limit. */
    internal fun stitch(sessionId: String, utterances: List<ChatUtterance>, options: ImportOptions): List<Piece> {
        var previousEnd: Long? = null
        return utterances.sortedBy { it.startMs }.map { utterance ->
            val gap = previousEnd?.let { (utterance.startMs - it).coerceIn(0, options.maxStitchGapMs) } ?: 0L
            previousEnd = utterance.endMs
            Piece(sessionId, utterance.startMs, utterance.endMs, gap)
        }
    }

    /** Drops every utterance that overlaps a different speaker's, so each file holds one voice. */
    internal fun withoutOverlaps(utterances: List<ChatUtterance>): List<ChatUtterance> {
        val sorted = utterances.sortedBy { it.startMs }
        return sorted.filterIndexed { index, utterance ->
            sorted.indices.none { other ->
                other != index &&
                    sorted[other].speakerCode != utterance.speakerCode &&
                    sorted[other].startMs < utterance.endMs &&
                    utterance.startMs < sorted[other].endMs
            }
        }
    }

    /**
     * Maps a CHAT role onto the harness's. A non-target `Target_Child` is a sibling: CHAT gives
     * that role to whichever child a session focused on. A bare `Child` is a stranger, since CHAT
     * uses it for friends and playmates as often as for siblings, and wrongly enrolling a visitor
     * would flatter the result.
     *
     * Null for a speaker whose identity is unknown — `Participant`, as automatic diarization labels
     * an unattributed voice, `Unidentified` and `Uncertain`. Such a voice may be the target child,
     * and counting the target as a stranger would score correct detections as false alarms, so
     * their speech goes into no file.
     */
    internal fun roleOf(chatRole: String?): RecordingRole? = when (chatRole?.lowercase()) {
        "target_child", "brother", "sister", "sibling", "twin" -> RecordingRole.SIBLING
        "mother", "father" -> RecordingRole.PARENT
        "grandmother", "grandfather", "grandparent", "relative", "aunt", "uncle" -> RecordingRole.OTHER_HOUSEHOLD
        "media" -> RecordingRole.MEDIA
        "participant", "unidentified", "uncertain" -> null
        else -> RecordingRole.STRANGER
    }

    private fun recording(
        path: String,
        householdId: String,
        speakerId: String,
        role: RecordingRole,
        language: Language,
        contentType: ContentType,
        date: String,
    ) = Recording(
        path = path,
        householdId = householdId,
        speakerId = speakerId,
        role = role,
        language = language,
        codeSwitched = false,
        contentType = contentType,
        distanceMeters = null,
        placement = "childes",
        background = "childes",
        room = "home",
        date = date,
    )

    /** Keeps paths and ids to characters the strict manifest CSV and every filesystem accept. */
    internal fun sanitize(value: String): String = value.replace(Regex("[^A-Za-z0-9_-]"), "_")
}
