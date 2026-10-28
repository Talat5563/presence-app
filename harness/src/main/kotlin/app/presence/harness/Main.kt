package app.presence.harness

import app.presence.audio.SAMPLE_RATE_HZ
import app.presence.domain.Language
import app.presence.domain.VoiceProfile
import app.presence.engine.DetectionEngine
import app.presence.engine.Embedding
import app.presence.engine.EvidenceAccumulator
import app.presence.engine.HouseholdMatcher
import app.presence.engine.SpeakerEmbedder
import app.presence.engine.VadGate
import app.presence.enrollment.EnrollmentClip
import app.presence.enrollment.EnrollmentPolicy
import app.presence.enrollment.EnrollmentResult
import app.presence.enrollment.EnrollmentService
import app.presence.onnx.OnnxModelSession
import app.presence.onnx.OnnxSpeakerEmbedder
import app.presence.onnx.SileroVad
import app.presence.audio.Framer
import java.nio.file.Path
import kotlin.io.path.createDirectories
import kotlin.io.path.exists
import kotlin.io.path.readText
import kotlin.io.path.writeText
import kotlin.system.exitProcess

private const val USAGE = """
Phase 0 accuracy harness.

Usage:
  ./gradlew :harness:run --args="--manifest DIR --models DIR --out DIR [--holdout h3,h4] [--evaluate dev|holdout] [--threshold N] [--input-name N] [--output-name N] [--embeddings FILE]"

  --manifest DIR     directory containing recordings.csv and events.csv; audio paths are
                     resolved relative to it
  --models DIR       directory containing silero_vad.onnx and speaker_embedding.onnx
  --out DIR          where report.md and results.csv are written
  --holdout LIST     comma-separated household ids reserved for final measurement only
  --evaluate MODE    "dev" (default) sweeps ten thresholds and tunes on every household not in
                     --holdout, reporting the best verdict per slice across the sweep.
                     "holdout" measures ONE pinned operating point, given by --threshold, only on
                     --holdout households. Sequence: choose a threshold by looking at dev sweep
                     results, then re-run with --evaluate holdout --threshold N to get the one
                     honest final number. Sweeping the holdout and reporting its best verdict would
                     be tuning on the data the holdout exists to protect, so holdout mode refuses to
                     sweep — it requires --threshold and reports exactly that one outcome per slice.
  --threshold N      the fire threshold to measure. Required with --evaluate holdout; ignored (the
                     sweep is used instead) in --evaluate dev.
  --input-name N     embedding model input tensor name (default: feats)
  --output-name N    embedding model output tensor name (default: embs)
  --embeddings FILE  also write every enrollment-clip and observation embedding to FILE (CSV),
                     for trying scoring changes offline without re-embedding the audio

Caveat: a detection in an unlabelled recording — including a target-child recording with no
labelled events at all — counts as a false alarm. That is the right default for an unlabelled
corpus, but it biases the verdict pessimistically wherever labelling is incomplete; the report's
Warnings section names any target-child recording with zero labelled events so this can be checked.

Run ./tools/fetch-models.sh first. See spec §9.1 for the manifest format.
"""

fun main(argv: Array<String>) {
    val args = try {
        parseArgs(argv)
    } catch (failure: IllegalArgumentException) {
        System.err.println("error: ${failure.message}")
        println(USAGE)
        exitProcess(2)
    }
    if (args.isEmpty() || "help" in args) {
        println(USAGE)
        return
    }
    for (required in listOf("manifest", "models", "out")) {
        if (required !in args) {
            System.err.println("error: --$required is required")
            println(USAGE)
            exitProcess(2)
        }
    }

    val manifestDir = Path.of(args.getValue("manifest"))
    val modelsDir = Path.of(args.getValue("models"))
    val outDir = Path.of(args.getValue("out"))

    val recordings = Manifest.readRecordings(manifestDir.resolve("recordings.csv"))
    val events = Manifest.readEvents(manifestDir.resolve("events.csv"))
    validateEvents(recordings, events)
    println("Loaded ${recordings.size} recordings and ${events.size} call events.")

    val holdout = args["holdout"]?.split(",")?.map { it.trim() }?.filter { it.isNotEmpty() }?.toSet().orEmpty()
    val households = recordings.map { it.householdId }.toSet()
    val unknownHoldout = holdout - households
    require(unknownHoldout.isEmpty()) {
        "--holdout names households not in the corpus: ${unknownHoldout.joinToString()}. " +
            "Known: ${households.sorted().joinToString()}"
    }
    if (holdout.isNotEmpty()) println("Holding out households: ${holdout.joinToString()}")

    val mode = args["evaluate"] ?: "dev"
    require(mode == "dev" || mode == "holdout") { "--evaluate must be 'dev' or 'holdout', was '$mode'" }
    require(mode == "dev" || holdout.isNotEmpty()) { "--evaluate holdout needs --holdout" }

    val pinnedThreshold = args["threshold"]?.toFloatOrNull()
    require(args["threshold"] == null || pinnedThreshold != null) {
        "--threshold must be a number, was '${args["threshold"]}'"
    }
    require(mode == "dev" || pinnedThreshold != null) {
        "--evaluate holdout requires --threshold: the operating point must be chosen on the dev " +
            "set, then measured once here. Sweeping the holdout and reporting its best verdict is " +
            "tuning on the data the holdout exists to protect."
    }

    val warnings = mutableListOf<String>()

    val targetsWithoutEvents = recordings
        .filter { it.role == RecordingRole.TARGET_CHILD && !it.contentType.isEnrollment }
        .map { it.path }
        .filterNot { path -> events.any { it.recordingPath == path } }
    if (targetsWithoutEvents.isNotEmpty()) {
        warnings += "${targetsWithoutEvents.size} target-child recording(s) have no labelled " +
            "events, so every detection in them counts as a false alarm: " +
            targetsWithoutEvents.take(5).joinToString() +
            if (targetsWithoutEvents.size > 5) ", ..." else ""
    }

    val dump = args["embeddings"]?.let { EmbeddingDump(Path.of(it)) }
    SileroVad(OnnxModelSession(modelsDir.resolve("silero_vad.onnx"))).use { vad ->
        OnnxSpeakerEmbedder(
            OnnxModelSession(modelsDir.resolve("speaker_embedding.onnx")),
            inputName = args["input-name"] ?: "feats",
            outputName = args["output-name"] ?: "embs",
        ).use { embedder ->
            val traces = ArrayList<Trace>()
            val totalToTrace = recordings.groupBy { it.householdId }.values.sumOf { householdRecordings ->
                val split = Evaluation.splitCorpus(householdRecordings)
                split.positives.size + split.negatives.size
            }
            var traced = 0

            for ((householdId, householdRecordings) in recordings.groupBy { it.householdId }) {
                // Checked on the manifest, not on the enrolled profiles: if one of two targets
                // failed to enrol, the other child's recordings would still be scored as positives.
                val targetSpeakers = householdRecordings
                    .filter { it.role == RecordingRole.TARGET_CHILD }
                    .map { it.speakerId }
                    .distinct()
                if (targetSpeakers.size > 1) {
                    val message = "SKIP household $householdId: $targetSpeakers are all target " +
                        "children — v1 listens for one; label the others SIBLING"
                    println(message)
                    warnings += message
                    continue
                }
                val split = Evaluation.splitCorpus(householdRecordings)
                val profiles = enrolProfiles(split.enrollment, manifestDir, vad, embedder, warnings, dump)
                val target = profiles.firstOrNull { it.role == VoiceProfile.Role.TARGET_CHILD }
                if (target == null) {
                    val message = "SKIP household $householdId: no target child enrolled"
                    println(message)
                    warnings += message
                    continue
                }
                val cohort = profiles - target
                if (cohort.isEmpty()) {
                    val message =
                        "SKIP household $householdId: no cohort — spec §5 requires the parent and siblings"
                    println(message)
                    warnings += message
                    continue
                }
                println("Household $householdId: target ${target.personId}, cohort ${cohort.map { it.personId }}")

                for (recording in split.positives + split.negatives) {
                    // One bad file must not abort a multi-hour run and discard every trace
                    // computed so far — WavReader.read is designed to be caught per file.
                    try {
                        traces += traceOf(recording, manifestDir, vad, embedder, target, cohort, dump)
                    } catch (failure: UnsupportedAudioException) {
                        val message = "$householdId: skipped ${recording.path} — ${failure.message}"
                        println("  $message")
                        warnings += message
                    }
                    traced += 1
                    if (traced % 25 == 0) println("  ... $traced/$totalToTrace recordings traced")
                }
            }
            dump?.close()

            // dev sweeps and tunes; holdout measures once, on data never used for tuning.
            val evaluated = when (mode) {
                "holdout" -> traces.filter { it.recording.householdId in holdout }
                else -> traces.filter { it.recording.householdId !in holdout }
            }
            // dev sweeps ten thresholds and reports the best; holdout measures exactly the one
            // pinned threshold chosen on dev — sweeping the holdout would be tuning on it.
            val thresholds = pinnedThreshold?.let { listOf(it) } ?: (2..20 step 2).map { it.toFloat() }
            val outcomes = buildList {
                addAll(Evaluation.sweep(evaluated, events, thresholds, "ALL"))
                addAll(Evaluation.sweep(evaluated, events, thresholds, "ENGLISH") {
                    it.language == Language.ENGLISH && !it.codeSwitched
                })
                addAll(Evaluation.sweep(evaluated, events, thresholds, "HEBREW") {
                    it.language == Language.HEBREW && !it.codeSwitched
                })
                addAll(Evaluation.sweep(evaluated, events, thresholds, "CODE_SWITCHED") { it.codeSwitched })
            }

            val provenanceFile = modelsDir.resolve("PROVENANCE.txt")
            val provenance = if (provenanceFile.exists()) provenanceFile.readText() else "unknown"

            outDir.createDirectories()
            outDir.resolve("report.md").writeText(Report.markdown(outcomes, provenance, warnings))
            outDir.resolve("results.csv").writeText(Report.csv(outcomes))
            outDir.resolve("observations.csv").writeText(Report.observationsCsv(evaluated))
            println("Wrote report.md, results.csv and observations.csv to $outDir")

            println()
            if (mode == "holdout") {
                println(
                    "holdout measurement at threshold $pinnedThreshold — this is the number " +
                        "(ENGLISH and HEBREW must each clear the bar independently — a system that " +
                        "passes ALL or one language and fails the other has not passed):",
                )
                for (sliceLabel in listOf("ALL", "ENGLISH", "HEBREW", "CODE_SWITCHED")) {
                    val outcome = outcomes.firstOrNull { it.label == sliceLabel }
                    println("  $sliceLabel: ${outcome?.verdict ?: "none"}")
                }
            } else {
                println(
                    "dev sweep — best verdict per slice (an operating point chosen here must be " +
                        "re-measured with --evaluate holdout --threshold N) (ENGLISH and HEBREW must " +
                        "each clear the bar independently — a system that passes ALL or one language " +
                        "and fails the other has not passed):",
                )
                for (sliceLabel in listOf("ALL", "ENGLISH", "HEBREW", "CODE_SWITCHED")) {
                    val best = outcomes.filter { it.label == sliceLabel }.minByOrNull { it.verdict.rank }
                    println("  $sliceLabel: ${best?.verdict ?: "none"}")
                }
            }
        }
    }
}

private fun enrolProfiles(
    enrollment: List<Recording>,
    root: Path,
    vad: SileroVad,
    embedder: SpeakerEmbedder,
    warnings: MutableList<String>,
    dump: EmbeddingDump? = null,
): List<VoiceProfile> =
    enrollment.groupBy { it.speakerId }.mapNotNull { (speakerId, group) ->
        // Runtime observations are VAD-trimmed windows of at most 3 s each; embedding a whole
        // unfiltered read passage in one shot would be a systematic mismatch that depresses
        // targetSimilarity against the absolute minTargetSimilarity gate. Running enrollment audio
        // through the same VadGate the runtime uses, and embedding per segment, keeps the two
        // comparable.
        val clips = group.flatMap { recording ->
            val language = recording.language ?: return@flatMap emptyList()
            val samples = try {
                WavReader.read(root.resolve(recording.path))
            } catch (failure: UnsupportedAudioException) {
                warnings += "$speakerId: skipped ${recording.path} — ${failure.message}"
                return@flatMap emptyList()
            }

            vad.reset()
            val vadGate = VadGate(vad)
            val segments = buildList {
                val framer = Framer(vad.frameSize)
                for (frame in framer.push(samples)) vadGate.accept(frame)?.let { add(it) }
                vadGate.flush()?.let { add(it) }
            }

            segments.filter { it.durationMs >= 800 }.map { segment ->
                val embedding = embedder.embed(segment.samples)
                dump?.write("enroll", recording, timestampMs = 0, embedding)
                EnrollmentClip(
                    language = language,
                    embedding = embedding,
                    speechSeconds = segment.samples.size / SAMPLE_RATE_HZ.toDouble(),
                )
            }
        }
        if (clips.isEmpty()) {
            warnings += "$speakerId: no usable enrollment audio"
            return@mapNotNull null
        }
        // Require only the languages this person actually has material for. Demanding both would
        // flag every monolingual household, drowning real shortfalls in expected noise.
        val policy = EnrollmentPolicy(
            minSecondsPerLanguage = 20.0,
            requiredLanguages = clips.map { it.language }.toSet(),
        )
        when (val result = EnrollmentService(policy).enroll(speakerId, group.first().role.toProfileRole(), clips)) {
            is EnrollmentResult.Complete -> result.profile
            is EnrollmentResult.Incomplete -> {
                val message = "$speakerId: enrolled partially, short by ${result.shortfallSeconds}"
                println("  $message")
                warnings += message
                result.profile
            }
        }
    }

private fun traceOf(
    recording: Recording,
    root: Path,
    vad: SileroVad,
    embedder: SpeakerEmbedder,
    target: VoiceProfile,
    cohort: List<VoiceProfile>,
    dump: EmbeddingDump? = null,
): Trace {
    val samples = WavReader.read(root.resolve(recording.path))
    val observations = ArrayList<TimedScore>()

    // The engine embeds, then scores, then reports — so captured embeddings pair with
    // observations by position.
    val embedded = ArrayList<Embedding>()
    val capturing = object : SpeakerEmbedder by embedder {
        override fun embed(samples: FloatArray) = embedder.embed(samples).also { embedded += it }
    }

    vad.reset()
    val engine = DetectionEngine(
        vadGate = VadGate(vad),
        embedder = if (dump != null) capturing else embedder,
        matcher = HouseholdMatcher(target, cohort),
        accumulator = EvidenceAccumulator(),
        targetPersonId = target.personId,
        observationListener = { timestampMs, score -> observations += TimedScore(timestampMs, score) },
    )

    val framer = Framer(vad.frameSize)
    for (frame in framer.push(samples)) engine.accept(frame)
    engine.flush()

    if (dump != null) {
        check(embedded.size == observations.size) {
            "${recording.path}: ${embedded.size} embeddings for ${observations.size} observations"
        }
        observations.forEachIndexed { index, timed -> dump.write("obs", recording, timed.timestampMs, embedded[index]) }
    }

    return Trace(recording, observations, samples.size.toDouble() / SAMPLE_RATE_HZ)
}

private val KNOWN_FLAGS = setOf(
    "help", "manifest", "models", "out", "holdout", "evaluate", "threshold", "input-name", "output-name",
    "embeddings",
)

internal fun parseArgs(argv: Array<String>): Map<String, String> = parseFlags(argv, KNOWN_FLAGS)

/** Parses `--flag value` pairs, plus a bare `--help`, rejecting any flag not in [known]. */
internal fun parseFlags(argv: Array<String>, known: Set<String>): Map<String, String> {
    val args = mutableMapOf<String, String>()
    var index = 0
    while (index < argv.size) {
        val token = argv[index]
        require(token.startsWith("--")) { "unexpected argument: $token" }
        val key = token.removePrefix("--")
        require(key in known) {
            "unknown flag --$key. Known flags: ${known.sorted().joinToString { "--$it" }}"
        }
        if (key == "help") {
            args["help"] = "true"
            index += 1
            continue
        }
        require(index + 1 < argv.size) { "missing value for --$key" }
        val value = argv[index + 1]
        // A flag-shaped value almost always means the real value was forgotten.
        require(!value.startsWith("--")) { "--$key expects a value but was followed by $value" }
        args[key] = value
        index += 2
    }
    return args
}
