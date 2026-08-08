package app.presence.harness

import app.presence.audio.SAMPLE_RATE_HZ
import app.presence.domain.Language
import app.presence.domain.VoiceProfile
import app.presence.engine.DetectionEngine
import app.presence.engine.EvidenceAccumulator
import app.presence.engine.HouseholdMatcher
import app.presence.engine.Observation
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
  ./gradlew :harness:run --args="--manifest DIR --models DIR --out DIR [--holdout h3,h4] [--input-name N] [--output-name N]"

  --manifest DIR   directory containing recordings.csv and events.csv; audio paths are
                   resolved relative to it
  --models DIR     directory containing silero_vad.onnx and speaker_embedding.onnx
  --out DIR        where report.md and results.csv are written
  --holdout LIST   comma-separated household ids reserved for final measurement only
  --input-name N   embedding model input tensor name (default: feats)
  --output-name N  embedding model output tensor name (default: embs)

Run ./tools/fetch-models.sh first. See spec §9.1 for the manifest format.
"""

fun main(argv: Array<String>) {
    val args = parseArgs(argv)
    if (args.isEmpty() || "help" in args) {
        println(USAGE)
        return
    }

    val manifestDir = Path.of(args.getValue("manifest"))
    val modelsDir = Path.of(args.getValue("models"))
    val outDir = Path.of(args.getValue("out"))
    val holdout = args["holdout"]?.split(",")?.map { it.trim() }?.toSet().orEmpty()

    val recordings = Manifest.readRecordings(manifestDir.resolve("recordings.csv"))
    val events = Manifest.readEvents(manifestDir.resolve("events.csv"))
    println("Loaded ${recordings.size} recordings and ${events.size} call events.")
    if (holdout.isNotEmpty()) println("Holding out households: ${holdout.joinToString()}")

    SileroVad(OnnxModelSession(modelsDir.resolve("silero_vad.onnx"))).use { vad ->
        OnnxSpeakerEmbedder(
            OnnxModelSession(modelsDir.resolve("speaker_embedding.onnx")),
            inputName = args["input-name"] ?: "feats",
            outputName = args["output-name"] ?: "embs",
        ).use { embedder ->
            val traces = ArrayList<Trace>()

            for ((householdId, householdRecordings) in recordings.groupBy { it.householdId }) {
                val split = Evaluation.splitCorpus(householdRecordings)
                val profiles = enrolProfiles(split.enrollment, manifestDir, embedder)
                val target = profiles.firstOrNull { it.role == VoiceProfile.Role.TARGET_CHILD }
                if (target == null) {
                    println("SKIP household $householdId: no target child enrolled")
                    continue
                }
                val cohort = profiles - target
                if (cohort.isEmpty()) {
                    println("SKIP household $householdId: no cohort — spec §5 requires the parent and siblings")
                    continue
                }
                println("Household $householdId: target ${target.personId}, cohort ${cohort.map { it.personId }}")

                for (recording in split.positives + split.negatives) {
                    traces += traceOf(recording, manifestDir, vad, embedder, target, cohort)
                }
            }

            val evaluated = traces.filter { it.recording.householdId !in holdout }
            val thresholds = (2..20 step 2).map { it.toFloat() }
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
            outDir.resolve("report.md").writeText(Report.markdown(outcomes, provenance))
            outDir.resolve("results.csv").writeText(Report.csv(outcomes))
            println("Wrote ${outDir.resolve("report.md")} and ${outDir.resolve("results.csv")}")

            val best = outcomes.filter { it.label == "ALL" }.minByOrNull { it.verdict.ordinal }
            println("Best overall verdict: ${best?.verdict ?: "none"}")
        }
    }
}

private fun enrolProfiles(
    enrollment: List<Recording>,
    root: Path,
    embedder: SpeakerEmbedder,
): List<VoiceProfile> {
    val service = EnrollmentService(EnrollmentPolicy(minSecondsPerLanguage = 20.0))
    return enrollment.groupBy { it.speakerId }.mapNotNull { (speakerId, group) ->
        val clips = group.mapNotNull { recording ->
            val language = recording.language ?: return@mapNotNull null
            val samples = WavReader.read(root.resolve(recording.path))
            if (samples.size < SAMPLE_RATE_HZ) return@mapNotNull null
            EnrollmentClip(language, embedder.embed(samples), samples.size.toDouble() / SAMPLE_RATE_HZ)
        }
        if (clips.isEmpty()) return@mapNotNull null
        when (val result = service.enroll(speakerId, group.first().role.toProfileRole(), clips)) {
            is EnrollmentResult.Complete -> result.profile
            is EnrollmentResult.Incomplete -> {
                println("  $speakerId enrolled partially; short by ${result.shortfallSeconds}")
                result.profile
            }
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
): Trace {
    val samples = WavReader.read(root.resolve(recording.path))
    val observations = ArrayList<Observation>()

    vad.reset()
    val engine = DetectionEngine(
        vadGate = VadGate(vad),
        embedder = embedder,
        matcher = HouseholdMatcher(target, cohort),
        accumulator = EvidenceAccumulator(),
        targetPersonId = target.personId,
        observationListener = { observations += it },
    )

    val framer = Framer(vad.frameSize)
    for (frame in framer.push(samples)) engine.accept(frame)
    engine.flush()

    return Trace(recording, observations, samples.size.toDouble() / SAMPLE_RATE_HZ)
}

private fun parseArgs(argv: Array<String>): Map<String, String> {
    val args = mutableMapOf<String, String>()
    var index = 0
    while (index < argv.size) {
        val token = argv[index]
        if (!token.startsWith("--")) {
            System.err.println("unexpected argument: $token")
            exitProcess(1)
        }
        val key = token.removePrefix("--")
        if (key == "help") {
            args["help"] = "true"
            index += 1
        } else {
            require(index + 1 < argv.size) { "missing value for --$key" }
            args[key] = argv[index + 1]
            index += 2
        }
    }
    return args
}
