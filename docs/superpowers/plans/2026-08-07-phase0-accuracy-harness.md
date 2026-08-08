# Phase 0 Accuracy Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the real `DetectionEngine` as Kotlin Multiplatform common code plus a JVM evaluation harness that measures false alarms per hour and miss rate on child speech, producing the go/no-go answer defined in spec §4.3.

**Architecture:** The engine is pure `commonMain` Kotlin — audio framing, VAD gating, speaker embedding comparison, household cohort scoring, and log-likelihood evidence accumulation — with model inference behind interfaces so tests use fakes and production uses ONNX Runtime. A JVM-only `:harness` module reads WAV corpora, drives the engine, sweeps operating points, and writes a report. Nothing here is throwaway: this is the shipping engine from spec §3, validated before any UI exists.

**Tech Stack:** Kotlin Multiplatform 2.1.0 (JVM target now, Android and iOS later), Gradle 8.11 with Kotlin DSL and a version catalog, ONNX Runtime Java 1.20.0, `kotlin("test")` on JUnit Platform, `javax.sound.sampled` for WAV I/O.

## Global Constraints

Every task's requirements implicitly include these. Values are copied verbatim from the spec.

- **All audio is mono PCM at 16 kHz**, as floats in `[-1.0, 1.0]`. Reject anything else loudly (spec §9.1).
- **No network access from engine code.** The harness may download models via an explicit script; the engine never does (spec §11.1).
- **No raw audio is persisted** by engine code. Embeddings only (spec §11.2).
- **Real family audio never enters version control.** `.gitignore` already blocks `*.wav`, `*.m4a`, `**/voice-profiles/`, `**/enrollment-audio/`. Do not weaken it. Test fixtures are *generated in code*, not committed (spec §11.5).
- **Both languages must clear every criterion independently** — English, Hebrew, and code-switched speech are reported separately, never pooled (spec §4.3).
- **Per-language sub-centroids, never a pooled centroid.** Cross-lingual mismatch is a 40–49% domain shift; pooling blurs the profile (spec §4.2).
- **EER is not a product target.** Report false alarms per hour and miss rate per real event. EER is computed only for comparison with published work (spec §4.1).
- **Package root is `app.presence`.** Engine code lives in `app.presence.{audio,domain,engine,enrollment}`; harness code in `app.presence.harness`.
- **Go / no-go thresholds (spec §4.3),** stated as the conjunctions the code actually applies: **ship** at ≤ 0.5 false alarms per hour *and* ≥ 80% of real events detected within 10 s; **iterate** at ≤ 2 false alarms per hour *and* ≥ 50% detection; **fall back** otherwise. Both conditions are required at each tier — a system with 5 false alarms per hour is unusable whatever its detection rate, and one detecting 40% of events is useless however quiet it is.
- **Two test suites, and no test ever skips itself.** The default suite runs pure-logic tests that use fakes — no models, no network, green on a fresh clone. Tests needing a real model file are annotated `@Tag("model")`, excluded from the default suite, and run by `./gradlew integrationTest`, which **fails loudly** if the model is missing. A test that prints "SKIP" and returns still reports as passing, which is worse than no test at all.
- **Task names:** `:shared` is a Kotlin Multiplatform module, so its default test task is **`jvmTest`, not `test`** — a root `./gradlew test` would silently run nothing there. Use `./gradlew check` for everything, or `:shared:jvmTest` / `:harness:test` when targeting one module. `:shared:integrationTest` and `:harness:integrationTest` run the model-tagged tests.
- **Do not guess tensor names or shapes.** They were verified against the real model files and are recorded below. If a model is ever swapped, re-verify rather than assume.

## Verified environment and model facts

These are established, not assumptions. Implementers should rely on them rather than re-deriving.

**Toolchain** — already installed and on `PATH`: Temurin JDK 17 at `~/.local/toolchains/jdk-17.0.20+8`
and Gradle 8.11 at `~/.local/toolchains/gradle-8.11`. `java -version` reports 17 and `gradle
--version` reports 8.11. If a shell reports Java 8, re-source `~/.bashrc`.

**Models** — already downloaded into `models/` (git-ignored), with `models/PROVENANCE.txt` recording
their URLs:

| File | Signature | Notes |
|---|---|---|
| `models/silero_vad.onnx` | in `input [B, samples]`, `state [2, B, 128]`, `sr` (int64 scalar); out `output [B, 1]`, `stateN` | Silero **v5** — a single `state` tensor, not v4's separate `h`/`c` |
| `models/speaker_embedding.onnx` | in `feats [B, T, 80]`; out `embs [B, 256]` | WeSpeaker `wespeaker_en_voxceleb_resnet34_LM`, 26 MB |

**The embedding model consumes 80-bin log-mel filterbank features, not raw audio.** Every candidate
checked (WeSpeaker, 3D-Speaker CAM++, NeMo SpeakerNet, NeMo TitaNet) works this way, so Task 13's
front end is unavoidable rather than a consequence of the model choice.

WeSpeaker was chosen deliberately: L3 fine-tuning (spec §4.1) trains in WeSpeaker and exports ONNX,
so the improved model will be a drop-in replacement with the same tensor shapes.

## Design note: interfaces instead of `expect`/`actual`

Spec §3 lists `ModelRunner` under `platform/ (expect)`. This plan uses **plain interfaces in `commonMain` with platform implementations injected via constructor** instead. Same boundary, same portability, but fakes can be substituted in tests — which is the entire point of building the engine in common code. `AudioCapture` will still need `expect`/`actual` when the Android app arrives; that is out of scope here because the harness feeds audio from files.

## File Structure

```
settings.gradle.kts                 module wiring, repositories
build.gradle.kts                    root, plugins declared but not applied
gradle.properties                   JVM args, Kotlin code style
gradle/libs.versions.toml           version catalog — single source of dependency truth

shared/build.gradle.kts             KMP module, JVM target for now
shared/src/commonMain/kotlin/app/presence/
  audio/AudioTypes.kt               sample rate constant, ms<->samples conversion
  audio/Framer.kt                   stream -> fixed-size frames, stable across pushes
  audio/Fft.kt                      radix-2 FFT and real power spectrum
  audio/MelFilterBank.kt            Kaldi-compatible triangular mel filters
  audio/FbankExtractor.kt           80-bin log-mel features — what the model actually eats
  domain/Language.kt                ENGLISH, HEBREW
  domain/VoiceProfile.kt            per-language centroids, negatives, role
  domain/DetectionEvent.kt          what the engine emits
  engine/Embedding.kt               L2-normalized vector, cosine, centroid
  engine/SpeakerEmbedder.kt         interface
  engine/VoiceActivityDetector.kt   interface
  engine/VadGate.kt                 hysteresis segmentation into SpeechSegment
  engine/HouseholdMatcher.kt        L1: cohort scoring -> log-likelihood ratio
  engine/EvidenceAccumulator.kt     L2: rolling-window LLR sum, cooldown
  engine/DetectionEngine.kt         composes the pipeline
  enrollment/EnrollmentService.kt   clips -> VoiceProfile, per-language shortfall
shared/src/jvmMain/kotlin/app/presence/onnx/
  OnnxModelSession.kt               ORT session lifecycle + signature introspection
  SileroVad.kt                      VoiceActivityDetector via ONNX
  OnnxSpeakerEmbedder.kt            SpeakerEmbedder via ONNX

harness/build.gradle.kts            JVM application module
harness/src/main/kotlin/app/presence/harness/
  Csv.kt                            strict comma-only CSV reader with clear errors
  Manifest.kt                       recordings.csv + events.csv -> typed records
  WavReader.kt                      16 kHz mono float decoding, stereo downmix
  Metrics.kt                        FA/hour, miss rate, latency, EER
  Evaluation.kt                     runs the engine over a manifest, sweeps thresholds
  Report.kt                         markdown + CSV output, split by language
  Main.kt                           CLI entry point (parseArgs is internal, so it is testable)

tools/list-models.sh                enumerate sherpa-onnx model release assets
tools/fetch-models.sh               download chosen models into models/
models/                             gitignored — *.onnx is already excluded
```

---

### Task 1: Project scaffold and a passing test

**Files:**
- Create: `settings.gradle.kts`, `build.gradle.kts`, `gradle.properties`, `gradle/libs.versions.toml`
- Create: `shared/build.gradle.kts`
- Create: `shared/src/commonMain/kotlin/app/presence/audio/AudioTypes.kt`
- Test: `shared/src/commonTest/kotlin/app/presence/audio/AudioTypesTest.kt`

**Interfaces:**
- Consumes: nothing.
- Produces: `app.presence.audio.SAMPLE_RATE_HZ: Int`, `millisToSamples(ms: Long): Int`, `samplesToMillis(sampleCount: Int): Long`. Every later task uses these.

- [ ] **Step 1: Create the Gradle wrapper**

Run from the repository root:

```bash
gradle wrapper --gradle-version 8.11
```

If `gradle` is not installed, install it first (`sdk install gradle 8.11`, `brew install gradle`, or download from gradle.org). The wrapper files must be committed — `.gitignore` already has `!gradle/wrapper/gradle-wrapper.jar`.

- [ ] **Step 2: Write the version catalog**

`gradle/libs.versions.toml`:

```toml
[versions]
kotlin = "2.1.0"
onnxruntime = "1.20.0"

[libraries]
onnxruntime = { module = "com.microsoft.onnxruntime:onnxruntime", version.ref = "onnxruntime" }

[plugins]
kotlinMultiplatform = { id = "org.jetbrains.kotlin.multiplatform", version.ref = "kotlin" }
kotlinJvm = { id = "org.jetbrains.kotlin.jvm", version.ref = "kotlin" }
```

These versions are known-good. Newer releases very likely exist; bumping is fine but do it as its own commit so a breakage is easy to bisect.

- [ ] **Step 3: Write the build files**

`settings.gradle.kts`:

```kotlin
pluginManagement {
    repositories {
        gradlePluginPortal()
        mavenCentral()
    }
}

dependencyResolutionManagement {
    repositories {
        mavenCentral()
    }
}

rootProject.name = "presence"

include(":shared")
```

`build.gradle.kts`:

```kotlin
plugins {
    alias(libs.plugins.kotlinMultiplatform) apply false
    alias(libs.plugins.kotlinJvm) apply false
}
```

`gradle.properties`:

```properties
org.gradle.jvmargs=-Xmx3g
org.gradle.caching=true
kotlin.code.style=official
```

`shared/build.gradle.kts`:

```kotlin
plugins {
    alias(libs.plugins.kotlinMultiplatform)
}

kotlin {
    jvmToolchain(17)

    jvm()

    sourceSets {
        commonTest.dependencies {
            implementation(kotlin("test"))
        }
        jvmMain.dependencies {
            implementation(libs.onnxruntime)
        }
    }
}

// Two suites (see Global Constraints). The default suite runs pure-logic tests only; anything
// needing a real model file is tagged "model" and runs under `integrationTest`, which fails
// loudly when the model is absent rather than skipping silently.
//
// The exclusion below MUST skip the `integrationTest` task itself. `withType<Test>()` also
// matches `integrationTest` (it is a Test task too), and JUnit Platform treats a tag that is
// both included and excluded as excluded — which would silently run zero model tests forever,
// the exact behaviour the global test contract forbids.
tasks.withType<Test>().configureEach {
    if (name != "integrationTest") {
        useJUnitPlatform {
            excludeTags("model")
        }
    }
}

tasks.register<Test>("integrationTest") {
    description = "Runs tests that require the ONNX model files in models/."
    group = "verification"
    val jvmTest = tasks.named<Test>("jvmTest").get()
    testClassesDirs = jvmTest.testClassesDirs
    classpath = jvmTest.classpath
    useJUnitPlatform {
        includeTags("model")
    }
}
```

- [ ] **Step 4: Write the failing test**

`shared/src/commonTest/kotlin/app/presence/audio/AudioTypesTest.kt`:

```kotlin
package app.presence.audio

import kotlin.test.Test
import kotlin.test.assertEquals

class AudioTypesTest {

    @Test
    fun `sample rate is 16 kHz`() {
        assertEquals(16_000, SAMPLE_RATE_HZ)
    }

    @Test
    fun `converts milliseconds to samples`() {
        assertEquals(16_000, millisToSamples(1_000))
        assertEquals(512, millisToSamples(32))
        assertEquals(0, millisToSamples(0))
    }

    @Test
    fun `converts samples to milliseconds`() {
        assertEquals(1_000L, samplesToMillis(16_000))
        assertEquals(32L, samplesToMillis(512))
    }

    @Test
    fun `round trips a three second duration`() {
        assertEquals(3_000L, samplesToMillis(millisToSamples(3_000)))
    }
}
```

- [ ] **Step 5: Run the test and verify it fails**

Run: `./gradlew :shared:jvmTest`
Expected: FAIL — unresolved reference `SAMPLE_RATE_HZ`.

- [ ] **Step 6: Write the minimal implementation**

`shared/src/commonMain/kotlin/app/presence/audio/AudioTypes.kt`:

```kotlin
package app.presence.audio

/**
 * Every sample that reaches the engine is mono PCM at this rate, as floats in [-1.0, 1.0].
 * Spec §9.1 requires corpora at 16 kHz mono; the harness rejects anything else rather than
 * silently resampling, because resampling artefacts corrupt the spectral detail that speaker
 * embeddings depend on.
 */
const val SAMPLE_RATE_HZ: Int = 16_000

/** Duration in milliseconds to a sample count at [SAMPLE_RATE_HZ]. */
fun millisToSamples(ms: Long): Int = (ms * SAMPLE_RATE_HZ / 1_000L).toInt()

/** Sample count at [SAMPLE_RATE_HZ] to a duration in milliseconds. */
fun samplesToMillis(sampleCount: Int): Long = sampleCount.toLong() * 1_000L / SAMPLE_RATE_HZ
```

- [ ] **Step 7: Run the test and verify it passes**

Run: `./gradlew :shared:jvmTest`
Expected: PASS, 4 tests.

- [ ] **Step 8: Commit**

```bash
git add settings.gradle.kts build.gradle.kts gradle.properties gradle/ gradlew gradlew.bat shared/
git commit -m "feat: Gradle scaffold and audio duration primitives"
```

---

### Task 2: Framer — stream to fixed-size frames

**Files:**
- Create: `shared/src/commonMain/kotlin/app/presence/audio/Framer.kt`
- Test: `shared/src/commonTest/kotlin/app/presence/audio/FramerTest.kt`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `class Framer(frameSize: Int, hopSize: Int = frameSize)` with `push(samples: FloatArray): List<FloatArray>` and `reset()`. Task 11 and Task 16 feed audio through it.

Why this exists: the VAD model requires exactly-sized frames, but audio arrives in arbitrary chunks. The Framer holds leftovers between calls so frame boundaries stay stable no matter how the input is chunked — a property the tests pin down explicitly, because getting it wrong produces a VAD that works on files and fails on a live microphone.

- [ ] **Step 1: Write the failing test**

`shared/src/commonTest/kotlin/app/presence/audio/FramerTest.kt`:

```kotlin
package app.presence.audio

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class FramerTest {

    private fun ramp(n: Int, from: Int = 0) = FloatArray(n) { (from + it).toFloat() }

    @Test
    fun `emits whole frames and keeps the remainder`() {
        val framer = Framer(frameSize = 4)

        val frames = framer.push(ramp(10))

        assertEquals(2, frames.size)
        assertContentEquals(floatArrayOf(0f, 1f, 2f, 3f), frames[0])
        assertContentEquals(floatArrayOf(4f, 5f, 6f, 7f), frames[1])
    }

    @Test
    fun `remainder joins the next push so boundaries do not shift`() {
        val framer = Framer(frameSize = 4)

        framer.push(ramp(10))
        val frames = framer.push(ramp(4, from = 10))

        assertEquals(1, frames.size)
        assertContentEquals(floatArrayOf(8f, 9f, 10f, 11f), frames[0])
    }

    @Test
    fun `chunking of the input does not change the frames produced`() {
        val whole = Framer(frameSize = 4).push(ramp(16))

        val piecemeal = Framer(frameSize = 4).let { framer ->
            ramp(16).toList().chunked(3).flatMap { framer.push(it.toFloatArray()) }
        }

        assertEquals(whole.size, piecemeal.size)
        whole.forEachIndexed { i, expected -> assertContentEquals(expected, piecemeal[i]) }
    }

    @Test
    fun `overlapping hop advances by the hop size`() {
        val framer = Framer(frameSize = 4, hopSize = 2)

        val frames = framer.push(ramp(8))

        assertEquals(3, frames.size)
        assertContentEquals(floatArrayOf(0f, 1f, 2f, 3f), frames[0])
        assertContentEquals(floatArrayOf(2f, 3f, 4f, 5f), frames[1])
        assertContentEquals(floatArrayOf(4f, 5f, 6f, 7f), frames[2])
    }

    /**
     * The intersection of the two tests above, and the one that actually matters: an
     * implementation retaining a fixed `frameSize - hopSize` samples instead of tracking the real
     * unconsumed offset passes every other test here and still drops samples.
     */
    @Test
    fun `overlapping frames survive being pushed in uneven chunks`() {
        val expected = Framer(frameSize = 4, hopSize = 3).push(ramp(16))

        val framer = Framer(frameSize = 4, hopSize = 3)
        val actual = mutableListOf<FloatArray>()
        var offset = 0
        for (size in listOf(5, 1, 1, 4, 5)) {
            actual += framer.push(ramp(size, from = offset))
            offset += size
        }

        assertEquals(expected.size, actual.size)
        expected.forEachIndexed { index, frame -> assertContentEquals(frame, actual[index]) }
    }

    @Test
    fun `emits nothing until a full frame is available`() {
        val framer = Framer(frameSize = 4)

        assertTrue(framer.push(ramp(3)).isEmpty())
        assertEquals(1, framer.push(ramp(1, from = 3)).size)
    }

    @Test
    fun `reset discards buffered samples`() {
        val framer = Framer(frameSize = 4)

        framer.push(ramp(3))
        framer.reset()

        assertTrue(framer.push(ramp(3)).isEmpty())
    }

    @Test
    fun `rejects a non-positive frame size`() {
        assertFailsWith<IllegalArgumentException> { Framer(frameSize = 0) }
    }

    @Test
    fun `rejects a hop larger than the frame`() {
        assertFailsWith<IllegalArgumentException> { Framer(frameSize = 4, hopSize = 5) }
    }
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `./gradlew :shared:jvmTest --tests "app.presence.audio.FramerTest"`
Expected: FAIL — unresolved reference `Framer`.

- [ ] **Step 3: Write the minimal implementation**

`shared/src/commonMain/kotlin/app/presence/audio/Framer.kt`:

```kotlin
package app.presence.audio

/**
 * Splits a continuous audio stream into fixed-size frames.
 *
 * Leftover samples are held between [push] calls, so the frames produced depend only on the
 * sample stream and never on how that stream happened to be chunked. The VAD model needs
 * exactly-sized frames; audio arrives in whatever sizes the capture layer feels like.
 *
 * @param frameSize samples per emitted frame
 * @param hopSize samples to advance between frames; equal to [frameSize] means no overlap
 */
class Framer(
    private val frameSize: Int,
    private val hopSize: Int = frameSize,
) {
    init {
        require(frameSize > 0) { "frameSize must be positive, was $frameSize" }
        require(hopSize in 1..frameSize) { "hopSize must be in 1..$frameSize, was $hopSize" }
    }

    private var buffer = FloatArray(0)

    /** Appends [samples] and returns every complete frame that is now available. */
    fun push(samples: FloatArray): List<FloatArray> {
        buffer = buffer + samples
        if (buffer.size < frameSize) return emptyList()

        val frames = ArrayList<FloatArray>()
        var offset = 0
        while (offset + frameSize <= buffer.size) {
            frames += buffer.copyOfRange(offset, offset + frameSize)
            offset += hopSize
        }
        buffer = buffer.copyOfRange(offset, buffer.size)
        return frames
    }

    /** Discards buffered samples. Call between independent recordings. */
    fun reset() {
        buffer = FloatArray(0)
    }
}
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `./gradlew :shared:jvmTest --tests "app.presence.audio.FramerTest"`
Expected: PASS, 9 tests.

- [ ] **Step 5: Commit**

```bash
git add shared/src/commonMain/kotlin/app/presence/audio/Framer.kt \
        shared/src/commonTest/kotlin/app/presence/audio/FramerTest.kt
git commit -m "feat: Framer splits audio streams into stable fixed-size frames"
```

---

### Task 3: Embedding — L2-normalized vectors, cosine, centroid

**Files:**
- Create: `shared/src/commonMain/kotlin/app/presence/engine/Embedding.kt`
- Test: `shared/src/commonTest/kotlin/app/presence/engine/EmbeddingTest.kt`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `class Embedding` with `internal values: FloatArray` and public `dimension: Int`, and factory `Embedding.ofNormalized(values: FloatArray): Embedding`; top-level `cosine(a: Embedding, b: Embedding): Float` and `centroidOf(embeddings: List<Embedding>): Embedding`. Tasks 4, 5, 6, 8 and 12 all depend on these exact names.

Design point worth understanding before writing it: `Embedding` can only be constructed normalized. That makes `cosine` a plain dot product and removes an entire class of bug where an unnormalized vector silently produces similarity scores above 1.0 that then poison the accumulator. The constructor is private; the factory is the only door.

- [ ] **Step 1: Write the failing test**

`shared/src/commonTest/kotlin/app/presence/engine/EmbeddingTest.kt`:

```kotlin
package app.presence.engine

import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class EmbeddingTest {

    private fun assertClose(expected: Float, actual: Float, tolerance: Float = 1e-5f) {
        assertTrue(
            abs(expected - actual) <= tolerance,
            "expected $expected but was $actual (tolerance $tolerance)",
        )
    }

    private fun magnitude(embedding: Embedding): Float {
        var sum = 0.0
        for (v in embedding.values) sum += v.toDouble() * v
        return sqrt(sum).toFloat()
    }

    @Test
    fun `normalizes on construction`() {
        val embedding = Embedding.ofNormalized(floatArrayOf(3f, 4f))

        assertClose(1f, magnitude(embedding))
        assertClose(0.6f, embedding.values[0])
        assertClose(0.8f, embedding.values[1])
    }

    @Test
    fun `reports its dimension`() {
        assertClose(192f, Embedding.ofNormalized(FloatArray(192) { 1f }).dimension.toFloat())
    }

    @Test
    fun `rejects an empty vector`() {
        assertFailsWith<IllegalArgumentException> { Embedding.ofNormalized(FloatArray(0)) }
    }

    @Test
    fun `rejects a zero vector`() {
        assertFailsWith<IllegalArgumentException> { Embedding.ofNormalized(floatArrayOf(0f, 0f)) }
    }

    @Test
    fun `cosine of a vector with itself is one`() {
        val embedding = Embedding.ofNormalized(floatArrayOf(1f, 2f, 3f))

        assertClose(1f, cosine(embedding, embedding))
    }

    @Test
    fun `cosine of orthogonal vectors is zero`() {
        val a = Embedding.ofNormalized(floatArrayOf(1f, 0f))
        val b = Embedding.ofNormalized(floatArrayOf(0f, 1f))

        assertClose(0f, cosine(a, b))
    }

    @Test
    fun `cosine of opposed vectors is minus one`() {
        val a = Embedding.ofNormalized(floatArrayOf(1f, 0f))
        val b = Embedding.ofNormalized(floatArrayOf(-1f, 0f))

        assertClose(-1f, cosine(a, b))
    }

    @Test
    fun `cosine ignores magnitude because inputs are normalized`() {
        val small = Embedding.ofNormalized(floatArrayOf(1f, 1f))
        val large = Embedding.ofNormalized(floatArrayOf(100f, 100f))

        assertClose(1f, cosine(small, large))
    }

    @Test
    fun `cosine rejects a dimension mismatch`() {
        val a = Embedding.ofNormalized(floatArrayOf(1f, 0f))
        val b = Embedding.ofNormalized(floatArrayOf(1f, 0f, 0f))

        assertFailsWith<IllegalArgumentException> { cosine(a, b) }
    }

    @Test
    fun `centroid of one embedding is that embedding`() {
        val only = Embedding.ofNormalized(floatArrayOf(1f, 2f, 3f))

        assertClose(1f, cosine(only, centroidOf(listOf(only))))
    }

    @Test
    fun `centroid sits between its inputs and stays normalized`() {
        val a = Embedding.ofNormalized(floatArrayOf(1f, 0f))
        val b = Embedding.ofNormalized(floatArrayOf(0f, 1f))

        val centroid = centroidOf(listOf(a, b))

        assertClose(1f, magnitude(centroid))
        assertClose(cosine(centroid, a), cosine(centroid, b))
        assertTrue(cosine(centroid, a) > 0.7f, "centroid should resemble both inputs")
    }

    @Test
    fun `centroid of opposing vectors is rejected rather than producing NaN`() {
        val forward = Embedding.ofNormalized(floatArrayOf(1f, 0f))
        val backward = Embedding.ofNormalized(floatArrayOf(-1f, 0f))

        assertFailsWith<IllegalArgumentException> { centroidOf(listOf(forward, backward)) }
    }

    @Test
    fun `centroid rejects an empty list`() {
        assertFailsWith<IllegalArgumentException> { centroidOf(emptyList()) }
    }

    @Test
    fun `centroid rejects mixed dimensions`() {
        val a = Embedding.ofNormalized(floatArrayOf(1f, 0f))
        val b = Embedding.ofNormalized(floatArrayOf(1f, 0f, 0f))

        assertFailsWith<IllegalArgumentException> { centroidOf(listOf(a, b)) }
    }
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `./gradlew :shared:jvmTest --tests "app.presence.engine.EmbeddingTest"`
Expected: FAIL — unresolved reference `Embedding`.

- [ ] **Step 3: Write the minimal implementation**

`shared/src/commonMain/kotlin/app/presence/engine/Embedding.kt`:

```kotlin
package app.presence.engine

import kotlin.math.sqrt

/**
 * A speaker embedding, always L2-normalized.
 *
 * The constructor is private and [ofNormalized] is the only way in, so every instance in the
 * system has unit magnitude. That makes [cosine] a plain dot product and removes the class of
 * bug where an unnormalized vector yields similarities above 1.0 that then distort the
 * evidence accumulator downstream.
 */
class Embedding private constructor(
    /**
     * The unit-length components.
     *
     * `internal` rather than public: a `FloatArray` is mutable, and one stray write from outside
     * would silently break the invariant that makes [cosine] a plain dot product. A defensive copy
     * on every read would be safer still, but [cosine] runs once per profile per observation and
     * copying the vector each time is a cost this hot path should not pay.
     */
    internal val values: FloatArray,
) {

    val dimension: Int get() = values.size

    companion object {
        /** Normalizes [values] to unit length. The input array is not modified. */
        fun ofNormalized(values: FloatArray): Embedding {
            require(values.isNotEmpty()) { "an embedding must have at least one dimension" }
            var sumOfSquares = 0.0
            for (v in values) sumOfSquares += v.toDouble() * v
            val magnitude = sqrt(sumOfSquares)
            require(magnitude > 0.0) { "an embedding must have non-zero magnitude" }
            return Embedding(FloatArray(values.size) { (values[it] / magnitude).toFloat() })
        }
    }
}

/**
 * Cosine similarity in [-1, 1]. Both operands are unit vectors by construction, so this is
 * their dot product.
 */
fun cosine(a: Embedding, b: Embedding): Float {
    require(a.dimension == b.dimension) {
        "dimension mismatch: ${a.dimension} vs ${b.dimension}"
    }
    var sum = 0.0
    for (i in a.values.indices) sum += a.values[i].toDouble() * b.values[i]
    return sum.toFloat()
}

/**
 * The mean of [embeddings], re-normalized to unit length.
 *
 * This is how a [app.presence.domain.VoiceProfile] centroid is built during enrollment: no
 * training, just arithmetic over embeddings, which is why enrollment completes in seconds and
 * works identically for live and imported audio (spec §4.1).
 */
fun centroidOf(embeddings: List<Embedding>): Embedding {
    require(embeddings.isNotEmpty()) { "cannot build a centroid from zero embeddings" }
    val dimension = embeddings.first().dimension
    // Accumulate in Double, like ofNormalized and cosine — summing many unit vectors in Float
    // loses precision that a similarity threshold can feel.
    val accumulator = DoubleArray(dimension)
    for (embedding in embeddings) {
        require(embedding.dimension == dimension) {
            "dimension mismatch in centroid input: expected $dimension, got ${embedding.dimension}"
        }
        for (i in 0 until dimension) accumulator[i] += embedding.values[i].toDouble()
    }
    // Antipodal inputs sum to ~zero; ofNormalized rejects that rather than returning NaN.
    return Embedding.ofNormalized(FloatArray(dimension) { accumulator[it].toFloat() })
}
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `./gradlew :shared:jvmTest --tests "app.presence.engine.EmbeddingTest"`
Expected: PASS, 22 tests.

- [ ] **Step 5: Commit**

```bash
git add shared/src/commonMain/kotlin/app/presence/engine/Embedding.kt \
        shared/src/commonTest/kotlin/app/presence/engine/EmbeddingTest.kt
git commit -m "feat: normalized speaker embeddings with cosine and centroid"
```

---

### Task 4: VoiceProfile and Language — per-language sub-centroids

**Files:**
- Create: `shared/src/commonMain/kotlin/app/presence/domain/Language.kt`
- Create: `shared/src/commonMain/kotlin/app/presence/domain/VoiceProfile.kt`
- Test: `shared/src/commonTest/kotlin/app/presence/domain/VoiceProfileTest.kt`

**Interfaces:**
- Consumes: `Embedding` (Task 3).
- Produces: `enum class Language { ENGLISH, HEBREW }`; `data class VoiceProfile(personId: String, role: VoiceProfile.Role, centroids: Map<Language, Embedding>, negatives: List<Embedding> = emptyList(), enrolledSecondsByLanguage: Map<Language, Double> = emptyMap())` with `enum class Role { TARGET_CHILD, SIBLING, PARENT, OTHER }` and `languages: Set<Language>`. Tasks 5, 6, 8, 12 and 15 use these.

This is where spec §4.2 becomes code. The centroid map is keyed by language — there is deliberately no way to express a pooled cross-language centroid, because a 40–49% domain shift blurs a pooled centroid rather than broadening it.

- [ ] **Step 1: Write the failing test**

`shared/src/commonTest/kotlin/app/presence/domain/VoiceProfileTest.kt`:

```kotlin
package app.presence.domain

import app.presence.engine.Embedding
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class VoiceProfileTest {

    private fun embedding(vararg values: Float) = Embedding.ofNormalized(floatArrayOf(*values))

    @Test
    fun `exposes the languages it was enrolled in`() {
        val profile = VoiceProfile(
            personId = "maya",
            role = VoiceProfile.Role.TARGET_CHILD,
            centroids = mapOf(
                Language.ENGLISH to embedding(1f, 0f),
                Language.HEBREW to embedding(0f, 1f),
            ),
        )

        assertEquals(setOf(Language.ENGLISH, Language.HEBREW), profile.languages)
    }

    @Test
    fun `allows a single-language profile`() {
        val profile = VoiceProfile(
            personId = "guest",
            role = VoiceProfile.Role.OTHER,
            centroids = mapOf(Language.ENGLISH to embedding(1f, 0f)),
        )

        assertEquals(setOf(Language.ENGLISH), profile.languages)
    }

    @Test
    fun `rejects a profile with no centroids`() {
        val failure = assertFailsWith<IllegalArgumentException> {
            VoiceProfile(
                personId = "nobody",
                role = VoiceProfile.Role.OTHER,
                centroids = emptyMap(),
            )
        }
        assertTrue(failure.message!!.contains("nobody"))
    }

    @Test
    fun `rejects a blank person id`() {
        assertFailsWith<IllegalArgumentException> {
            VoiceProfile(
                personId = "  ",
                role = VoiceProfile.Role.OTHER,
                centroids = mapOf(Language.ENGLISH to embedding(1f, 0f)),
            )
        }
    }

    @Test
    fun `rejects centroids of differing dimensions`() {
        assertFailsWith<IllegalArgumentException> {
            VoiceProfile(
                personId = "maya",
                role = VoiceProfile.Role.TARGET_CHILD,
                centroids = mapOf(
                    Language.ENGLISH to embedding(1f, 0f),
                    Language.HEBREW to embedding(0f, 1f, 0f),
                ),
            )
        }
    }

    @Test
    fun `rejects a negative example of a different dimension`() {
        val failure = assertFailsWith<IllegalArgumentException> {
            VoiceProfile(
                personId = "maya",
                role = VoiceProfile.Role.TARGET_CHILD,
                centroids = mapOf(Language.ENGLISH to embedding(1f, 0f)),
                negatives = listOf(embedding(0f, 1f, 0f)),
            )
        }
        assertTrue(failure.message!!.contains("maya"))
    }

    @Test
    fun `defaults to no negatives and no recorded durations`() {
        val profile = VoiceProfile(
            personId = "maya",
            role = VoiceProfile.Role.TARGET_CHILD,
            centroids = mapOf(Language.ENGLISH to embedding(1f, 0f)),
        )

        assertTrue(profile.negatives.isEmpty())
        assertTrue(profile.enrolledSecondsByLanguage.isEmpty())
    }
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `./gradlew :shared:jvmTest --tests "app.presence.domain.VoiceProfileTest"`
Expected: FAIL — unresolved reference `VoiceProfile`.

- [ ] **Step 3: Write the minimal implementation**

`shared/src/commonMain/kotlin/app/presence/domain/Language.kt`:

```kotlin
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
```

`shared/src/commonMain/kotlin/app/presence/domain/VoiceProfile.kt`:

```kotlin
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
        // Negatives are compared against these centroids downstream, so a mismatched one must
        // fail here rather than inside cosine(), far from where the bad data entered.
        val dimensions = (centroids.values + negatives).map { it.dimension }.distinct()
        require(dimensions.size == 1) {
            "profile '$personId' mixes embedding dimensions: $dimensions"
        }
    }

    val languages: Set<Language> get() = centroids.keys
}
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `./gradlew :shared:jvmTest --tests "app.presence.domain.VoiceProfileTest"`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add shared/src/commonMain/kotlin/app/presence/domain/ \
        shared/src/commonTest/kotlin/app/presence/domain/
git commit -m "feat: VoiceProfile with per-language sub-centroids"
```

---

### Task 5: VadGate — hysteresis segmentation

**Files:**
- Create: `shared/src/commonMain/kotlin/app/presence/engine/VoiceActivityDetector.kt`
- Create: `shared/src/commonMain/kotlin/app/presence/engine/VadGate.kt`
- Test: `shared/src/commonTest/kotlin/app/presence/engine/VadGateTest.kt`

**Interfaces:**
- Consumes: `SAMPLE_RATE_HZ`, `samplesToMillis` (Task 1).
- Produces: `interface VoiceActivityDetector { val frameSize: Int; fun speechProbability(frame: FloatArray): Float; fun reset() }`; `class SpeechSegment(val samples: FloatArray, val startSampleIndex: Long)` with `durationMs: Long`; `class VadGateConfig(startThreshold, endThreshold, hangoverFrames, minSegmentMs, maxSegmentMs)`; `class VadGate(vad, config)` with `accept(frame: FloatArray): SpeechSegment?`, `flush(): SpeechSegment?`, `reset()`. Tasks 9, 12 and 16 use these.

This gate is the reason battery cost is survivable: the expensive embedding model only wakes for actual speech, roughly a 5% duty cycle in a normal room (spec §4). Two thresholds rather than one — speech *starts* at a higher probability than it *ends* — so a brief dip mid-word does not chop an utterance in half.

The tests drive a scripted fake VAD rather than a real model. That is deliberate: segmentation logic and model quality are separate concerns, and only one of them can be tested deterministically.

- [ ] **Step 1: Write the failing test**

`shared/src/commonTest/kotlin/app/presence/engine/VadGateTest.kt`:

```kotlin
package app.presence.engine

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** Replays a fixed list of probabilities, one per frame. */
private class ScriptedVad(
    private val probabilities: List<Float>,
    override val frameSize: Int = 512,
) : VoiceActivityDetector {
    private var index = 0
    override fun speechProbability(frame: FloatArray): Float =
        probabilities.getOrElse(index++) { 0f }
    override fun reset() { index = 0 }
}

class VadGateTest {

    /** 512 samples at 16 kHz is 32 ms, so ~31 frames per second. */
    private val frame = FloatArray(512) { 0.1f }

    private fun gate(
        script: List<Float>,
        config: VadGateConfig = VadGateConfig(),
    ): Pair<VadGate, List<Float>> = VadGate(ScriptedVad(script), config) to script

    private fun runAll(script: List<Float>, config: VadGateConfig = VadGateConfig()): List<SpeechSegment> {
        val (vadGate, _) = gate(script, config)
        val segments = mutableListOf<SpeechSegment>()
        repeat(script.size) { vadGate.accept(frame)?.let { segments += it } }
        vadGate.flush()?.let { segments += it }
        return segments
    }

    @Test
    fun `emits nothing for pure silence`() {
        assertTrue(runAll(List(50) { 0.05f }).isEmpty())
    }

    @Test
    fun `emits one segment for one continuous utterance`() {
        // 40 speech frames (~1.28 s), then enough silence to close the segment.
        val script = List(40) { 0.9f } + List(20) { 0.02f }

        val segments = runAll(script)

        assertEquals(1, segments.size)
        assertTrue(segments[0].durationMs >= 1_000, "expected ~1.28 s, got ${segments[0].durationMs} ms")
    }

    @Test
    fun `does not split on a brief dip below the start threshold`() {
        // Dips to 0.4 — under the 0.5 start threshold but over the 0.35 end threshold.
        val script = List(20) { 0.9f } + List(3) { 0.4f } + List(20) { 0.9f } + List(20) { 0.02f }

        val segments = runAll(script)

        assertEquals(1, segments.size, "a brief dip must not chop the utterance in two")
    }

    /**
     * The test that makes the two thresholds load-bearing. The dip above is shorter than the
     * hangover, so an implementation with endThreshold == startThreshold would pass it anyway.
     * This dip is longer than the hangover, so only genuine hysteresis — 0.4 counting as
     * continuation because it clears the 0.35 end threshold though not the 0.5 start threshold —
     * keeps this as one utterance.
     */
    @Test
    fun `a sustained dip between the two thresholds does not split the utterance`() {
        val script = List(20) { 0.9f } + List(10) { 0.4f } + List(20) { 0.9f } + List(20) { 0.01f }

        val segments = runAll(script)

        assertEquals(1, segments.size, "hysteresis should hold this together as one utterance")
    }

    @Test
    fun `splits two utterances separated by real silence`() {
        val script = List(20) { 0.9f } + List(20) { 0.01f } + List(20) { 0.9f } + List(20) { 0.01f }

        val segments = runAll(script)

        assertEquals(2, segments.size)
    }

    @Test
    fun `discards a segment shorter than the minimum`() {
        // 5 frames is 160 ms, well under the 400 ms floor.
        val script = List(5) { 0.9f } + List(20) { 0.01f }

        assertTrue(runAll(script).isEmpty())
    }

    @Test
    fun `caps a very long utterance at the maximum segment length`() {
        val config = VadGateConfig(maxSegmentMs = 1_000)
        // 100 frames is ~3.2 s of unbroken speech.
        val script = List(100) { 0.9f } + List(20) { 0.01f }

        val segments = runAll(script, config)

        assertTrue(segments.size >= 3, "expected the 3.2 s utterance to be cut into ~1 s pieces")
        segments.forEach {
            assertTrue(it.durationMs <= 1_100, "segment of ${it.durationMs} ms exceeded the cap")
        }
    }

    @Test
    fun `flush emits an utterance still open at end of stream`() {
        val vadGate = VadGate(ScriptedVad(List(40) { 0.9f }))
        repeat(40) { vadGate.accept(frame) }

        val flushed = vadGate.flush()

        assertNotNull(flushed)
        assertTrue(flushed.durationMs >= 1_000)
    }

    @Test
    fun `flush returns null when no utterance is open`() {
        val vadGate = VadGate(ScriptedVad(List(10) { 0.01f }))
        repeat(10) { vadGate.accept(frame) }

        assertNull(vadGate.flush())
    }

    @Test
    fun `reports where in the stream each segment started`() {
        val script = List(20) { 0.01f } + List(40) { 0.9f } + List(20) { 0.01f }

        val segments = runAll(script)

        assertEquals(1, segments.size)
        // Speech begins at frame 20 => 20 * 512 = 10240 samples => 640 ms.
        assertEquals(10_240L, segments[0].startSampleIndex)
    }

    @Test
    fun `reset clears an in-progress utterance`() {
        val vadGate = VadGate(ScriptedVad(List(60) { 0.9f }))
        repeat(30) { vadGate.accept(frame) }

        vadGate.reset()

        assertNull(vadGate.flush())
    }
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `./gradlew :shared:jvmTest --tests "app.presence.engine.VadGateTest"`
Expected: FAIL — unresolved reference `VoiceActivityDetector`.

- [ ] **Step 3: Write the VAD interface**

`shared/src/commonMain/kotlin/app/presence/engine/VoiceActivityDetector.kt`:

```kotlin
package app.presence.engine

/**
 * Frame-level speech probability.
 *
 * Implementations are stateful — Silero carries recurrent state between frames — so a single
 * instance must not be shared across concurrent streams, and [reset] must be called between
 * independent recordings.
 */
interface VoiceActivityDetector {

    /** Exact number of samples each frame passed to [speechProbability] must contain. */
    val frameSize: Int

    /** Returns P(speech) in [0, 1] for one frame of exactly [frameSize] samples. */
    fun speechProbability(frame: FloatArray): Float

    /** Clears recurrent state. Call between independent recordings. */
    fun reset()
}
```

- [ ] **Step 4: Write the gate**

`shared/src/commonMain/kotlin/app/presence/engine/VadGate.kt`:

```kotlin
package app.presence.engine

import app.presence.audio.samplesToMillis

/** A contiguous run of speech, with where it began in the stream. */
class SpeechSegment(
    val samples: FloatArray,
    val startSampleIndex: Long,
) {
    val durationMs: Long get() = samplesToMillis(samples.size)
}

/**
 * @param startThreshold probability at which a new utterance begins
 * @param endThreshold probability below which frames count toward closing it; deliberately
 *   lower than [startThreshold] so a brief dip mid-word does not chop an utterance in half
 * @param hangoverFrames consecutive sub-[endThreshold] frames required to close an utterance
 * @param minSegmentMs segments shorter than this are discarded as noise
 * @param maxSegmentMs an unbroken utterance is cut at this length so evidence flows steadily
 *   into the accumulator rather than arriving in one late lump
 */
class VadGateConfig(
    val startThreshold: Float = 0.5f,
    val endThreshold: Float = 0.35f,
    val hangoverFrames: Int = 8,
    val minSegmentMs: Long = 400,
    val maxSegmentMs: Long = 3_000,
) {
    init {
        require(startThreshold in 0f..1f) { "startThreshold must be in 0..1" }
        require(endThreshold in 0f..startThreshold) { "endThreshold must be in 0..startThreshold" }
        require(hangoverFrames > 0) { "hangoverFrames must be positive" }
        require(minSegmentMs > 0) { "minSegmentMs must be positive" }
        require(maxSegmentMs >= minSegmentMs) { "maxSegmentMs must be >= minSegmentMs" }
    }
}

/**
 * Turns a frame-by-frame speech probability into discrete speech segments.
 *
 * This is the gate that makes battery cost survivable: the expensive embedding model only wakes
 * for actual speech, roughly a 5% duty cycle in a normal room (spec §4).
 */
class VadGate(
    private val vad: VoiceActivityDetector,
    private val config: VadGateConfig = VadGateConfig(),
) {
    private val current = ArrayList<FloatArray>()
    private var currentSampleCount = 0
    private var inSpeech = false
    private var silentFrames = 0
    private var consumedSamples = 0L
    private var segmentStart = 0L

    /** Feeds one frame. Returns a segment if one completed on this frame. */
    fun accept(frame: FloatArray): SpeechSegment? {
        val probability = vad.speechProbability(frame)
        var completed: SpeechSegment? = null

        if (!inSpeech) {
            if (probability >= config.startThreshold) {
                inSpeech = true
                silentFrames = 0
                segmentStart = consumedSamples
                current.clear()
                currentSampleCount = 0
                append(frame)
            }
        } else {
            append(frame)
            silentFrames = if (probability < config.endThreshold) silentFrames + 1 else 0
            val reachedCap = samplesToMillis(currentSampleCount) >= config.maxSegmentMs
            if (silentFrames >= config.hangoverFrames || reachedCap) {
                completed = close()
            }
        }

        consumedSamples += frame.size
        return completed
    }

    /** Emits an utterance still open at end of stream, if it is long enough. */
    fun flush(): SpeechSegment? = if (inSpeech) close() else null

    /** Discards any in-progress utterance and the stream position. */
    fun reset() {
        current.clear()
        currentSampleCount = 0
        inSpeech = false
        silentFrames = 0
        consumedSamples = 0L
        segmentStart = 0L
        vad.reset()
    }

    private fun append(frame: FloatArray) {
        current += frame
        currentSampleCount += frame.size
    }

    private fun close(): SpeechSegment? {
        // The hangover frames that closed this segment are trailing silence. They must not pad the
        // segment past the minimum-duration floor, and they must not reach the embedder. Trim only
        // the tail: filtering sub-threshold frames as they arrive would splice silence out of the
        // middle of an utterance too, leaving the samples non-contiguous so that
        // startSampleIndex + size no longer marks the real end of speech — which is exactly the
        // arithmetic DetectionEngine uses to timestamp observations.
        val kept = current.take(current.size - silentFrames)
        val merged = FloatArray(kept.sumOf { it.size })
        var offset = 0
        for (chunk in kept) {
            chunk.copyInto(merged, offset)
            offset += chunk.size
        }
        val start = segmentStart
        current.clear()
        currentSampleCount = 0
        inSpeech = false
        silentFrames = 0
        return if (samplesToMillis(merged.size) >= config.minSegmentMs) {
            SpeechSegment(merged, start)
        } else {
            null
        }
    }
}
```

- [ ] **Step 5: Run the test and verify it passes**

Run: `./gradlew :shared:jvmTest --tests "app.presence.engine.VadGateTest"`
Expected: PASS, 12 tests.

- [ ] **Step 6: Commit**

```bash
git add shared/src/commonMain/kotlin/app/presence/engine/VoiceActivityDetector.kt \
        shared/src/commonMain/kotlin/app/presence/engine/VadGate.kt \
        shared/src/commonTest/kotlin/app/presence/engine/VadGateTest.kt
git commit -m "feat: VadGate segments speech with start/end hysteresis"
```

---

### Task 6: EnrollmentService — clips to profile, with per-language shortfall

**Files:**
- Create: `shared/src/commonMain/kotlin/app/presence/enrollment/EnrollmentService.kt`
- Test: `shared/src/commonTest/kotlin/app/presence/enrollment/EnrollmentServiceTest.kt`

**Interfaces:**
- Consumes: `Embedding`, `centroidOf` (Task 3); `Language`, `VoiceProfile` (Task 4).
- Produces: `data class EnrollmentClip(language: Language, embedding: Embedding, speechSeconds: Double)`; `data class EnrollmentPolicy(minSecondsPerLanguage: Double = 45.0, requiredLanguages: Set<Language> = setOf(ENGLISH, HEBREW))`; `sealed interface EnrollmentResult` with `Complete(profile)` and `Incomplete(profile, shortfallSeconds)`; `class EnrollmentService(policy)` with `enroll(personId, role, clips): EnrollmentResult`. Tasks 16 and 17 use this to build profiles from corpus audio.

Two spec rules become executable here. Progress is measured in **seconds of usable speech**, not clip count, because published results show utterances under about a second are near-worthless (spec §5). And enrollment is **not complete until every required language is covered** (spec §4.2) — the returned shortfall map is exactly what the enrollment UI needs to tell a parent which language is still short.

- [ ] **Step 1: Write the failing test**

`shared/src/commonTest/kotlin/app/presence/enrollment/EnrollmentServiceTest.kt`:

```kotlin
package app.presence.enrollment

import app.presence.domain.Language
import app.presence.domain.VoiceProfile
import app.presence.engine.Embedding
import app.presence.engine.cosine
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class EnrollmentServiceTest {

    private val service = EnrollmentService()

    private fun clip(language: Language, seconds: Double, vararg values: Float) =
        EnrollmentClip(language, Embedding.ofNormalized(floatArrayOf(*values)), seconds)

    private fun enough(language: Language, vararg values: Float) =
        clip(language, 50.0, *values)

    @Test
    fun `completes when both languages meet the minimum`() {
        val result = service.enroll(
            personId = "maya",
            role = VoiceProfile.Role.TARGET_CHILD,
            clips = listOf(
                enough(Language.ENGLISH, 1f, 0f),
                enough(Language.HEBREW, 0f, 1f),
            ),
        )

        val complete = assertIs<EnrollmentResult.Complete>(result)
        assertEquals(setOf(Language.ENGLISH, Language.HEBREW), complete.profile.languages)
    }

    @Test
    fun `is incomplete when Hebrew is missing entirely`() {
        val result = service.enroll(
            personId = "maya",
            role = VoiceProfile.Role.TARGET_CHILD,
            clips = listOf(enough(Language.ENGLISH, 1f, 0f)),
        )

        val incomplete = assertIs<EnrollmentResult.Incomplete>(result)
        assertEquals(45.0, incomplete.shortfallSeconds[Language.HEBREW])
        assertNull(incomplete.shortfallSeconds[Language.ENGLISH])
    }

    @Test
    fun `reports the exact remaining seconds per language`() {
        val result = service.enroll(
            personId = "maya",
            role = VoiceProfile.Role.TARGET_CHILD,
            clips = listOf(
                clip(Language.ENGLISH, 30.0, 1f, 0f),
                clip(Language.HEBREW, 5.0, 0f, 1f),
            ),
        )

        val incomplete = assertIs<EnrollmentResult.Incomplete>(result)
        assertEquals(15.0, incomplete.shortfallSeconds[Language.ENGLISH])
        assertEquals(40.0, incomplete.shortfallSeconds[Language.HEBREW])
    }

    @Test
    fun `still returns a usable partial profile when incomplete`() {
        val result = service.enroll(
            personId = "maya",
            role = VoiceProfile.Role.TARGET_CHILD,
            clips = listOf(clip(Language.ENGLISH, 10.0, 1f, 0f)),
        )

        val incomplete = assertIs<EnrollmentResult.Incomplete>(result)
        val profile = assertNotNull(incomplete.profile)
        assertEquals(setOf(Language.ENGLISH), profile.languages)
    }

    @Test
    fun `returns no profile at all when there are no clips`() {
        val result = service.enroll(
            personId = "maya",
            role = VoiceProfile.Role.TARGET_CHILD,
            clips = emptyList(),
        )

        val incomplete = assertIs<EnrollmentResult.Incomplete>(result)
        assertNull(incomplete.profile)
    }

    @Test
    fun `averages clips within a language into one centroid`() {
        val result = service.enroll(
            personId = "maya",
            role = VoiceProfile.Role.TARGET_CHILD,
            clips = listOf(
                clip(Language.ENGLISH, 25.0, 1f, 0f),
                clip(Language.ENGLISH, 25.0, 0f, 1f),
                enough(Language.HEBREW, 0f, 1f),
            ),
        )

        val complete = assertIs<EnrollmentResult.Complete>(result)
        val english = complete.profile.centroids[Language.ENGLISH]!!
        val first = Embedding.ofNormalized(floatArrayOf(1f, 0f))
        val second = Embedding.ofNormalized(floatArrayOf(0f, 1f))
        assertTrue(cosine(english, first) > 0.7f)
        assertTrue(cosine(english, second) > 0.7f)
    }

    @Test
    fun `records enrolled seconds per language on the profile`() {
        val result = service.enroll(
            personId = "maya",
            role = VoiceProfile.Role.TARGET_CHILD,
            clips = listOf(
                clip(Language.ENGLISH, 25.0, 1f, 0f),
                clip(Language.ENGLISH, 30.0, 1f, 0f),
                enough(Language.HEBREW, 0f, 1f),
            ),
        )

        val complete = assertIs<EnrollmentResult.Complete>(result)
        assertEquals(55.0, complete.profile.enrolledSecondsByLanguage[Language.ENGLISH])
        assertEquals(50.0, complete.profile.enrolledSecondsByLanguage[Language.HEBREW])
    }

    @Test
    fun `a single-language policy suits a monolingual household`() {
        val monolingual = EnrollmentService(
            EnrollmentPolicy(requiredLanguages = setOf(Language.HEBREW)),
        )

        val result = monolingual.enroll(
            personId = "savta",
            role = VoiceProfile.Role.OTHER,
            clips = listOf(enough(Language.HEBREW, 0f, 1f)),
        )

        assertIs<EnrollmentResult.Complete>(result)
    }

    @Test
    fun `rejects a policy with no required languages`() {
        kotlin.test.assertFailsWith<IllegalArgumentException> {
            EnrollmentPolicy(requiredLanguages = emptySet())
        }
    }
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `./gradlew :shared:jvmTest --tests "app.presence.enrollment.EnrollmentServiceTest"`
Expected: FAIL — unresolved reference `EnrollmentService`.

- [ ] **Step 3: Write the minimal implementation**

`shared/src/commonMain/kotlin/app/presence/enrollment/EnrollmentService.kt`:

```kotlin
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
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `./gradlew :shared:jvmTest --tests "app.presence.enrollment.EnrollmentServiceTest"`
Expected: PASS, 9 tests.

- [ ] **Step 5: Commit**

```bash
git add shared/src/commonMain/kotlin/app/presence/enrollment/ \
        shared/src/commonTest/kotlin/app/presence/enrollment/
git commit -m "feat: EnrollmentService enforces bilingual coverage in seconds of speech"
```

---

### Task 7: HouseholdMatcher — L1 cohort scoring

**Files:**
- Create: `shared/src/commonMain/kotlin/app/presence/engine/HouseholdMatcher.kt`
- Test: `shared/src/commonTest/kotlin/app/presence/engine/HouseholdMatcherTest.kt`

**Interfaces:**
- Consumes: `Embedding`, `cosine` (Task 3); `VoiceProfile` (Task 4).
- Produces: `data class MatchScore(targetSimilarity: Float, bestNonTargetSimilarity: Float, cohortZ: Float, logLikelihoodRatio: Float)`; `data class MatcherConfig(softmaxTemperature, minCohortZ, fallbackCohortMean, fallbackCohortStdDev, maxAbsLlr)`; `class HouseholdMatcher(target: VoiceProfile, cohort: List<VoiceProfile>, config: MatcherConfig)` with `score(observation: Embedding): MatchScore`. Tasks 9 and 16 use it.

This is rung L1 of the accuracy ladder, and the reason it exists is worth holding in mind while implementing. Published EERs measure *open-set* verification against an absolute threshold. We instead have a known household roster, which turns the question into "which of these people does this most resemble" — a relative comparison that attacks sibling confusion directly (spec §4.1).

Three mechanisms combine:

1. **Softmax over the roster** gives a closed-set posterior, converted to a log-likelihood ratio.
2. **A cohort z-score** handles the open-set case the softmax cannot: a stranger or the television resembles nobody in particular, and a posterior alone would still crown a winner. Below the z floor, no positive evidence is allowed through.
3. **Negative examples** from the "not them" button veto a match, so a drifting profile decays toward silence rather than toward firing on a sibling.

The constructor requires a non-empty cohort. That is spec §5's household-enrollment requirement made unskippable in code.

- [ ] **Step 1: Write the failing test**

`shared/src/commonTest/kotlin/app/presence/engine/HouseholdMatcherTest.kt`:

```kotlin
package app.presence.engine

import app.presence.domain.Language
import app.presence.domain.VoiceProfile
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class HouseholdMatcherTest {

    private fun profile(
        id: String,
        role: VoiceProfile.Role,
        vararg centroids: Pair<Language, FloatArray>,
    ) = VoiceProfile(
        personId = id,
        role = role,
        centroids = centroids.associate { (language, values) ->
            language to Embedding.ofNormalized(values)
        },
    )

    private fun observation(vararg values: Float) = Embedding.ofNormalized(floatArrayOf(*values))

    private val maya = profile(
        "maya", VoiceProfile.Role.TARGET_CHILD,
        Language.ENGLISH to floatArrayOf(1f, 0f, 0f),
    )
    private val parent = profile(
        "parent", VoiceProfile.Role.PARENT,
        Language.ENGLISH to floatArrayOf(0f, 1f, 0f),
    )
    private val sibling = profile(
        "sibling", VoiceProfile.Role.SIBLING,
        Language.ENGLISH to floatArrayOf(0f, 0f, 1f),
    )

    private fun matcher(
        target: VoiceProfile = maya,
        cohort: List<VoiceProfile> = listOf(parent, sibling),
        config: MatcherConfig = MatcherConfig(),
    ) = HouseholdMatcher(target, cohort, config)

    @Test
    fun `the target's own voice produces strong positive evidence`() {
        val score = matcher().score(observation(1f, 0f, 0f))

        assertTrue(score.logLikelihoodRatio > 3f, "expected strong evidence, got ${score.logLikelihoodRatio}")
        assertTrue(abs(1f - score.targetSimilarity) < 1e-5f)
    }

    @Test
    fun `a cohort member's voice produces negative evidence`() {
        val score = matcher().score(observation(0f, 1f, 0f))

        assertTrue(score.logLikelihoodRatio < 0f, "expected negative evidence, got ${score.logLikelihoodRatio}")
    }

    @Test
    fun `a sibling is not mistaken for the target`() {
        val score = matcher().score(observation(0f, 0f, 1f))

        assertTrue(score.logLikelihoodRatio < 0f)
        assertTrue(score.bestNonTargetSimilarity > score.targetSimilarity)
    }

    @Test
    fun `a stranger resembling everyone equally produces no positive evidence`() {
        // Equidistant from all three profiles: the softmax alone would still pick a winner,
        // so this is the case the z gate exists for.
        val score = matcher().score(observation(0.577f, 0.577f, 0.577f))

        assertTrue(score.logLikelihoodRatio <= 0f, "got ${score.logLikelihoodRatio}")
    }

    @Test
    fun `the z gate suppresses a weak win over a small cohort`() {
        // One cohort member means no usable cohort statistics, so the fallback mean and
        // standard deviation apply: z = (0.18 - 0.15) / 0.10 = 0.30, below the 0.5 floor.
        val single = matcher(cohort = listOf(parent))

        val score = single.score(observation(0.18f, 0.05f, 0.9823f))

        assertTrue(score.targetSimilarity > score.bestNonTargetSimilarity, "target should nominally win")
        assertTrue(score.cohortZ < 0.5f, "z was ${score.cohortZ}")
        assertEquals(0f, score.logLikelihoodRatio, "a weak win must not become positive evidence")
    }

    @Test
    fun `a negative example vetoes a match`() {
        val withNegative = maya.copy(negatives = listOf(observation(1f, 0f, 0f)))

        val score = matcher(target = withNegative).score(observation(1f, 0f, 0f))

        assertEquals(0f, score.logLikelihoodRatio)
    }

    /**
     * The test the z gate cannot satisfy. A winning target always sits far above the cohort mean
     * in standard-deviation terms, so only the absolute floor can reject a voice that beats the
     * household without resembling anyone much. Needs a fourth dimension so the observation can
     * hold energy matching nobody.
     */
    @Test
    fun `a weak absolute match produces no positive evidence even when it beats the cohort`() {
        val target = profile("maya", VoiceProfile.Role.TARGET_CHILD, Language.ENGLISH to floatArrayOf(1f, 0f, 0f, 0f))
        val parent = profile("parent", VoiceProfile.Role.PARENT, Language.ENGLISH to floatArrayOf(0f, 1f, 0f, 0f))
        val sibling = profile("sibling", VoiceProfile.Role.SIBLING, Language.ENGLISH to floatArrayOf(0f, 0f, 1f, 0f))
        // Similarity 0.30 to the target, 0.05 to each cohort member, the rest matching nobody.
        val stranger = observation(0.30f, 0.05f, 0.05f, 0.9513f)

        val score = HouseholdMatcher(target, listOf(parent, sibling)).score(stranger)

        assertTrue(score.targetSimilarity > score.bestNonTargetSimilarity, "target should nominally win")
        assertTrue(score.cohortZ >= 0.5f, "the z gate does not bind here — that is the point")
        assertEquals(0f, score.logLikelihoodRatio, "the absolute floor must suppress this")
    }

    @Test
    fun `a negative example close to the target also vetoes`() {
        val nearMiss = maya.copy(negatives = listOf(observation(0.97f, 0.2431f, 0f)))

        val score = matcher(target = nearMiss).score(observation(1f, 0f, 0f))

        assertEquals(0f, score.logLikelihoodRatio)
    }

    @Test
    fun `a negative example unlike the observation does not veto`() {
        val withNegative = maya.copy(negatives = listOf(observation(0f, 0f, 1f)))

        val score = matcher(target = withNegative).score(observation(1f, 0f, 0f))

        assertTrue(score.logLikelihoodRatio > 3f)
    }

    @Test
    fun `the best-matching language sub-centroid is used`() {
        val bilingual = profile(
            "maya", VoiceProfile.Role.TARGET_CHILD,
            Language.ENGLISH to floatArrayOf(1f, 0f, 0f),
            Language.HEBREW to floatArrayOf(0f, 1f, 0f),
        )
        // The observation matches the Hebrew centroid; scoring must not average the two.
        val score = HouseholdMatcher(bilingual, listOf(sibling)).score(observation(0f, 1f, 0f))

        assertTrue(abs(1f - score.targetSimilarity) < 1e-5f, "got ${score.targetSimilarity}")
    }

    @Test
    fun `evidence is clamped to the configured maximum`() {
        val score = matcher(config = MatcherConfig(maxAbsLlr = 2f)).score(observation(1f, 0f, 0f))

        assertTrue(score.logLikelihoodRatio <= 2f)
    }

    @Test
    fun `reports the best non-target similarity`() {
        val score = matcher().score(observation(0f, 0.6f, 0.8f))

        assertTrue(abs(0.8f - score.bestNonTargetSimilarity) < 1e-5f, "got ${score.bestNonTargetSimilarity}")
    }

    @Test
    fun `an empty cohort is rejected because household enrollment is required`() {
        val failure = assertFailsWith<IllegalArgumentException> {
            HouseholdMatcher(maya, emptyList())
        }
        assertTrue(failure.message!!.contains("cohort"))
    }
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `./gradlew :shared:jvmTest --tests "app.presence.engine.HouseholdMatcherTest"`
Expected: FAIL — unresolved reference `HouseholdMatcher`.

- [ ] **Step 3: Write the minimal implementation**

`shared/src/commonMain/kotlin/app/presence/engine/HouseholdMatcher.kt`:

```kotlin
package app.presence.engine

import app.presence.domain.VoiceProfile
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * @param cohortZ how far the target's similarity stands above the rest of the household, in
 *   standard deviations. This is what distinguishes "it is the target" from "it is a stranger
 *   who happens to resemble the target slightly more than the others".
 * @param logLikelihoodRatio the evidence this observation contributes. Positive favours the
 *   target; the accumulator sums these across segments.
 */
data class MatchScore(
    val targetSimilarity: Float,
    val bestNonTargetSimilarity: Float,
    val cohortZ: Float,
    val logLikelihoodRatio: Float,
)

/**
 * @param softmaxTemperature lower values sharpen the closed-set posterior. Tuned in Phase 0.
 * @param minCohortZ evidence floor for the open-set case; below this, no positive evidence passes
 * @param fallbackCohortMean used when the cohort is too small for real statistics
 * @param fallbackCohortStdDev likewise
 * @param maxAbsLlr keeps one confident observation from dominating an entire window
 */
data class MatcherConfig(
    val softmaxTemperature: Float = 0.1f,
    val minCohortZ: Float = 0.5f,
    /**
     * Absolute similarity the target must reach before any positive evidence is allowed.
     *
     * The z-score alone cannot do this job. For two cohort similarities, `max − mean` equals the
     * population standard deviation exactly, and a winning target must exceed `max(cohort)`, so
     * `z > 1` always — the z floor never binds for a normal-sized household. Without an absolute
     * floor, a stranger who resembles the target more than the family does passes unchallenged.
     * Calibrate against real embeddings in Phase 0: this is the single most important open-set knob.
     */
    val minTargetSimilarity: Float = 0.35f,
    /** Floor on the cohort standard deviation, in cosine units — keeps z finite and meaningful. */
    val minCohortStdDev: Float = 0.02f,
    /** How close a negative example may come to the target before it vetoes the match. */
    val negativeVetoMargin: Float = 0.05f,
    val fallbackCohortMean: Float = 0.15f,
    val fallbackCohortStdDev: Float = 0.10f,
    val maxAbsLlr: Float = 4.0f,
) {
    init {
        require(softmaxTemperature > 0f) { "softmaxTemperature must be positive" }
        require(fallbackCohortStdDev > 0f) { "fallbackCohortStdDev must be positive" }
        require(maxAbsLlr > 0f) { "maxAbsLlr must be positive" }
    }
}

/**
 * Scores an observation against the whole household rather than against an absolute threshold.
 *
 * Rung L1 of the accuracy ladder (spec §4.1). Published EERs measure open-set verification with
 * a fixed threshold; we have a known roster, which turns the question into "which of these
 * people does this most resemble" and attacks sibling confusion directly.
 *
 * @param cohort every other enrolled household member. Must not be empty: enrolling only the
 *   target child is a design error (spec §5), and this constructor is where that rule is enforced.
 */
class HouseholdMatcher(
    private val target: VoiceProfile,
    private val cohort: List<VoiceProfile>,
    private val config: MatcherConfig = MatcherConfig(),
) {
    init {
        require(cohort.isNotEmpty()) {
            "cohort must not be empty: the parent and any siblings must be enrolled (spec §5)"
        }
    }

    fun score(observation: Embedding): MatchScore {
        val targetSimilarity = bestSimilarity(target, observation)
        val cohortSimilarities = cohort.map { bestSimilarity(it, observation) }
        val bestNonTarget = cohortSimilarities.max()

        val z = cohortZScore(targetSimilarity, cohortSimilarities)
        var llr = closedSetLogLikelihoodRatio(targetSimilarity, cohortSimilarities)

        // Open-set gates. A stranger or the television resembles nobody in particular, but the
        // posterior alone would still crown a winner among the enrolled. Both gates are needed:
        // the z floor only binds in the single-cohort-member fallback (see minTargetSimilarity),
        // and the absolute floor is what actually rejects a stranger in a normal household.
        if (z < config.minCohortZ) llr = minOf(llr, 0f)
        if (targetSimilarity < config.minTargetSimilarity) llr = minOf(llr, 0f)

        // "Not them" vetoes. A drifting profile should decay toward silence, never toward
        // firing on a sibling. The margin matters: without it the veto fires only when a negative
        // outright beats the target, which is not the drift case it exists for.
        val closestNegative = target.negatives.maxOfOrNull { cosine(observation, it) }
        if (closestNegative != null && closestNegative >= targetSimilarity - config.negativeVetoMargin) {
            llr = minOf(llr, 0f)
        }

        return MatchScore(
            targetSimilarity = targetSimilarity,
            bestNonTargetSimilarity = bestNonTarget,
            cohortZ = z,
            logLikelihoodRatio = llr.coerceIn(-config.maxAbsLlr, config.maxAbsLlr),
        )
    }

    /** The best-matching language sub-centroid, never an average across languages (spec §4.2). */
    private fun bestSimilarity(profile: VoiceProfile, observation: Embedding): Float =
        profile.centroids.values.maxOf { cosine(observation, it) }

    private fun cohortZScore(targetSimilarity: Float, cohortSimilarities: List<Float>): Float {
        if (cohortSimilarities.size < 2) {
            return (targetSimilarity - config.fallbackCohortMean) / config.fallbackCohortStdDev
        }
        val mean = cohortSimilarities.average().toFloat()
        var sumSquaredDeviation = 0.0
        for (similarity in cohortSimilarities) {
            val deviation = (similarity - mean).toDouble()
            sumSquaredDeviation += deviation * deviation
        }
        val stdDev = sqrt(sumSquaredDeviation / cohortSimilarities.size)
            .toFloat()
            .coerceAtLeast(config.minCohortStdDev)
        return (targetSimilarity - mean) / stdDev
    }

    private fun closedSetLogLikelihoodRatio(
        targetSimilarity: Float,
        cohortSimilarities: List<Float>,
    ): Float {
        val all = ArrayList<Float>(cohortSimilarities.size + 1)
        all += targetSimilarity
        all += cohortSimilarities

        // Subtract the maximum before exponentiating, for numerical stability.
        val maximum = all.max()
        var denominator = 0.0
        for (similarity in all) {
            denominator += exp(((similarity - maximum) / config.softmaxTemperature).toDouble())
        }
        val numerator = exp(((targetSimilarity - maximum) / config.softmaxTemperature).toDouble())
        val posterior = (numerator / denominator).coerceIn(1e-6, 1.0 - 1e-6)
        return ln(posterior / (1.0 - posterior)).toFloat()
    }
}
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `./gradlew :shared:jvmTest --tests "app.presence.engine.HouseholdMatcherTest"`
Expected: PASS, 13 tests.

- [ ] **Step 5: Commit**

```bash
git add shared/src/commonMain/kotlin/app/presence/engine/HouseholdMatcher.kt \
        shared/src/commonTest/kotlin/app/presence/engine/HouseholdMatcherTest.kt
git commit -m "feat: HouseholdMatcher scores against the roster, not a fixed threshold"
```

---

### Task 8: EvidenceAccumulator — L2 rolling-window LLR

**Files:**
- Create: `shared/src/commonMain/kotlin/app/presence/engine/EvidenceAccumulator.kt`
- Test: `shared/src/commonTest/kotlin/app/presence/engine/EvidenceAccumulatorTest.kt`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `data class Observation(timestampMs: Long, logLikelihoodRatio: Float)`; `data class AccumulatorConfig(windowMs: Long = 20_000, fireThreshold: Float = 6.0f, cooldownMs: Long = 60_000)`; `class EvidenceAccumulator(config)` with `accept(observation: Observation): Float?` and `reset()`. Tasks 9 and 16 use it.

**This is the primary accuracy mechanism, not a debouncer.** An earlier draft of the spec treated it as a convenience that stopped the app firing constantly; that was a design error, corrected in §4.1. No decision needs to rest on one 0.9-second word — a child seeking attention talks for tens of seconds, and summing weak per-segment evidence over a window is what turns an unusable per-utterance error rate into a confident decision.

The threshold on the *sum* is therefore the real operating point of the whole product, and Task 17 sweeps it.

- [ ] **Step 1: Write the failing test**

`shared/src/commonTest/kotlin/app/presence/engine/EvidenceAccumulatorTest.kt`:

```kotlin
package app.presence.engine

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class EvidenceAccumulatorTest {

    private fun accumulator(
        windowMs: Long = 20_000,
        fireThreshold: Float = 6.0f,
        cooldownMs: Long = 60_000,
    ) = EvidenceAccumulator(AccumulatorConfig(windowMs, fireThreshold, cooldownMs))

    private fun assertClose(expected: Float, actual: Float) {
        assertTrue(abs(expected - actual) < 1e-4f, "expected $expected but was $actual")
    }

    @Test
    fun `one weak observation does not fire`() {
        assertNull(accumulator().accept(Observation(0, 2.0f)))
    }

    @Test
    fun `evidence accumulates across observations until it fires`() {
        val subject = accumulator()

        assertNull(subject.accept(Observation(0, 2.0f)))
        assertNull(subject.accept(Observation(1_000, 2.0f)))
        val fired = subject.accept(Observation(2_000, 2.0f))

        assertClose(6.0f, assertNotNull(fired))
    }

    @Test
    fun `one very strong observation can fire alone`() {
        assertClose(7.0f, assertNotNull(accumulator().accept(Observation(0, 7.0f))))
    }

    @Test
    fun `observations older than the window stop counting`() {
        val subject = accumulator()

        subject.accept(Observation(0, 2.0f))
        subject.accept(Observation(1_000, 2.0f))
        // 25 s later both earlier observations are outside the 20 s window.
        val fired = subject.accept(Observation(25_000, 2.0f))

        assertNull(fired)
    }

    @Test
    fun `negative evidence cancels positive evidence`() {
        val subject = accumulator()

        subject.accept(Observation(0, 4.0f))
        subject.accept(Observation(1_000, -3.0f))
        assertNull(subject.accept(Observation(2_000, 4.0f)))

        assertClose(7.0f, assertNotNull(subject.accept(Observation(3_000, 2.0f))))
    }

    @Test
    fun `the window is cleared after firing`() {
        val subject = accumulator(cooldownMs = 0)

        subject.accept(Observation(0, 7.0f))

        assertNull(subject.accept(Observation(1_000, 2.0f)))
    }

    @Test
    fun `cooldown suppresses a second detection`() {
        val subject = accumulator()

        assertNotNull(subject.accept(Observation(0, 7.0f)))
        assertNull(subject.accept(Observation(1_000, 7.0f)))
        assertNull(subject.accept(Observation(30_000, 7.0f)))
    }

    @Test
    fun `firing resumes once the cooldown has elapsed`() {
        val subject = accumulator()

        subject.accept(Observation(0, 7.0f))

        assertNotNull(subject.accept(Observation(61_000, 7.0f)))
    }

    @Test
    fun `reset clears both the window and the cooldown`() {
        val subject = accumulator()

        subject.accept(Observation(0, 7.0f))
        subject.reset()

        assertNotNull(subject.accept(Observation(1_000, 7.0f)))
    }

    @Test
    fun `rejects an observation that goes backwards in time`() {
        val subject = accumulator()
        subject.accept(Observation(5_000, 1.0f))

        assertFailsWith<IllegalArgumentException> { subject.accept(Observation(4_999, 1.0f)) }
    }

    @Test
    fun `an observation exactly one window old still counts`() {
        val subject = accumulator()

        subject.accept(Observation(0, 4.0f))

        assertClose(6.0f, assertNotNull(subject.accept(Observation(20_000, 2.0f))))
    }

    @Test
    fun `an observation one millisecond past the window does not count`() {
        val subject = accumulator()

        subject.accept(Observation(0, 4.0f))

        assertNull(subject.accept(Observation(20_001, 2.0f)))
    }

    /**
     * Pins the deliberate choice not to accumulate while suppressed. Had the two in-cooldown
     * observations counted, the third would have fired on evidence gathered while the accumulator
     * was not allowed to act — turning the cooldown into a delay rather than a requirement for
     * fresh reason to interrupt.
     */
    @Test
    fun `evidence arriving during cooldown does not carry across the boundary`() {
        val subject = accumulator()
        assertNotNull(subject.accept(Observation(0, 7.0f)))

        assertNull(subject.accept(Observation(30_000, 3.0f)))
        assertNull(subject.accept(Observation(50_000, 3.0f)))

        // Cooldown has exactly elapsed, so this is allowed to fire — but on its own evidence,
        // which is below the threshold.
        assertNull(subject.accept(Observation(60_000, 3.0f)))
    }

    /**
     * The positive half of the cooldown boundary. The test above asserts null at the boundary,
     * which passes whether the boundary correctly allows evaluation or an off-by-one suppresses
     * it entirely — this one fires on its own evidence at exactly `cooldownMs` elapsed, so a
     * `<=` in place of `<` would be caught.
     */
    @Test
    fun `firing is allowed the moment the cooldown has exactly elapsed`() {
        val subject = accumulator()
        assertNotNull(subject.accept(Observation(0, 7.0f)))

        assertClose(7.0f, assertNotNull(subject.accept(Observation(60_000, 7.0f))))
    }

    @Test
    fun `rejects a non-positive window`() {
        assertFailsWith<IllegalArgumentException> { AccumulatorConfig(windowMs = 0) }
    }
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `./gradlew :shared:jvmTest --tests "app.presence.engine.EvidenceAccumulatorTest"`
Expected: FAIL — unresolved reference `EvidenceAccumulator`.

- [ ] **Step 3: Write the minimal implementation**

`shared/src/commonMain/kotlin/app/presence/engine/EvidenceAccumulator.kt`:

```kotlin
package app.presence.engine

/** One scored speech observation. [timestampMs] is a position in the audio stream. */
data class Observation(
    val timestampMs: Long,
    val logLikelihoodRatio: Float,
)

/**
 * @param windowMs how far back evidence counts. Longer is more accurate and slower to react;
 *   this trade-off is the core tuning decision of the product (spec §9).
 * @param fireThreshold accumulated evidence required to trigger. The real operating point —
 *   Phase 0 sweeps it rather than guessing.
 * @param cooldownMs quiet period after firing, so one long conversation produces one
 *   intervention rather than a stream of them.
 */
data class AccumulatorConfig(
    val windowMs: Long = 20_000,
    val fireThreshold: Float = 6.0f,
    val cooldownMs: Long = 60_000,
) {
    init {
        require(windowMs > 0) { "windowMs must be positive" }
        require(cooldownMs >= 0) { "cooldownMs must not be negative" }
    }
}

/**
 * Sums per-segment evidence over a rolling window and fires when it crosses the threshold.
 *
 * Rung L2, and **the primary accuracy mechanism of this design** (spec §4.1) — not a debouncer.
 * Per-utterance error on children's spontaneous speech is far too high to decide on one
 * 0.9-second word; a child seeking attention talks for tens of seconds, and accumulating weak
 * evidence across that span is what makes a confident decision possible.
 */
class EvidenceAccumulator(
    private val config: AccumulatorConfig = AccumulatorConfig(),
) {
    private val window = ArrayDeque<Observation>()
    private var lastFiredAtMs: Long? = null
    private var lastTimestampMs: Long? = null

    /** Returns the accumulated evidence if this observation triggers a detection, else null. */
    fun accept(observation: Observation): Float? {
        val previous = lastTimestampMs
        require(previous == null || observation.timestampMs >= previous) {
            "observations must arrive in non-decreasing timestamp order: " +
                "got ${observation.timestampMs} after $previous"
        }
        lastTimestampMs = observation.timestampMs

        val firedAt = lastFiredAtMs
        if (firedAt != null && observation.timestampMs - firedAt < config.cooldownMs) {
            // Deliberately does not accumulate. Evidence gathered while suppressed would
            // otherwise fire the instant the cooldown lapses, with no new speech since — which
            // makes the cooldown a delay rather than a requirement for fresh reason to interrupt.
            return null
        }

        window.addLast(observation)
        // Strict `<`: an observation exactly windowMs old still counts.
        val cutoff = observation.timestampMs - config.windowMs
        while (window.isNotEmpty() && window.first().timestampMs < cutoff) {
            window.removeFirst()
        }

        var total = 0f
        for (entry in window) total += entry.logLikelihoodRatio
        if (total < config.fireThreshold) return null

        lastFiredAtMs = observation.timestampMs
        window.clear()
        return total
    }

    /** Clears accumulated evidence, any active cooldown, and the stream position. */
    fun reset() {
        window.clear()
        lastFiredAtMs = null
        lastTimestampMs = null
    }
}
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `./gradlew :shared:jvmTest --tests "app.presence.engine.EvidenceAccumulatorTest"`
Expected: PASS, 15 tests.

- [ ] **Step 5: Commit**

```bash
git add shared/src/commonMain/kotlin/app/presence/engine/EvidenceAccumulator.kt \
        shared/src/commonTest/kotlin/app/presence/engine/EvidenceAccumulatorTest.kt
git commit -m "feat: EvidenceAccumulator sums LLR across segments with cooldown"
```

---

### Task 9: DetectionEngine — composing the pipeline

**Files:**
- Create: `shared/src/commonMain/kotlin/app/presence/engine/SpeakerEmbedder.kt`
- Create: `shared/src/commonMain/kotlin/app/presence/domain/DetectionEvent.kt`
- Create: `shared/src/commonMain/kotlin/app/presence/engine/DetectionEngine.kt`
- Test: `shared/src/commonTest/kotlin/app/presence/engine/DetectionEngineTest.kt`

**Interfaces:**
- Consumes: `SAMPLE_RATE_HZ` (Task 1); `VadGate`, `SpeechSegment`, `VoiceActivityDetector` (Task 5); `HouseholdMatcher` (Task 7); `EvidenceAccumulator`, `Observation` (Task 8).
- Produces: `interface SpeakerEmbedder { val dimension: Int; fun embed(samples: FloatArray): Embedding }`; `data class DetectionEvent(personId: String, timestampMs: Long, accumulatedEvidence: Float, peakSimilarity: Float)`; `class DetectionEngineConfig(minEmbedDurationMs, embedWindowMs, embedHopMs)`; `class DetectionEngine(vadGate, embedder, matcher, accumulator, targetPersonId, config)` with `accept(frame: FloatArray): DetectionEvent?`, `flush(): DetectionEvent?`, `reset()`. Task 17 drives it.

This is the class spec §3 promised: it knows nothing about microphones, services, or screens, so the whole recognition pipeline is testable on the JVM.

One subtlety worth getting right: a long speech segment is cut into overlapping embedding windows, and **each window is a separate observation**, not averaged into one. Averaging would discard exactly the independent evidence the accumulator exists to sum.

- [ ] **Step 1: Write the failing test**

`shared/src/commonTest/kotlin/app/presence/engine/DetectionEngineTest.kt`:

```kotlin
package app.presence.engine

import app.presence.domain.DetectionEvent
import app.presence.domain.Language
import app.presence.domain.VoiceProfile
import kotlin.test.Test
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** Always reports speech, so tests control the audio content rather than the gating. */
private class AlwaysSpeech(override val frameSize: Int = 512) : VoiceActivityDetector {
    override fun speechProbability(frame: FloatArray) = 0.99f
    override fun reset() = Unit
}

private class NeverSpeech(override val frameSize: Int = 512) : VoiceActivityDetector {
    override fun speechProbability(frame: FloatArray) = 0.01f
    override fun reset() = Unit
}

/** Maps loud audio to the target's voice and quiet audio to the parent's. */
private class FakeEmbedder : SpeakerEmbedder {
    override val dimension = 3
    override fun embed(samples: FloatArray): Embedding {
        val mean = samples.average()
        return if (mean > 0.5) {
            Embedding.ofNormalized(floatArrayOf(1f, 0f, 0f))
        } else {
            Embedding.ofNormalized(floatArrayOf(0f, 1f, 0f))
        }
    }
}

class DetectionEngineTest {

    private fun profile(id: String, role: VoiceProfile.Role, vararg values: Float) = VoiceProfile(
        personId = id,
        role = role,
        centroids = mapOf(Language.ENGLISH to Embedding.ofNormalized(floatArrayOf(*values))),
    )

    private val maya = profile("maya", VoiceProfile.Role.TARGET_CHILD, 1f, 0f, 0f)
    private val parent = profile("parent", VoiceProfile.Role.PARENT, 0f, 1f, 0f)
    private val sibling = profile("sibling", VoiceProfile.Role.SIBLING, 0f, 0f, 1f)

    private fun engine(vad: VoiceActivityDetector = AlwaysSpeech()) = DetectionEngine(
        vadGate = VadGate(vad),
        embedder = FakeEmbedder(),
        matcher = HouseholdMatcher(maya, listOf(parent, sibling)),
        accumulator = EvidenceAccumulator(),
        targetPersonId = "maya",
    )

    private fun targetFrame() = FloatArray(512) { 1.0f }
    private fun parentFrame() = FloatArray(512) { 0.1f }

    /** 200 frames is ~6.4 s, enough for two capped 3 s segments. */
    private fun feed(subject: DetectionEngine, frame: () -> FloatArray, count: Int = 200): DetectionEvent? {
        var event: DetectionEvent? = null
        repeat(count) { if (event == null) event = subject.accept(frame()) else subject.accept(frame()) }
        return event
    }

    @Test
    fun `fires once enough evidence for the target accumulates`() {
        val event = feed(engine(), ::targetFrame)

        val fired = assertNotNull(event, "expected a detection after ~6 s of the target's voice")
        assertTrue(fired.personId == "maya")
        assertTrue(fired.accumulatedEvidence >= 6f)
    }

    @Test
    fun `does not fire for a cohort member's voice`() {
        assertNull(feed(engine(), ::parentFrame))
    }

    @Test
    fun `does not fire when there is no speech`() {
        assertNull(feed(engine(NeverSpeech()), ::targetFrame))
    }

    @Test
    fun `ignores a segment too short to embed`() {
        val subject = engine()

        // 20 frames is 640 ms — above the 400 ms VAD floor, below the 800 ms embed floor.
        repeat(20) { subject.accept(targetFrame()) }

        assertNull(subject.flush())
    }

    @Test
    fun `flush emits a detection completed by the final segment`() {
        val subject = engine()
        var duringStream: DetectionEvent? = null

        // 140 frames is ~4.5 s: one capped 3 s segment plus a 1.5 s tail left open.
        repeat(140) { if (duringStream == null) duringStream = subject.accept(targetFrame()) }

        val flushed = subject.flush()
        assertTrue(duringStream != null || flushed != null, "the stream should have produced a detection")
    }

    @Test
    fun `reset clears accumulated evidence`() {
        val subject = engine()

        repeat(100) { subject.accept(targetFrame()) }
        subject.reset()

        // A single 3 s segment on its own is not enough to reach the threshold.
        assertNull(feed(subject, ::targetFrame, count = 94))
    }

    @Test
    fun `a long segment yields several observations rather than one average`() {
        // With a 6 s cap, one segment spans two 3 s embedding windows, so the threshold is
        // reachable inside a single segment. Averaging the windows would prevent that.
        val subject = DetectionEngine(
            vadGate = VadGate(AlwaysSpeech(), VadGateConfig(maxSegmentMs = 6_000)),
            embedder = FakeEmbedder(),
            matcher = HouseholdMatcher(maya, listOf(parent, sibling)),
            accumulator = EvidenceAccumulator(),
            targetPersonId = "maya",
        )

        var event: DetectionEvent? = null
        repeat(190) { if (event == null) event = subject.accept(targetFrame()) }

        assertNotNull(event, "a single long segment should contribute multiple observations")
    }
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `./gradlew :shared:jvmTest --tests "app.presence.engine.DetectionEngineTest"`
Expected: FAIL — unresolved reference `SpeakerEmbedder`.

- [ ] **Step 3: Write the embedder interface and the event type**

`shared/src/commonMain/kotlin/app/presence/engine/SpeakerEmbedder.kt`:

```kotlin
package app.presence.engine

/**
 * Turns speech audio into a speaker embedding.
 *
 * An interface rather than an `expect` declaration so tests can substitute a fake — the whole
 * point of keeping the engine in common code. Production implementations wrap ONNX Runtime.
 */
interface SpeakerEmbedder {

    /** Dimensionality of the embeddings this implementation produces. */
    val dimension: Int

    /** Embeds mono 16 kHz samples. The result is L2-normalized by [Embedding.ofNormalized]. */
    fun embed(samples: FloatArray): Embedding
}
```

`shared/src/commonMain/kotlin/app/presence/domain/DetectionEvent.kt`:

```kotlin
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
```

- [ ] **Step 4: Write the engine**

`shared/src/commonMain/kotlin/app/presence/engine/DetectionEngine.kt`:

```kotlin
package app.presence.engine

import app.presence.audio.SAMPLE_RATE_HZ
import app.presence.audio.millisToSamples
import app.presence.domain.DetectionEvent

/**
 * @param minEmbedDurationMs segments shorter than this are dropped unembedded. Published work
 *   found utterances averaging 0.92 s too short to evaluate meaningfully (spec §4.1).
 * @param embedWindowMs length of each embedding window within a longer segment
 * @param embedHopMs advance between windows; overlap means more observations per segment
 */
class DetectionEngineConfig(
    val minEmbedDurationMs: Long = 800,
    val embedWindowMs: Long = 3_000,
    val embedHopMs: Long = 1_500,
) {
    init {
        require(minEmbedDurationMs > 0) { "minEmbedDurationMs must be positive" }
        require(embedWindowMs >= minEmbedDurationMs) { "embedWindowMs must be >= minEmbedDurationMs" }
        require(embedHopMs in 1..embedWindowMs) { "embedHopMs must be in 1..embedWindowMs" }
    }
}

/**
 * The complete detection pipeline: frames in, [DetectionEvent] out.
 *
 * Knows nothing about microphones, foreground services, or screens (spec §3), which is what
 * makes the whole recognizer testable on the JVM against recorded fixtures.
 */
class DetectionEngine(
    private val vadGate: VadGate,
    private val embedder: SpeakerEmbedder,
    private val matcher: HouseholdMatcher,
    private val accumulator: EvidenceAccumulator,
    private val targetPersonId: String,
    private val config: DetectionEngineConfig = DetectionEngineConfig(),
) {
    private class Window(val offsetSamples: Int, val samples: FloatArray)

    /** Feeds one VAD-sized frame. Returns an event if the threshold was crossed. */
    fun accept(frame: FloatArray): DetectionEvent? = vadGate.accept(frame)?.let(::process)

    /** Processes any utterance still open at end of stream. */
    fun flush(): DetectionEvent? = vadGate.flush()?.let(::process)

    /** Clears all state. Call between independent recordings. */
    fun reset() {
        vadGate.reset()
        accumulator.reset()
    }

    private fun process(segment: SpeechSegment): DetectionEvent? {
        if (segment.durationMs < config.minEmbedDurationMs) return null

        var event: DetectionEvent? = null
        for (window in windowsOf(segment.samples)) {
            val score = matcher.score(embedder.embed(window.samples))
            val timestampMs = toMillis(segment.startSampleIndex + window.offsetSamples + window.samples.size)
            val evidence = accumulator.accept(Observation(timestampMs, score.logLikelihoodRatio))
            // Only the first fire in a segment is returned, since accept() yields at most one
            // event. Unreachable under the default 60 s cooldown, which far exceeds any segment,
            // but a caller configuring a cooldown shorter than maxSegmentMs would have a second
            // within-segment detection coalesced into the first.
            if (evidence != null && event == null) {
                event = DetectionEvent(
                    personId = targetPersonId,
                    timestampMs = timestampMs,
                    accumulatedEvidence = evidence,
                    peakSimilarity = score.targetSimilarity,
                )
            }
        }
        return event
    }

    /**
     * Splits a segment into overlapping embedding windows.
     *
     * Each window becomes its own observation rather than being averaged into one: averaging
     * would discard exactly the independent evidence the accumulator exists to sum (spec §4.1).
     */
    private fun windowsOf(samples: FloatArray): List<Window> {
        val windowSamples = millisToSamples(config.embedWindowMs)
        val hopSamples = millisToSamples(config.embedHopMs)
        if (samples.size <= windowSamples) return listOf(Window(0, samples))

        val windows = ArrayList<Window>()
        var offset = 0
        while (offset + windowSamples <= samples.size) {
            windows += Window(offset, samples.copyOfRange(offset, offset + windowSamples))
            offset += hopSamples
        }
        // Cover the tail with a final window ending at the segment boundary — but only when the
        // striding loop missed at least one hop of audio. Without a guard, a segment a few
        // milliseconds longer than one window emits a second window overlapping the first by
        // ~99%, and the accumulator sums near-identical audio as independent evidence.
        //
        // The threshold is the hop, not the minimum embed duration: overlap with the previous
        // window is `windowSamples - uncovered`, so requiring a full hop bounds the tail's overlap
        // at exactly the same 50% that every regular striding pair already has. A smaller
        // threshold would leave the last pair markedly more correlated than the rest, while the
        // accumulator sums them all as though equally independent.
        val uncovered = samples.size - (offset - hopSamples + windowSamples)
        if (uncovered >= hopSamples) {
            val start = samples.size - windowSamples
            windows += Window(start, samples.copyOfRange(start, samples.size))
        }
        return windows
    }

    private fun toMillis(sampleIndex: Long): Long = sampleIndex * 1_000L / SAMPLE_RATE_HZ
}
```

- [ ] **Step 5: Run the test and verify it passes**

Run: `./gradlew :shared:jvmTest --tests "app.presence.engine.DetectionEngineTest"`
Expected: PASS, 7 tests.

- [ ] **Step 6: Run the whole suite**

Run: `./gradlew :shared:jvmTest`
Expected: PASS, 81 tests across all engine classes.

- [ ] **Step 7: Commit**

```bash
git add shared/src/commonMain/kotlin/app/presence/engine/SpeakerEmbedder.kt \
        shared/src/commonMain/kotlin/app/presence/engine/DetectionEngine.kt \
        shared/src/commonMain/kotlin/app/presence/domain/DetectionEvent.kt \
        shared/src/commonTest/kotlin/app/presence/engine/DetectionEngineTest.kt
git commit -m "feat: DetectionEngine composes VAD, embedding, matching and accumulation"
```

---

### Task 10: Harness module and WAV reading

**Files:**
- Modify: `settings.gradle.kts` — add `include(":harness")`
- Create: `harness/build.gradle.kts`
- Create: `harness/src/main/kotlin/app/presence/harness/WavReader.kt`
- Test: `harness/src/test/kotlin/app/presence/harness/WavReaderTest.kt`

**Interfaces:**
- Consumes: `SAMPLE_RATE_HZ` (Task 1).
- Produces: `object WavReader` with `read(path: java.nio.file.Path): FloatArray`; `class UnsupportedAudioException(message: String) : Exception(message)`. Tasks 16 and 17 use it.

The reader **refuses to resample**. Spec §9.1 requires corpora at 16 kHz mono, and silently resampling would corrupt exactly the spectral detail speaker embeddings depend on. A wrong-format file produces an exception naming the file and giving the exact `ffmpeg` command to fix it — the engineer running Phase 0 over a few thousand files needs the fix, not a stack trace.

Stereo is downmixed, because that is lossless in the sense that matters and a stereo phone recording is common.

- [ ] **Step 1: Write the failing test**

`harness/src/test/kotlin/app/presence/harness/WavReaderTest.kt`:

```kotlin
package app.presence.harness

import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.writeText
import javax.sound.sampled.AudioFileFormat
import javax.sound.sampled.AudioFormat
import javax.sound.sampled.AudioInputStream
import javax.sound.sampled.AudioSystem
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class WavReaderTest {

    /** Writes a 16-bit PCM WAV so tests never depend on a committed binary fixture. */
    private fun writeWav(
        path: Path,
        samplesPerChannel: List<List<Short>>,
        sampleRate: Float = 16_000f,
    ) {
        val channels = samplesPerChannel.size
        val frames = samplesPerChannel[0].size
        val buffer = ByteBuffer.allocate(frames * channels * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (frame in 0 until frames) {
            for (channel in 0 until channels) buffer.putShort(samplesPerChannel[channel][frame])
        }
        val format = AudioFormat(sampleRate, 16, channels, true, false)
        AudioSystem.write(
            AudioInputStream(buffer.array().inputStream(), format, frames.toLong()),
            AudioFileFormat.Type.WAVE,
            path.toFile(),
        )
    }

    private fun tempWav(name: String): Path =
        Files.createTempDirectory("wav-reader-test").resolve(name)

    @Test
    fun `reads 16 kHz mono into normalized floats`() {
        val path = tempWav("mono.wav")
        writeWav(path, listOf(listOf(0, 16384, -16384, 32767)))

        val samples = WavReader.read(path)

        assertEquals(4, samples.size)
        assertTrue(abs(0f - samples[0]) < 1e-3f)
        assertTrue(abs(0.5f - samples[1]) < 1e-3f, "got ${samples[1]}")
        assertTrue(abs(-0.5f - samples[2]) < 1e-3f, "got ${samples[2]}")
        assertTrue(samples[3] in 0.99f..1.0f, "got ${samples[3]}")
    }

    @Test
    fun `keeps every sample in the normalized range`() {
        val path = tempWav("extremes.wav")
        writeWav(path, listOf(listOf(-32768, 32767, 0)))

        WavReader.read(path).forEach {
            assertTrue(it in -1.0f..1.0f, "sample $it escaped the normalized range")
        }
    }

    @Test
    fun `downmixes stereo to mono by averaging the channels`() {
        val path = tempWav("stereo.wav")
        writeWav(path, listOf(listOf(16384, 0), listOf(0, 16384)))

        val samples = WavReader.read(path)

        assertEquals(2, samples.size)
        assertTrue(abs(0.25f - samples[0]) < 1e-3f, "got ${samples[0]}")
        assertTrue(abs(0.25f - samples[1]) < 1e-3f, "got ${samples[1]}")
    }

    @Test
    fun `rejects a wrong sample rate instead of resampling`() {
        val path = tempWav("44k.wav")
        writeWav(path, listOf(listOf(0, 1000)), sampleRate = 44_100f)

        val failure = assertFailsWith<UnsupportedAudioException> { WavReader.read(path) }

        assertTrue(failure.message!!.contains("44100"), "message should name the actual rate")
        assertTrue(failure.message!!.contains("ffmpeg"), "message should give the fix command")
        assertTrue(failure.message!!.contains("44k.wav"), "message should name the file")
    }

    @Test
    fun `rejects a missing file with a clear message`() {
        val failure = assertFailsWith<UnsupportedAudioException> {
            WavReader.read(Path.of("does-not-exist.wav"))
        }

        assertTrue(failure.message!!.contains("does-not-exist.wav"))
    }

    @Test
    fun `rejects a bit depth other than 16, naming the fix`() {
        val path = tempWav("8bit.wav")
        val samples = byteArrayOf(0, 64, -64)
        val format = AudioFormat(16_000f, 8, 1, true, false)
        AudioSystem.write(
            AudioInputStream(samples.inputStream(), format, samples.size.toLong()),
            AudioFileFormat.Type.WAVE,
            path.toFile(),
        )

        val failure = assertFailsWith<UnsupportedAudioException> { WavReader.read(path) }

        assertTrue(failure.message!!.contains("8-bit"), failure.message!!)
        assertTrue(failure.message!!.contains("ffmpeg"), "should give the fix command")
    }

    @Test
    fun `reports an unreadable file as unsupported rather than leaking a checked exception`() {
        val path = tempWav("not-audio.wav")
        path.writeText("this is not a wav file at all")

        val failure = assertFailsWith<UnsupportedAudioException> { WavReader.read(path) }

        assertTrue(failure.message!!.contains("not-audio.wav"))
    }

    @Test
    fun `reads a file with many samples`() {
        val path = tempWav("long.wav")
        writeWav(path, listOf(List(16_000) { (it % 1000).toShort() }))

        assertEquals(16_000, WavReader.read(path).size)
    }
}
```

- [ ] **Step 2: Add the module and build file**

Append to `settings.gradle.kts`:

```kotlin
include(":harness")
```

`harness/build.gradle.kts`:

```kotlin
plugins {
    alias(libs.plugins.kotlinJvm)
    application
}

kotlin {
    jvmToolchain(17)
}

dependencies {
    implementation(project(":shared"))
    testImplementation(kotlin("test"))
}

application {
    mainClass.set("app.presence.harness.MainKt")
}

// Same two-suite split as :shared — see Global Constraints.
tasks.test {
    useJUnitPlatform {
        excludeTags("model")
    }
}

tasks.register<Test>("integrationTest") {
    description = "Runs tests that require the ONNX model files in models/."
    group = "verification"
    testClassesDirs = sourceSets.test.get().output.classesDirs
    classpath = sourceSets.test.get().runtimeClasspath
    useJUnitPlatform {
        includeTags("model")
    }
}
```

- [ ] **Step 3: Run the test and verify it fails**

Run: `./gradlew :harness:test`
Expected: FAIL — unresolved reference `WavReader`.

- [ ] **Step 4: Write the minimal implementation**

`harness/src/main/kotlin/app/presence/harness/WavReader.kt`:

```kotlin
package app.presence.harness

import app.presence.audio.SAMPLE_RATE_HZ
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Path
import javax.sound.sampled.AudioInputStream
import javax.sound.sampled.AudioSystem
import javax.sound.sampled.UnsupportedAudioFileException
import kotlin.io.path.exists
import kotlin.io.path.name

/** Thrown when a file cannot be used as-is. The message always says how to fix it. */
class UnsupportedAudioException(message: String) : Exception(message)

/**
 * Reads 16 kHz mono 16-bit PCM WAV into normalized floats.
 *
 * Deliberately **refuses to resample**. Spec §9.1 mandates 16 kHz mono corpora, and silent
 * resampling would corrupt exactly the spectral detail speaker embeddings depend on — a bug
 * that would show up as unexplained accuracy loss rather than as an error.
 */
object WavReader {

    private const val BITS_PER_SAMPLE = 16
    private const val FULL_SCALE = 32_768f

    fun read(path: Path): FloatArray {
        if (!path.exists()) {
            throw UnsupportedAudioException("audio file not found: $path")
        }

        // Everything touching the audio layer sits inside this try. AudioSystem throws checked
        // exceptions on both open and read, and Task 17 walks thousands of corpus files catching
        // UnsupportedAudioException per file — a foreign type escaping here would kill the batch.
        return try {
            AudioSystem.getAudioInputStream(path.toFile()).use { audio -> decode(path, audio) }
        } catch (failure: UnsupportedAudioFileException) {
            throw UnsupportedAudioException("${path.name}: not a readable audio file (${failure.message})")
        } catch (failure: IOException) {
            throw UnsupportedAudioException("${path.name}: could not be read (${failure.message})")
        }
    }

    private fun decode(path: Path, audio: AudioInputStream): FloatArray {
        val format = audio.format

        if (format.sampleRate.toInt() != SAMPLE_RATE_HZ) {
            throw UnsupportedAudioException(
                "${path.name}: sample rate is ${format.sampleRate.toInt()} Hz, " +
                    "expected $SAMPLE_RATE_HZ Hz. Convert it with:\n" +
                    "  ffmpeg -i ${path.name} -ar $SAMPLE_RATE_HZ -ac 1 -c:a pcm_s16le converted.wav",
            )
        }
        if (format.sampleSizeInBits != BITS_PER_SAMPLE) {
            throw UnsupportedAudioException(
                "${path.name}: ${format.sampleSizeInBits}-bit samples, expected $BITS_PER_SAMPLE-bit PCM. " +
                    "Convert it with:\n" +
                    "  ffmpeg -i ${path.name} -ar $SAMPLE_RATE_HZ -ac 1 -c:a pcm_s16le converted.wav",
            )
        }

        val channels = format.channels
        val bytes = audio.readAllBytes()
        val order = if (format.isBigEndian) ByteOrder.BIG_ENDIAN else ByteOrder.LITTLE_ENDIAN
        val shorts = ByteBuffer.wrap(bytes).order(order).asShortBuffer()

        // The one place a corrupt file could otherwise degrade quietly.
        if (shorts.limit() % channels != 0) {
            throw UnsupportedAudioException(
                "${path.name}: ${shorts.limit()} samples do not divide evenly into $channels " +
                    "channels — the file is truncated",
            )
        }

        val frameCount = shorts.limit() / channels
        val samples = FloatArray(frameCount)
        for (frame in 0 until frameCount) {
            var sum = 0f
            for (channel in 0 until channels) {
                sum += shorts.get(frame * channels + channel) / FULL_SCALE
            }
            // Dividing by 32_768 already bounds this to [-1, 1]; the clamp is belt-and-braces
            // against a malformed frame count rather than the thing enforcing the range.
            samples[frame] = (sum / channels).coerceIn(-1f, 1f)
        }
        return samples
    }
}
```

- [ ] **Step 5: Run the test and verify it passes**

Run: `./gradlew :harness:test`
Expected: PASS, 8 tests.

- [ ] **Step 6: Commit**

```bash
git add settings.gradle.kts harness/
git commit -m "feat: harness module with a strict 16 kHz mono WAV reader"
```

---

### Task 11: ONNX session and model acquisition

**Files:**
- Create: `tools/list-models.sh`
- Create: `tools/fetch-models.sh`
- Create: `shared/src/jvmMain/kotlin/app/presence/onnx/OnnxModelSession.kt`
- Test: `shared/src/jvmTest/kotlin/app/presence/onnx/OnnxModelSessionTest.kt`
- Modify: `README.md` — add a "Running Phase 0" section

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `class OnnxModelSession(modelPath: java.nio.file.Path) : AutoCloseable` with `inputNames: List<String>`, `outputNames: List<String>`, `describeSignature(): String`, `floatTensor(data: FloatArray, shape: LongArray): OnnxTensor`, `runTensors(inputs: Map<String, OnnxTensor>, block: (OrtSession.Result) -> T): T`, and `close()`. Tasks 12 and 14 build on it.

**The model files already exist** in `models/` (see Global Constraints) with their signatures
verified, so this task does not need to discover anything. The scripts exist for reproducibility —
a fresh clone, or a future model swap — not because anything here is unknown.

`describeSignature()` earns its place anyway: it is how the next person confirms a swapped model
before writing an adapter against it, and how you diagnose an ONNX Runtime shape complaint.

- [ ] **Step 1: Write the model-listing script**

`tools/list-models.sh`:

```bash
#!/usr/bin/env bash
# Lists the speaker-recognition models sherpa-onnx publishes, plus the Silero VAD asset.
# Note the upstream release tag really is spelled "recongition".
#
# You do not need this to run Phase 0 — fetch-models.sh already defaults to the chosen model.
# It exists for swapping models later, e.g. after L3 fine-tuning.
set -euo pipefail

echo "== sherpa-onnx speaker embedding models =="
curl -fsSL \
  "https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/tags/speaker-recongition-models" \
  | grep '"browser_download_url"' \
  | sed -E 's/.*"([^"]+)".*/\1/' \
  | sort

echo
echo "== Silero VAD =="
echo "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx"
echo
echo "NOTE: every one of these consumes filterbank features, not raw audio."
echo "Check any replacement's signature before trusting the existing adapters,"
echo "and check its licence before shipping (spec §10, open question 2)."
```

- [ ] **Step 2: Write the fetch script**

`tools/fetch-models.sh`:

```bash
#!/usr/bin/env bash
# Downloads the models Phase 0 needs into models/. Binaries are gitignored (*.onnx).
#
# Usage: ./tools/fetch-models.sh [embedding-model-url]
#   Defaults to the model Phase 0 was designed around. Run ./tools/list-models.sh for others.
set -euo pipefail

DEFAULT_EMBEDDING="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx"
EMBEDDING_URL="${1:-$DEFAULT_EMBEDDING}"
VAD_URL="https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx"
DEST="models"

mkdir -p "$DEST"

download() {
  local url="$1" dest="$2" label="$3"
  local tmp="${dest}.partial"

  echo "Fetching ${label}..."
  curl -fsSL "$url" -o "$tmp"

  # An ONNX file is protobuf; its opening bytes are never an HTML document. This catches a 200
  # response that is a landing page rather than the model — the one failure mode curl -f misses.
  if head -c 512 "$tmp" | grep -qiE '<!doctype html|<html'; then
    rm -f "$tmp"
    echo "error: ${url} returned an HTML page, not a model" >&2
    exit 1
  fi

  local size
  size=$(wc -c < "$tmp")
  if [ "$size" -lt 1000000 ]; then
    rm -f "$tmp"
    echo "error: ${url} returned only ${size} bytes; expected a multi-megabyte model" >&2
    exit 1
  fi

  # Only replace the existing file once the download is known good, so a failure mid-transfer
  # cannot destroy a working model.
  mv "$tmp" "$dest"
}

download "$VAD_URL" "$DEST/silero_vad.onnx" "Silero VAD"
download "$EMBEDDING_URL" "$DEST/speaker_embedding.onnx" "speaker embedding model"

# Which model produced which numbers is part of every Phase 0 result.
cat > "$DEST/PROVENANCE.txt" <<PROVENANCE
vad: $VAD_URL
embedding: $EMBEDDING_URL
PROVENANCE

ls -lh "$DEST"
echo
echo "Recorded provenance in $DEST/PROVENANCE.txt — quote it in any results you report."
```

Make both executable and confirm the models are present:

```bash
chmod +x tools/list-models.sh tools/fetch-models.sh
ls -lh models/
```

They should already be there. If not, run `./tools/fetch-models.sh`.

- [ ] **Step 3: Write the failing test**

`shared/src/jvmTest/kotlin/app/presence/onnx/OnnxModelSessionTest.kt`:

```kotlin
package app.presence.onnx

import ai.onnxruntime.TensorInfo
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import org.junit.jupiter.api.Tag

/** Requires the files in models/. Run with ./gradlew :shared:integrationTest */
@Tag("model")
class OnnxModelSessionTest {

    private val vadModel = Path.of("../models/silero_vad.onnx")
    private val embeddingModel = Path.of("../models/speaker_embedding.onnx")

    @Test
    fun `reports a missing model file clearly`() {
        val failure = assertFailsWith<IllegalArgumentException> {
            OnnxModelSession(Path.of("../models/not-a-real-model.onnx"))
        }
        assertTrue(failure.message!!.contains("not-a-real-model.onnx"))
        assertTrue(failure.message!!.contains("fetch-models"), "should say how to fix it")
    }

    @Test
    fun `exposes the verified Silero VAD signature`() {
        OnnxModelSession(vadModel).use { session ->
            println("=== Silero VAD ===\n${session.describeSignature()}")

            assertTrue("input" in session.inputNames, "inputs were ${session.inputNames}")
            assertTrue("state" in session.inputNames, "expected v5 single state tensor")
            assertTrue("sr" in session.inputNames)
            assertTrue("output" in session.outputNames)
            assertTrue("stateN" in session.outputNames)

            // Shapes, not just names: Task 12 hardcodes these dimensions.
            val state = session.inputInfo.getValue("state").info as TensorInfo
            assertEquals(2L, state.shape[0], "Silero v5 carries 2 state layers")
            assertEquals(128L, state.shape[2], "Silero v5 state width is 128")
            // stateN is deliberately not asserted: its static metadata reports a fully dynamic
            // [-1, -1, -1] rather than mirroring state's declared shape.
        }
    }

    @Test
    fun `exposes the verified embedding model signature`() {
        OnnxModelSession(embeddingModel).use { session ->
            println("=== Speaker embedding ===\n${session.describeSignature()}")

            assertTrue("feats" in session.inputNames, "inputs were ${session.inputNames}")
            assertTrue("embs" in session.outputNames, "outputs were ${session.outputNames}")

            // Shapes, not just names. A swapped model exposing the same names with a different
            // feature width or embedding dimension would otherwise pass, and Tasks 13 and 14
            // hardcode 80 and 256.
            val feats = session.inputInfo.getValue("feats").info as TensorInfo
            assertEquals(3, feats.shape.size, "feats should be [B, T, 80]")
            assertEquals(80L, feats.shape[2], "Task 13's filterbank produces 80 bins")

            val embs = session.outputInfo.getValue("embs").info as TensorInfo
            assertEquals(2, embs.shape.size, "embs should be [B, 256]")
            assertEquals(256L, embs.shape[1], "Task 14 expects 256 dimensions")
        }
    }
}
```

- [ ] **Step 4: Run the test and verify it fails**

Run: `./gradlew :shared:integrationTest --tests "app.presence.onnx.OnnxModelSessionTest"`
Expected: FAIL — unresolved reference `OnnxModelSession`.

- [ ] **Step 5: Write the implementation**

`shared/src/jvmMain/kotlin/app/presence/onnx/OnnxModelSession.kt`:

```kotlin
package app.presence.onnx

import ai.onnxruntime.NodeInfo
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.nio.file.Path
import kotlin.io.path.exists

/**
 * A loaded ONNX model.
 *
 * [describeSignature] exists so a swapped model can be checked before an adapter is written
 * against it — input and output names vary between models and between versions of the same model.
 */
class OnnxModelSession(modelPath: Path) : AutoCloseable {

    init {
        require(modelPath.exists()) {
            "ONNX model not found: ${modelPath.toAbsolutePath()} — run ./tools/fetch-models.sh"
        }
    }

    private val environment: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession =
        environment.createSession(modelPath.toAbsolutePath().toString(), OrtSession.SessionOptions())

    val inputNames: List<String> get() = session.inputNames.toList()
    val outputNames: List<String> get() = session.outputNames.toList()

    /** Full node info, so a test can assert declared tensor shapes and not just names. */
    val inputInfo: Map<String, NodeInfo> get() = session.inputInfo
    val outputInfo: Map<String, NodeInfo> get() = session.outputInfo

    /** Human-readable dump of every input and output, with shapes. */
    fun describeSignature(): String = buildString {
        appendLine("inputs:")
        session.inputInfo.forEach { (name, info) -> appendLine("  $name: ${info.info}") }
        appendLine("outputs:")
        session.outputInfo.forEach { (name, info) -> appendLine("  $name: ${info.info}") }
    }

    /**
     * Builds a float tensor of the given shape. [runTensors] closes it for you.
     *
     * ONNX Runtime copies a non-direct buffer into its own direct allocation synchronously here,
     * so the caller's array is not aliased and may be reused or mutated once this returns.
     */
    fun floatTensor(data: FloatArray, shape: LongArray): OnnxTensor =
        OnnxTensor.createTensor(environment, FloatBuffer.wrap(data), shape)

    /** Runs the model, hands the result to [block], and closes every tensor afterwards. */
    fun <T> runTensors(inputs: Map<String, OnnxTensor>, block: (OrtSession.Result) -> T): T =
        try {
            session.run(inputs).use(block)
        } finally {
            inputs.values.forEach { it.close() }
        }

    override fun close() {
        session.close()
    }
}
```

- [ ] **Step 6: Run the test and verify it passes**

Run: `./gradlew :shared:integrationTest --tests "app.presence.onnx.OnnxModelSessionTest"`
Expected: PASS, 3 tests, with both signatures printed.

Also confirm the split works — the model tests must not run in the default suite:

Run: `./gradlew :shared:jvmTest`
Expected: PASS, and `OnnxModelSessionTest` does **not** appear.

- [ ] **Step 7: Document the workflow in the README**

Add to `README.md`, just before the `## Attribution` heading:

```markdown
## Running Phase 0

Requires JDK 17 and Gradle 8.11.

```bash
./tools/fetch-models.sh                         # download models into models/ (gitignored)
./gradlew check                                 # pure-logic tests — no models, no network
./gradlew integrationTest                       # tests that need the real model files
./gradlew :harness:run --args="--help"          # evaluation CLI
```

Model files are never committed. `models/PROVENANCE.txt` records which ones a given result used —
quote it whenever you report numbers.
```

- [ ] **Step 8: Commit**

```bash
git add tools/ shared/src/jvmMain/ shared/src/jvmTest/ README.md
git commit -m "feat: ONNX session wrapper and model acquisition scripts"
```

---

### Task 12: SileroVad — the real voice activity detector

**Files:**
- Create: `shared/src/jvmMain/kotlin/app/presence/onnx/SileroVad.kt`
- Test: `shared/src/jvmTest/kotlin/app/presence/onnx/SileroVadTest.kt`

**Interfaces:**
- Consumes: `VoiceActivityDetector` (Task 5); `OnnxModelSession` (Task 11).
- Produces: `class SileroVad(session: OnnxModelSession, frameSize: Int = 512) : VoiceActivityDetector, AutoCloseable`. Task 17 constructs it.

**The signature is verified** (see Global Constraints): this is Silero **v5** — inputs `input
[B, samples]`, `state [2, B, 128]`, and `sr` as an int64 scalar; outputs `output [B, 1]` and
`stateN`. That is a single state tensor, not v4's separate `h` and `c`, and the implementation
below matches it.

Silero at 16 kHz requires exactly 512-sample frames. That is why `Framer` exists.

- [ ] **Step 1: Write the failing test**

`shared/src/jvmTest/kotlin/app/presence/onnx/SileroVadTest.kt`:

```kotlin
package app.presence.onnx

import app.presence.audio.SAMPLE_RATE_HZ
import java.nio.file.Path
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import org.junit.jupiter.api.Tag

/** Requires models/silero_vad.onnx. Run with ./gradlew :shared:integrationTest */
@Tag("model")
class SileroVadTest {

    private val modelPath = Path.of("../models/silero_vad.onnx")

    private fun withVad(block: (SileroVad) -> Unit) =
        SileroVad(OnnxModelSession(modelPath)).use(block)

    private fun silence() = FloatArray(512)

    private fun tone(frequencyHz: Double = 220.0) = FloatArray(512) {
        (0.3 * sin(2 * PI * frequencyHz * it / SAMPLE_RATE_HZ)).toFloat()
    }

    private fun noise(seed: Int) =
        Random(seed).let { rng -> FloatArray(512) { rng.nextFloat() * 0.02f - 0.01f } }

    /** Broadband noise at amplitude 0.5 — the one synthetic signal Silero reacts to. */
    private fun loudNoise(seed: Int) =
        Random(seed).let { rng -> FloatArray(512) { rng.nextFloat() - 0.5f } }

    @Test
    fun `reports the frame size the model requires`() {
        withVad { vad -> assertEquals(512, vad.frameSize) }
    }

    @Test
    fun `returns a probability in range for any input`() {
        withVad { vad ->
            listOf(silence(), tone(), noise(1)).forEach { frame ->
                val probability = vad.speechProbability(frame)
                assertTrue(probability in 0f..1f, "probability $probability outside [0,1]")
                assertTrue(probability.isFinite())
            }
        }
    }

    /**
     * Silero is not an energy detector — it responds to speech specifically. A pure tone has no
     * broadband excitation, no formant structure and no temporal dynamics, so the model rates it
     * no higher than silence. Measured on the real model: silence 5.42e-4, tone 5.26e-4 (the tone
     * marginally *lower*). Broadband noise is the only synthetic signal it reacts to — 1.46e-3 with
     * a fresh seed per frame as below (ratio ~2.7), or 8.22e-4 replaying one frame, which is why
     * the diagnostic's fixed-seed reading is lower than this test's.
     *
     * So this is a smoke test that the model is genuinely a function of its input rather than
     * returning a constant. Real speech discrimination is measured on actual recordings in
     * Task 17; it cannot be established with synthetic audio.
     */
    @Test
    fun `distinguishes broadband noise from digital silence`() {
        withVad { vad ->
            // Averaged over several frames: the recurrent state settles after the first few.
            val silenceScore = (1..10).map { vad.speechProbability(silence()) }.average()
            vad.reset()
            val noiseScore = (1..10).map { vad.speechProbability(loudNoise(it)) }.average()

            assertTrue(
                noiseScore > silenceScore * 1.5,
                "broadband noise ($noiseScore) should score clearly above silence ($silenceScore)",
            )
        }
    }

    /**
     * Records what the model actually does with synthetic signals, for calibration. Asserts only
     * what must hold regardless — an ordering assertion here would be wishful thinking, since
     * none of these inputs is speech.
     */
    @Test
    fun `reports synthetic signal probabilities for inspection`() {
        withVad { vad ->
            val readings = linkedMapOf<String, Double>()
            listOf(
                "silence" to { silence() },
                "tone-150Hz" to { tone(150.0) },
                "tone-220Hz" to { tone(220.0) },
                "quiet-noise" to { noise(1) },
                "loud-noise" to { loudNoise(1) },
            ).forEach { (label, make) ->
                vad.reset()
                readings[label] = (1..10).map { vad.speechProbability(make()) }.average()
            }
            println("Silero synthetic-signal probabilities: $readings")

            readings.forEach { (label, value) ->
                assertTrue(value in 0.0..1.0 && value.isFinite(), "$label produced $value")
            }
        }
    }

    @Test
    fun `carries recurrent state between frames`() {
        withVad { vad ->
            val first = vad.speechProbability(tone())
            val tenth = (1..9).map { vad.speechProbability(tone()) }.last()

            assertTrue(
                abs(first - tenth) > 1e-6f,
                "state should evolve; got $first then $tenth",
            )
        }
    }

    @Test
    fun `rejects a frame of the wrong size`() {
        withVad { vad ->
            val failure = assertFailsWith<IllegalArgumentException> {
                vad.speechProbability(FloatArray(256))
            }
            assertTrue(failure.message!!.contains("512"))
        }
    }

    @Test
    fun `reset returns the detector to its initial response`() {
        withVad { vad ->
            val first = vad.speechProbability(tone())
            repeat(20) { vad.speechProbability(tone()) }
            vad.reset()
            val afterReset = vad.speechProbability(tone())

            assertTrue(
                abs(first - afterReset) < 1e-4f,
                "expected $first after reset, got $afterReset",
            )
        }
    }
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `./gradlew :shared:integrationTest --tests "app.presence.onnx.SileroVadTest"`
Expected: FAIL — unresolved reference `SileroVad`.

- [ ] **Step 3: Write the implementation**

`shared/src/jvmMain/kotlin/app/presence/onnx/SileroVad.kt`:

```kotlin
package app.presence.onnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import app.presence.audio.SAMPLE_RATE_HZ
import app.presence.engine.VoiceActivityDetector

/**
 * Silero VAD v5 via ONNX Runtime.
 *
 * Verified signature: inputs `input [B, samples]`, `state [2, B, 128]`, `sr` (int64 scalar);
 * outputs `output [B, 1]`, `stateN`.
 *
 * The model is recurrent, so state carries between frames and [reset] matters between recordings.
 */
class SileroVad(
    private val session: OnnxModelSession,
    override val frameSize: Int = 512,
) : VoiceActivityDetector, AutoCloseable {

    private companion object {
        const val STATE_LAYERS = 2
        const val STATE_SIZE = 128
    }

    private val environment: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var state = FloatArray(STATE_LAYERS * STATE_SIZE)

    override fun speechProbability(frame: FloatArray): Float {
        require(frame.size == frameSize) {
            "Silero at $SAMPLE_RATE_HZ Hz needs exactly $frameSize samples, got ${frame.size} — use Framer"
        }

        val audioTensor = session.floatTensor(frame.copyOf(), longArrayOf(1, frameSize.toLong()))
        val stateTensor = session.floatTensor(
            state.copyOf(),
            longArrayOf(STATE_LAYERS.toLong(), 1, STATE_SIZE.toLong()),
        )
        val rateTensor = OnnxTensor.createTensor(environment, longArrayOf(SAMPLE_RATE_HZ.toLong()))

        return session.runTensors(
            mapOf("input" to audioTensor, "state" to stateTensor, "sr" to rateTensor),
        ) { result ->
            @Suppress("UNCHECKED_CAST")
            val probability = (result.get("output").get().value as Array<FloatArray>)[0][0]
            @Suppress("UNCHECKED_CAST")
            val nextState = result.get("stateN").get().value as Array<Array<FloatArray>>
            state = FloatArray(STATE_LAYERS * STATE_SIZE) { index ->
                nextState[index / STATE_SIZE][0][index % STATE_SIZE]
            }
            probability.coerceIn(0f, 1f)
        }
    }

    override fun reset() {
        state = FloatArray(STATE_LAYERS * STATE_SIZE)
    }

    override fun close() {
        session.close()
    }
}
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `./gradlew :shared:integrationTest --tests "app.presence.onnx.SileroVadTest"`
Expected: PASS, 7 tests.

If a cast fails, print the actual value's class — an ONNX output's `value` is a nested `Array`
whose depth equals the tensor rank, and `stateN`'s reported shape has dynamic dimensions.

- [ ] **Step 5: Commit**

```bash
git add shared/src/jvmMain/kotlin/app/presence/onnx/SileroVad.kt \
        shared/src/jvmTest/kotlin/app/presence/onnx/SileroVadTest.kt
git commit -m "feat: Silero VAD v5 adapter over ONNX Runtime"
```

---

### Task 13: Filterbank front end — FFT and Kaldi-compatible fbank

**Files:**
- Create: `shared/src/commonMain/kotlin/app/presence/audio/Fft.kt`
- Create: `shared/src/commonMain/kotlin/app/presence/audio/MelFilterBank.kt`
- Create: `shared/src/commonMain/kotlin/app/presence/audio/FbankExtractor.kt`
- Test: `shared/src/commonTest/kotlin/app/presence/audio/FftTest.kt`
- Test: `shared/src/commonTest/kotlin/app/presence/audio/FbankExtractorTest.kt`

**Interfaces:**
- Consumes: `SAMPLE_RATE_HZ` (Task 1).
- Produces: `object Fft` with `isPowerOfTwo(n: Int): Boolean`, `transform(re: DoubleArray, im: DoubleArray)`, `powerSpectrum(frame: DoubleArray): DoubleArray`; `class MelFilterBank(numBins, fftSize, sampleRate, lowFreqHz, highFreqHz)` with `apply(powerSpectrum: DoubleArray): DoubleArray` and companion `hzToMel(hz: Double): Double` / `melToHz(mel: Double): Double`; `class FbankExtractor(numMelBins: Int = 80, frameLengthMs: Int = 25, frameShiftMs: Int = 10)` with `extract(samples: FloatArray): Array<FloatArray>` returning `[frames][numMelBins]`. Task 14 depends on all of it.

**Why this task exists.** The speaker model's signature was verified against the real file: it is
`feats [B, T, 80] → embs [B, 256]`. It does **not** accept raw audio — it wants 80-bin log-mel
filterbank features. Every candidate model checked (WeSpeaker, 3D-Speaker CAM++, NeMo SpeakerNet,
NeMo TitaNet) takes features rather than a waveform, so this front end is unavoidable regardless of
which model ships. It lives in `commonMain` because Android and iOS will both need it.

**Getting the parameters exactly right matters more than it looks.** A subtly wrong filterbank does
not throw — it silently degrades accuracy, and you would spend Phase 0 blaming the model. These
values match what WeSpeaker uses to train, and must be reproduced exactly:

| Parameter | Value | Note |
|---|---|---|
| Mel bins | 80 | the model's input dimension |
| Frame length | 25 ms → 400 samples | |
| Frame shift | 10 ms → 160 samples | |
| FFT size | 512 | 400 rounded up to a power of two |
| Window | Hamming, `0.54 − 0.46·cos(2πi/(N−1))` | WeSpeaker uses hamming, **not** Kaldi's default povey |
| Waveform scaling | ×32768 | Kaldi operates on int16-range values; omitting this shifts every log energy |
| Remove DC offset | yes, per frame | subtract the frame mean before pre-emphasis |
| Pre-emphasis | 0.97, applied descending | `x[i] -= 0.97·x[i−1]` for `i` from end down to 1, then `x[0] -= 0.97·x[0]` |
| Dither | 0.0 | none at inference; dithering would make results non-reproducible |
| Mel scale | `1127·ln(1 + f/700)` | |
| Frequency range | 20 Hz to 8000 Hz | Kaldi's default low cutoff is 20 Hz |
| FFT bins used | 256 (`fftSize/2`) | Kaldi excludes the Nyquist bin |
| Log floor | 1.19209290e-7 | FLT_EPSILON; `ln(max(energy, floor))` |
| Normalization | subtract each bin's mean over time | mean only, no variance normalization |

Frame count with `snip_edges` semantics: `1 + (samples − 400) / 160`, integer division.

**The FFT is tested against a brute-force DFT**, which is a genuinely strong check — the naive
transform is obviously correct and independently written, so agreement to floating-point tolerance
means the fast one is right.

- [ ] **Step 1: Write the failing FFT test**

`shared/src/commonTest/kotlin/app/presence/audio/FftTest.kt`:

```kotlin
package app.presence.audio

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class FftTest {

    /** The obvious, slow, independently-written transform the fast one must agree with. */
    private fun bruteForceDft(re: DoubleArray, im: DoubleArray): Pair<DoubleArray, DoubleArray> {
        val n = re.size
        val outRe = DoubleArray(n)
        val outIm = DoubleArray(n)
        for (k in 0 until n) {
            var sumRe = 0.0
            var sumIm = 0.0
            for (t in 0 until n) {
                val angle = -2.0 * PI * k * t / n
                sumRe += re[t] * cos(angle) - im[t] * sin(angle)
                sumIm += re[t] * sin(angle) + im[t] * cos(angle)
            }
            outRe[k] = sumRe
            outIm[k] = sumIm
        }
        return outRe to outIm
    }

    @Test
    fun `recognizes powers of two`() {
        listOf(1, 2, 4, 512).forEach { assertTrue(Fft.isPowerOfTwo(it), "$it is a power of two") }
        listOf(0, 3, 400, -8).forEach { assertTrue(!Fft.isPowerOfTwo(it), "$it is not") }
    }

    @Test
    fun `agrees with a brute-force DFT on random input`() {
        val random = Random(42)
        val n = 64
        val re = DoubleArray(n) { random.nextDouble(-1.0, 1.0) }
        val im = DoubleArray(n) { random.nextDouble(-1.0, 1.0) }
        val (expectedRe, expectedIm) = bruteForceDft(re, im)

        val actualRe = re.copyOf()
        val actualIm = im.copyOf()
        Fft.transform(actualRe, actualIm)

        for (k in 0 until n) {
            assertTrue(abs(expectedRe[k] - actualRe[k]) < 1e-9, "real part at $k")
            assertTrue(abs(expectedIm[k] - actualIm[k]) < 1e-9, "imaginary part at $k")
        }
    }

    @Test
    fun `transforms a constant signal to a single DC spike`() {
        val n = 8
        val re = DoubleArray(n) { 1.0 }
        val im = DoubleArray(n)

        Fft.transform(re, im)

        assertTrue(abs(8.0 - re[0]) < 1e-9, "DC bin should hold the sum, got ${re[0]}")
        for (k in 1 until n) assertTrue(abs(re[k]) < 1e-9 && abs(im[k]) < 1e-9, "bin $k should be zero")
    }

    @Test
    fun `power spectrum peaks at the frequency of a pure tone`() {
        val n = 512
        val binOfInterest = 40
        val frame = DoubleArray(n) { cos(2.0 * PI * binOfInterest * it / n) }

        val power = Fft.powerSpectrum(frame)

        assertEquals(n / 2 + 1, power.size)
        val peak = power.indices.maxBy { power[it] }
        assertEquals(binOfInterest, peak)
    }

    @Test
    fun `power spectrum is never negative`() {
        val random = Random(7)
        val power = Fft.powerSpectrum(DoubleArray(256) { random.nextDouble(-1.0, 1.0) })

        assertTrue(power.all { it >= 0.0 })
    }

    @Test
    fun `rejects a non-power-of-two size`() {
        assertFailsWith<IllegalArgumentException> { Fft.transform(DoubleArray(3), DoubleArray(3)) }
    }

    @Test
    fun `rejects mismatched real and imaginary lengths`() {
        assertFailsWith<IllegalArgumentException> { Fft.transform(DoubleArray(4), DoubleArray(8)) }
    }
}
```

- [ ] **Step 2: Run it and verify it fails**

Run: `./gradlew :shared:jvmTest --tests "app.presence.audio.FftTest"`
Expected: FAIL — unresolved reference `Fft`.

- [ ] **Step 3: Write the FFT**

`shared/src/commonMain/kotlin/app/presence/audio/Fft.kt`:

```kotlin
package app.presence.audio

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/** Iterative in-place radix-2 Cooley-Tukey FFT. */
object Fft {

    fun isPowerOfTwo(n: Int): Boolean = n > 0 && (n and (n - 1)) == 0

    /** Transforms [re] and [im] in place. Both must be the same power-of-two length. */
    fun transform(re: DoubleArray, im: DoubleArray) {
        require(re.size == im.size) {
            "real and imaginary parts must be the same length: ${re.size} vs ${im.size}"
        }
        val n = re.size
        require(isPowerOfTwo(n)) { "FFT size must be a power of two, was $n" }

        // Bit-reversal permutation.
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j xor bit
            if (i < j) {
                val tempRe = re[i]; re[i] = re[j]; re[j] = tempRe
                val tempIm = im[i]; im[i] = im[j]; im[j] = tempIm
            }
        }

        var length = 2
        while (length <= n) {
            val angle = -2.0 * PI / length
            val stepRe = cos(angle)
            val stepIm = sin(angle)
            var block = 0
            while (block < n) {
                var twiddleRe = 1.0
                var twiddleIm = 0.0
                val half = length / 2
                for (k in 0 until half) {
                    val upperRe = re[block + k]
                    val upperIm = im[block + k]
                    val lowerRe = re[block + k + half] * twiddleRe - im[block + k + half] * twiddleIm
                    val lowerIm = re[block + k + half] * twiddleIm + im[block + k + half] * twiddleRe
                    re[block + k] = upperRe + lowerRe
                    im[block + k] = upperIm + lowerIm
                    re[block + k + half] = upperRe - lowerRe
                    im[block + k + half] = upperIm - lowerIm
                    val nextTwiddleRe = twiddleRe * stepRe - twiddleIm * stepIm
                    twiddleIm = twiddleRe * stepIm + twiddleIm * stepRe
                    twiddleRe = nextTwiddleRe
                }
                block += length
            }
            length = length shl 1
        }
    }

    /** |X[k]|² for k in 0..n/2 inclusive. [frame] is not modified. */
    fun powerSpectrum(frame: DoubleArray): DoubleArray {
        val re = frame.copyOf()
        val im = DoubleArray(frame.size)
        transform(re, im)
        return DoubleArray(frame.size / 2 + 1) { k -> re[k] * re[k] + im[k] * im[k] }
    }
}
```

- [ ] **Step 4: Run it and verify it passes**

Run: `./gradlew :shared:jvmTest --tests "app.presence.audio.FftTest"`
Expected: PASS, 7 tests.

- [ ] **Step 5: Write the failing fbank test**

`shared/src/commonTest/kotlin/app/presence/audio/FbankExtractorTest.kt`:

```kotlin
package app.presence.audio

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class FbankExtractorTest {

    private val extractor = FbankExtractor()

    private fun tone(frequencyHz: Double, samples: Int) = FloatArray(samples) {
        (0.5 * sin(2.0 * PI * frequencyHz * it / SAMPLE_RATE_HZ)).toFloat()
    }

    // --- mel scale ---

    @Test
    fun `mel conversion round trips`() {
        listOf(20.0, 440.0, 1_000.0, 8_000.0).forEach { hz ->
            val roundTripped = MelFilterBank.melToHz(MelFilterBank.hzToMel(hz))
            assertTrue(abs(hz - roundTripped) < 1e-6, "expected $hz, got $roundTripped")
        }
    }

    @Test
    fun `mel scale is monotonic and compresses high frequencies`() {
        val lowSpan = MelFilterBank.hzToMel(1_100.0) - MelFilterBank.hzToMel(100.0)
        val highSpan = MelFilterBank.hzToMel(7_000.0) - MelFilterBank.hzToMel(6_000.0)

        assertTrue(MelFilterBank.hzToMel(100.0) < MelFilterBank.hzToMel(1_000.0))
        assertTrue(lowSpan > highSpan, "a low-frequency kHz should span more mels than a high one")
    }

    @Test
    fun `every mel filter has some non-zero weight`() {
        val bank = MelFilterBank()
        val allOnes = DoubleArray(512 / 2 + 1) { 1.0 }

        val energies = bank.apply(allOnes)

        assertEquals(80, energies.size)
        energies.forEachIndexed { bin, energy ->
            assertTrue(energy > 0.0, "mel bin $bin has no weight at all")
        }
    }

    @Test
    fun `mel filters are ordered so a higher tone excites a higher bin`() {
        val bank = MelFilterBank()
        val lowPeak = bank.apply(spikeAt(hz = 300.0)).let { it.indices.maxBy { i -> it[i] } }
        val highPeak = bank.apply(spikeAt(hz = 3_000.0)).let { it.indices.maxBy { i -> it[i] } }

        assertTrue(lowPeak < highPeak, "300 Hz peaked at $lowPeak, 3000 Hz at $highPeak")
    }

    private fun spikeAt(hz: Double): DoubleArray {
        val bin = (hz * 512 / SAMPLE_RATE_HZ).toInt()
        return DoubleArray(512 / 2 + 1) { if (it == bin) 1.0 else 0.0 }
    }

    // --- extraction ---

    @Test
    fun `produces 80 bins per frame`() {
        val features = extractor.extract(tone(440.0, SAMPLE_RATE_HZ))

        assertTrue(features.isNotEmpty())
        features.forEach { assertEquals(80, it.size) }
    }

    @Test
    fun `frame count follows the snip-edges formula`() {
        // 16000 samples: 1 + (16000 - 400) / 160 = 98 frames.
        assertEquals(98, extractor.extract(tone(440.0, 16_000)).size)
        // Exactly one frame's worth of samples yields exactly one frame.
        assertEquals(1, extractor.extract(tone(440.0, 400)).size)
        // One shift more yields one frame more.
        assertEquals(2, extractor.extract(tone(440.0, 560)).size)
    }

    @Test
    fun `rejects audio shorter than one frame`() {
        assertFailsWith<IllegalArgumentException> { extractor.extract(tone(440.0, 399)) }
    }

    @Test
    fun `is deterministic`() {
        val samples = tone(440.0, 8_000)

        val first = extractor.extract(samples)
        val second = extractor.extract(samples.copyOf())

        first.forEachIndexed { frame, row ->
            row.forEachIndexed { bin, value ->
                assertEquals(value, second[frame][bin], "frame $frame bin $bin differs")
            }
        }
    }

    @Test
    fun `produces no NaN or infinity, even for digital silence`() {
        val features = extractor.extract(FloatArray(8_000))

        features.forEach { row ->
            row.forEach { assertTrue(it.isFinite(), "non-finite value $it in silence") }
        }
    }

    @Test
    fun `mean normalization leaves each bin centred on zero`() {
        val features = extractor.extract(tone(440.0, 16_000))

        for (bin in 0 until 80) {
            var sum = 0.0
            for (frame in features) sum += frame[bin]
            val mean = sum / features.size
            assertTrue(abs(mean) < 1e-3, "bin $bin has mean $mean after normalization")
        }
    }

    /**
     * Guards the log floor against swallowing pathologically quiet input, and in doing so pins the
     * ×32768 int16 scaling — the one constant no other test can reach.
     *
     * Mean normalization cancels any uniform gain exactly: scaling the waveform shifts every log
     * energy by the same constant, which the per-bin mean then subtracts. So the scaling is
     * observable only where the floor clips. This amplitude (2e-7, well below one 16-bit LSB) is
     * chosen to sit that far down deliberately; at realistic levels the scaling provably has no
     * effect on the output, which is why nothing else here can pin it. Noise rather than a tone,
     * because a stationary signal normalizes to zero regardless.
     *
     * Verified by breaking it: 3819 distinct values with the scaling, exactly 1 without (every bin
     * clips to the floor and normalizes to 0.0). An earlier attempt at amplitude 2e-5 did NOT
     * discriminate — 3840 against 297 — because each mel triangle sums over many FFT bins, lifting
     * unscaled energy back to roughly the floor and clipping only partially.
     */
    @Test
    fun `quiet audio still carries information rather than clipping to the log floor`() {
        val quiet = Random(11).let { rng -> FloatArray(8_000) { (rng.nextFloat() - 0.5f) * 2e-7f } }

        val features = extractor.extract(quiet)

        val distinctValues = features.flatMap { row -> row.toList() }.distinct()
        assertTrue(
            distinctValues.size > 1,
            "every value collapsed to ${distinctValues.firstOrNull()} — the log floor swallowed the signal",
        )
    }

    /**
     * Mean normalization is per-utterance, so a stationary tone yields bit-identical frames whose
     * every bin normalizes to exactly zero — its "peak" would be an arbitrary tie on bin 0. This
     * signal changes pitch halfway through, so early frames sit above the utterance mean in the
     * low bins and late frames sit above it in the high bins. That is the property that matters,
     * and it exercises the whole extractor including normalization.
     *
     * Measured on this filterbank: 300 Hz peaks at bin 10, 3000 Hz at bin 52 — the same bins the
     * raw pre-normalization spectra peak at.
     */
    @Test
    fun `energy moves to higher mel bins when the pitch rises mid-utterance`() {
        val lowThenHigh = tone(300.0, 8_000) + tone(3_000.0, 8_000)

        val features = extractor.extract(lowThenHigh)

        val early = features[features.size / 4]
        val late = features[3 * features.size / 4]
        val earlyPeak = early.indices.maxBy { early[it] }
        val latePeak = late.indices.maxBy { late[it] }

        assertTrue(
            earlyPeak < latePeak,
            "early frame peaked at bin $earlyPeak, late frame at bin $latePeak",
        )
    }
}
```

- [ ] **Step 6: Run it and verify it fails**

Run: `./gradlew :shared:jvmTest --tests "app.presence.audio.FbankExtractorTest"`
Expected: FAIL — unresolved reference `FbankExtractor`.

- [ ] **Step 7: Write the mel filterbank**

`shared/src/commonMain/kotlin/app/presence/audio/MelFilterBank.kt`:

```kotlin
package app.presence.audio

import kotlin.math.exp
import kotlin.math.ln

/**
 * Triangular mel filters, matching Kaldi's `MelBanks` so features line up with what the speaker
 * model was trained on.
 *
 * Kaldi uses `fftSize / 2` spectrum bins and excludes the Nyquist bin; reproducing that exactly
 * matters, because an off-by-one here shifts every filter slightly and quietly costs accuracy.
 */
class MelFilterBank(
    val numBins: Int = 80,
    val fftSize: Int = 512,
    val sampleRate: Int = SAMPLE_RATE_HZ,
    val lowFreqHz: Double = 20.0,
    val highFreqHz: Double = sampleRate / 2.0,
) {
    private val numFftBins = fftSize / 2
    private val weights: Array<DoubleArray>

    init {
        require(numBins > 0) { "numBins must be positive" }
        require(Fft.isPowerOfTwo(fftSize)) { "fftSize must be a power of two" }
        require(lowFreqHz < highFreqHz) { "lowFreqHz must be below highFreqHz" }

        val melLow = hzToMel(lowFreqHz)
        val melHigh = hzToMel(highFreqHz)
        val melDelta = (melHigh - melLow) / (numBins + 1)
        val hzPerBin = sampleRate.toDouble() / fftSize

        weights = Array(numBins) { bin ->
            val left = melLow + bin * melDelta
            val center = left + melDelta
            val right = center + melDelta
            DoubleArray(numFftBins) { k ->
                val mel = hzToMel(k * hzPerBin)
                when {
                    mel > left && mel <= center -> (mel - left) / (center - left)
                    mel > center && mel < right -> (right - mel) / (right - center)
                    else -> 0.0
                }
            }
        }
    }

    /** Sums [powerSpectrum] under each triangular filter. */
    fun apply(powerSpectrum: DoubleArray): DoubleArray {
        require(powerSpectrum.size >= numFftBins) {
            "power spectrum has ${powerSpectrum.size} bins, need at least $numFftBins"
        }
        return DoubleArray(numBins) { bin ->
            val filter = weights[bin]
            var energy = 0.0
            for (k in 0 until numFftBins) energy += filter[k] * powerSpectrum[k]
            energy
        }
    }

    companion object {
        fun hzToMel(hz: Double): Double = 1127.0 * ln(1.0 + hz / 700.0)
        fun melToHz(mel: Double): Double = 700.0 * (exp(mel / 1127.0) - 1.0)
    }
}
```

- [ ] **Step 8: Write the extractor**

`shared/src/commonMain/kotlin/app/presence/audio/FbankExtractor.kt`:

```kotlin
package app.presence.audio

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.max

/**
 * 80-bin log-mel filterbank features, matching what the speaker model expects
 * (`feats [B, T, 80]`).
 *
 * Every constant here mirrors the configuration the model was trained with. A mismatch does not
 * throw — it silently degrades accuracy, which is the worst possible failure mode for this project
 * because it looks like a bad model rather than a bad front end.
 */
class FbankExtractor(
    private val numMelBins: Int = 80,
    frameLengthMs: Int = 25,
    frameShiftMs: Int = 10,
    private val sampleRate: Int = SAMPLE_RATE_HZ,
    private val preEmphasis: Double = 0.97,
) {
    private companion object {
        /** FLT_EPSILON, matching Kaldi's log floor. */
        const val LOG_FLOOR = 1.19209290e-7

        /** Kaldi operates on int16-range values; omitting this shifts every log energy. */
        const val INT16_SCALE = 32768.0
    }

    private val frameLength = sampleRate * frameLengthMs / 1_000
    private val frameShift = sampleRate * frameShiftMs / 1_000
    private val fftSize = generateSequence(1) { it * 2 }.first { it >= frameLength }

    /** Hamming. WeSpeaker trains with this, not Kaldi's default povey window. */
    private val window = DoubleArray(frameLength) {
        0.54 - 0.46 * cos(2.0 * PI * it / (frameLength - 1))
    }

    private val melBank = MelFilterBank(numMelBins, fftSize, sampleRate)

    /** Returns `[frames][numMelBins]`, mean-normalized per bin across time. */
    fun extract(samples: FloatArray): Array<FloatArray> {
        require(samples.size >= frameLength) {
            "need at least $frameLength samples for one frame, got ${samples.size}"
        }

        val frameCount = 1 + (samples.size - frameLength) / frameShift
        val features = Array(frameCount) { FloatArray(numMelBins) }
        val buffer = DoubleArray(fftSize)

        for (frame in 0 until frameCount) {
            val offset = frame * frameShift

            for (i in 0 until frameLength) buffer[i] = samples[offset + i] * INT16_SCALE
            for (i in frameLength until fftSize) buffer[i] = 0.0

            // Remove the frame's DC offset.
            var sum = 0.0
            for (i in 0 until frameLength) sum += buffer[i]
            val mean = sum / frameLength
            for (i in 0 until frameLength) buffer[i] -= mean

            // Pre-emphasis, applied descending so each sample sees its untouched predecessor.
            for (i in frameLength - 1 downTo 1) buffer[i] -= preEmphasis * buffer[i - 1]
            buffer[0] -= preEmphasis * buffer[0]

            for (i in 0 until frameLength) buffer[i] *= window[i]

            val energies = melBank.apply(Fft.powerSpectrum(buffer))
            for (bin in 0 until numMelBins) {
                features[frame][bin] = ln(max(energies[bin], LOG_FLOOR)).toFloat()
            }
        }

        subtractMeanOverTime(features)
        return features
    }

    /** Mean normalization per bin. Mean only — the model was trained without variance norm. */
    private fun subtractMeanOverTime(features: Array<FloatArray>) {
        for (bin in 0 until numMelBins) {
            var sum = 0.0
            for (frame in features) sum += frame[bin]
            val mean = (sum / features.size).toFloat()
            for (frame in features) frame[bin] -= mean
        }
    }
}
```

- [ ] **Step 9: Run the tests and verify they pass**

Run: `./gradlew :shared:jvmTest --tests "app.presence.audio.*"`
Expected: PASS — FftTest 7, FbankExtractorTest 12, plus the earlier audio tests.

- [ ] **Step 10: Commit**

```bash
git add shared/src/commonMain/kotlin/app/presence/audio/Fft.kt \
        shared/src/commonMain/kotlin/app/presence/audio/MelFilterBank.kt \
        shared/src/commonMain/kotlin/app/presence/audio/FbankExtractor.kt \
        shared/src/commonTest/kotlin/app/presence/audio/FftTest.kt \
        shared/src/commonTest/kotlin/app/presence/audio/FbankExtractorTest.kt
git commit -m "feat: Kaldi-compatible 80-bin filterbank front end"
```

---

### Task 14: OnnxSpeakerEmbedder — the real embedding model

**Files:**
- Create: `shared/src/jvmMain/kotlin/app/presence/onnx/OnnxSpeakerEmbedder.kt`
- Test: `shared/src/jvmTest/kotlin/app/presence/onnx/OnnxSpeakerEmbedderTest.kt`

**Interfaces:**
- Consumes: `Embedding`, `SpeakerEmbedder`, `cosine` (Tasks 3, 9); `FbankExtractor` (Task 13); `OnnxModelSession` (Task 11).
- Produces: `class OnnxSpeakerEmbedder(session: OnnxModelSession, fbank: FbankExtractor = FbankExtractor(), inputName: String = "feats", outputName: String = "embs") : SpeakerEmbedder, AutoCloseable` with `dimension` discovered on first use. Task 17 constructs it.

**The signature is verified, not assumed.** The shipped model was inspected directly:

```
feats: [B, T, 80]  ->  embs: [B, 256]
```

So the embedder's job is: samples → `FbankExtractor` → `[1, T, 80]` tensor → model → 256 floats →
`Embedding.ofNormalized`. Tensor names default to the verified `feats` and `embs` but stay
constructor parameters, because a fine-tuned replacement model (spec §4.1, L3) could export
different names.

**On testing this honestly.** These tests prove the *plumbing* — that features reach the model in
the right shape and a unit-length 256-dim vector comes back. They cannot prove the model
discriminates children's voices, because synthetic tones are far outside its training distribution
and real separation is what Task 17 measures on actual corpora. The synthetic-voice check therefore
**reports** its similarity numbers for human inspection while asserting only what must be true
regardless: finite values in `[-1, 1]`. An assertion about the *ordering* of synthetic-tone
similarities would be a coin flip dressed up as a test.

- [ ] **Step 1: Write the failing test**

`shared/src/jvmTest/kotlin/app/presence/onnx/OnnxSpeakerEmbedderTest.kt`:

```kotlin
package app.presence.onnx

import app.presence.audio.SAMPLE_RATE_HZ
import app.presence.engine.cosine
import java.nio.file.Path
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import org.junit.jupiter.api.Tag

/** Requires models/speaker_embedding.onnx. Run with ./gradlew :shared:integrationTest */
@Tag("model")
class OnnxSpeakerEmbedderTest {

    private val modelPath = Path.of("../models/speaker_embedding.onnx")

    private fun withEmbedder(block: (OnnxSpeakerEmbedder) -> Unit) =
        OnnxSpeakerEmbedder(OnnxModelSession(modelPath)).use(block)

    /** A crude voice-like signal: a fundamental plus two harmonics. */
    private fun voice(fundamentalHz: Double, seconds: Double = 2.0): FloatArray {
        val count = (seconds * SAMPLE_RATE_HZ).toInt()
        return FloatArray(count) { i ->
            val t = i / SAMPLE_RATE_HZ.toDouble()
            (0.4 * sin(2 * PI * fundamentalHz * t) +
                0.2 * sin(2 * PI * fundamentalHz * 2 * t) +
                0.1 * sin(2 * PI * fundamentalHz * 3 * t)).toFloat()
        }
    }

    @Test
    fun `produces a unit-length embedding`() {
        withEmbedder { embedder ->
            val embedding = embedder.embed(voice(180.0))

            var sumOfSquares = 0.0
            for (v in embedding.values) sumOfSquares += v.toDouble() * v
            assertTrue(abs(1.0 - sqrt(sumOfSquares)) < 1e-4, "embedding was not unit length")
        }
    }

    @Test
    fun `produces the 256 dimensions the model declares`() {
        withEmbedder { embedder ->
            assertEquals(256, embedder.embed(voice(180.0)).dimension)
            assertEquals(256, embedder.dimension)
        }
    }

    @Test
    fun `every component is finite`() {
        withEmbedder { embedder ->
            embedder.embed(voice(180.0)).values.forEach {
                assertTrue(it.isFinite(), "non-finite component $it")
            }
        }
    }

    @Test
    fun `is deterministic for identical input`() {
        withEmbedder { embedder ->
            val samples = voice(180.0)

            val similarity = cosine(embedder.embed(samples), embedder.embed(samples.copyOf()))

            assertTrue(abs(1f - similarity) < 1e-4f, "expected 1.0, got $similarity")
        }
    }

    @Test
    fun `handles utterances of different lengths`() {
        withEmbedder { embedder ->
            // T varies with duration; the model takes a dynamic time axis.
            listOf(1.0, 2.0, 5.0).forEach { seconds ->
                assertEquals(256, embedder.embed(voice(180.0, seconds)).dimension)
            }
        }
    }

    @Test
    fun `reports synthetic voice separation for inspection`() {
        withEmbedder { embedder ->
            val speakerAFirst = embedder.embed(voice(120.0, seconds = 2.0))
            val speakerASecond = embedder.embed(voice(120.0, seconds = 3.0))
            val speakerB = embedder.embed(voice(260.0, seconds = 2.0))

            val sameSpeaker = cosine(speakerAFirst, speakerASecond)
            val differentSpeaker = cosine(speakerAFirst, speakerB)
            println("synthetic same-speaker=$sameSpeaker different-speaker=$differentSpeaker")

            // Only what must hold regardless of how the model treats synthetic tones. Real
            // separation is measured on actual corpora in Task 17 — asserting an ordering here
            // would be a coin flip, because these tones are far outside the training distribution.
            assertTrue(sameSpeaker.isFinite() && sameSpeaker in -1f..1f, "got $sameSpeaker")
            assertTrue(differentSpeaker.isFinite() && differentSpeaker in -1f..1f, "got $differentSpeaker")
        }
    }

    @Test
    fun `rejects audio too short to embed`() {
        withEmbedder { embedder ->
            assertFailsWith<IllegalArgumentException> { embedder.embed(FloatArray(100)) }
        }
    }
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `./gradlew :shared:integrationTest --tests "app.presence.onnx.OnnxSpeakerEmbedderTest"`
Expected: FAIL — unresolved reference `OnnxSpeakerEmbedder`.

- [ ] **Step 3: Write the implementation**

`shared/src/jvmMain/kotlin/app/presence/onnx/OnnxSpeakerEmbedder.kt`:

```kotlin
package app.presence.onnx

import app.presence.audio.FbankExtractor
import app.presence.audio.SAMPLE_RATE_HZ
import app.presence.audio.millisToSamples
import app.presence.engine.Embedding
import app.presence.engine.SpeakerEmbedder

/**
 * A speaker-embedding model via ONNX Runtime.
 *
 * The shipped model's verified signature is `feats [B, T, 80] -> embs [B, 256]`: it consumes
 * log-mel filterbank features, not raw audio, so [FbankExtractor] runs first. Tensor names stay
 * constructor parameters because a fine-tuned replacement (spec §4.1, L3) may export different
 * ones.
 */
class OnnxSpeakerEmbedder(
    private val session: OnnxModelSession,
    private val fbank: FbankExtractor = FbankExtractor(),
    private val inputName: String = "feats",
    private val outputName: String = "embs",
) : SpeakerEmbedder, AutoCloseable {

    private companion object {
        /**
         * Below this the embedding is dominated by noise.
         *
         * Independent of `DetectionEngine.minEmbedDurationMs` (800 ms), which is stricter. This is
         * the last line of defence for a caller that bypasses the engine — enrollment, for one.
         */
        const val MIN_DURATION_MS = 400L
    }

    private var discoveredDimension: Int = 0

    override val dimension: Int
        get() {
            check(discoveredDimension > 0) { "dimension is unknown until embed() has been called once" }
            return discoveredDimension
        }

    override fun embed(samples: FloatArray): Embedding {
        val minimumSamples = millisToSamples(MIN_DURATION_MS)
        require(samples.size >= minimumSamples) {
            "need at least $MIN_DURATION_MS ms ($minimumSamples samples) at $SAMPLE_RATE_HZ Hz, " +
                "got ${samples.size}"
        }

        val features = fbank.extract(samples)
        val frames = features.size
        val bins = features[0].size

        // Flatten [T][80] into the [1, T, 80] tensor the model declares.
        val flat = FloatArray(frames * bins)
        for (frame in 0 until frames) {
            features[frame].copyInto(flat, frame * bins)
        }
        val tensor = session.floatTensor(flat, longArrayOf(1, frames.toLong(), bins.toLong()))

        return session.runTensors(mapOf(inputName to tensor)) { result ->
            @Suppress("UNCHECKED_CAST")
            val raw = (result.get(outputName).get().value as Array<FloatArray>)[0]
            discoveredDimension = raw.size
            // Normalizing here means every downstream cosine is a plain dot product.
            Embedding.ofNormalized(raw)
        }
    }

    override fun close() {
        session.close()
    }
}
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `./gradlew :shared:integrationTest --tests "app.presence.onnx.OnnxSpeakerEmbedderTest"`
Expected: PASS, 7 tests. The separation test prints its similarity numbers — record them in your
report, they are the first real signal about the model.

If ONNX Runtime rejects the input, its exception lists the names and shapes it expected; compare
against `OnnxModelSession.describeSignature()` and adjust the constructor parameters rather than
guessing.

- [ ] **Step 5: Commit**

```bash
git add shared/src/jvmMain/kotlin/app/presence/onnx/OnnxSpeakerEmbedder.kt \
        shared/src/jvmTest/kotlin/app/presence/onnx/OnnxSpeakerEmbedderTest.kt
git commit -m "feat: ONNX speaker embedding adapter with filterbank front end"
```

---

### Task 15: Manifest — the corpus metadata format

**Files:**
- Create: `harness/src/main/kotlin/app/presence/harness/Csv.kt`
- Create: `harness/src/main/kotlin/app/presence/harness/Manifest.kt`
- Test: `harness/src/test/kotlin/app/presence/harness/ManifestTest.kt`

**Interfaces:**
- Consumes: `Language`, `VoiceProfile.Role` (Task 4).
- Produces: `object Csv` with `parse(text: String): List<Map<String, String>>`; `enum class RecordingRole { TARGET_CHILD, SIBLING, PARENT, OTHER_HOUSEHOLD, MEDIA, STRANGER, ROOM_TONE }` with `isHouseholdMember: Boolean` and `toProfileRole(): VoiceProfile.Role`; `enum class ContentType { SPONTANEOUS, READ, CALLING, ROOM_TONE }`; `data class Recording(...)`; `data class CallEvent(recordingPath: String, startMs: Long, targetSpeakerId: String)`; `object Manifest` with `readRecordings(path: Path): List<Recording>` and `readEvents(path: Path): List<CallEvent>`. Tasks 16 and 17 consume these.

This is the executable form of spec §9.1's labelling schema. Two files: `recordings.csv` describes every audio file, `events.csv` marks where in a recording the child actually called for the parent.

The CSV reader is deliberately strict — no quoting, no embedded commas — and says exactly which line and column is wrong. Whoever collects this data will hand-edit it in a spreadsheet, and a silent mis-parse would corrupt results in a way nobody notices until the conclusions are already drawn.

- [ ] **Step 1: Write the failing test**

`harness/src/test/kotlin/app/presence/harness/ManifestTest.kt`:

```kotlin
package app.presence.harness

import app.presence.domain.Language
import app.presence.domain.VoiceProfile
import java.nio.file.Files
import kotlin.io.path.writeText
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

class ManifestTest {

    private fun tempFile(name: String, content: String) =
        Files.createTempDirectory("manifest-test").resolve(name).also { it.writeText(content) }

    private val recordingsHeader =
        "path,household_id,speaker_id,role,language,code_switched,content_type," +
            "distance_m,placement,background,room,date"

    // --- Csv ---

    @Test
    fun `parses a header and rows into maps`() {
        val rows = Csv.parse("a,b\n1,2\n3,4\n")

        assertEquals(2, rows.size)
        assertEquals("1", rows[0]["a"])
        assertEquals("4", rows[1]["b"])
    }

    @Test
    fun `ignores blank lines and trims whitespace`() {
        val rows = Csv.parse("a, b \n\n 1 ,2\n\n")

        assertEquals(1, rows.size)
        assertEquals("1", rows[0]["a"])
        assertEquals("2", rows[0]["b"])
    }

    @Test
    fun `rejects a row with the wrong column count, naming the line`() {
        val failure = assertFailsWith<IllegalArgumentException> { Csv.parse("a,b\n1,2,3\n") }

        assertTrue(failure.message!!.contains("line 2"), failure.message!!)
        assertTrue(failure.message!!.contains("3"), failure.message!!)
    }

    @Test
    fun `rejects quoting because embedded commas are unsupported`() {
        val failure = assertFailsWith<IllegalArgumentException> { Csv.parse("a,b\n\"x,y\",2\n") }

        assertTrue(failure.message!!.contains("quote"), failure.message!!)
    }

    @Test
    fun `rejects an empty document`() {
        assertFailsWith<IllegalArgumentException> { Csv.parse("") }
    }

    @Test
    fun `rejects duplicate header columns rather than silently losing one`() {
        val failure = assertFailsWith<IllegalArgumentException> { Csv.parse("a,b,a\n1,2,3\n") }

        assertTrue(failure.message!!.contains("duplicate"), failure.message!!)
        assertTrue(failure.message!!.contains("a"))
    }

    // --- Recordings ---

    @Test
    fun `reads a target child recording`() {
        val file = tempFile(
            "recordings.csv",
            "$recordingsHeader\n" +
                "audio/maya-01.wav,h1,maya,target_child,hebrew,false,calling,1.0,table,quiet,kitchen,2026-08-01\n",
        )

        val recordings = Manifest.readRecordings(file)

        assertEquals(1, recordings.size)
        val recording = recordings[0]
        assertEquals("maya", recording.speakerId)
        assertEquals(RecordingRole.TARGET_CHILD, recording.role)
        assertEquals(Language.HEBREW, recording.language)
        assertEquals(ContentType.CALLING, recording.contentType)
        assertEquals(1.0, recording.distanceMeters)
        assertTrue(!recording.codeSwitched)
    }

    @Test
    fun `reads code-switched content`() {
        val file = tempFile(
            "recordings.csv",
            "$recordingsHeader\n" +
                "a.wav,h1,maya,target_child,hebrew,true,spontaneous,1.0,table,quiet,kitchen,2026-08-01\n",
        )

        assertTrue(Manifest.readRecordings(file)[0].codeSwitched)
    }

    @Test
    fun `reads room tone with no language`() {
        val file = tempFile(
            "recordings.csv",
            "$recordingsHeader\n" +
                "tone.wav,h1,room,room_tone,,false,room_tone,,table,quiet,kitchen,2026-08-01\n",
        )

        val recording = Manifest.readRecordings(file)[0]

        assertNull(recording.language)
        assertNull(recording.distanceMeters)
        assertEquals(RecordingRole.ROOM_TONE, recording.role)
    }

    @Test
    fun `maps household roles to profile roles`() {
        assertEquals(VoiceProfile.Role.TARGET_CHILD, RecordingRole.TARGET_CHILD.toProfileRole())
        assertEquals(VoiceProfile.Role.SIBLING, RecordingRole.SIBLING.toProfileRole())
        assertEquals(VoiceProfile.Role.PARENT, RecordingRole.PARENT.toProfileRole())
        assertEquals(VoiceProfile.Role.OTHER, RecordingRole.OTHER_HOUSEHOLD.toProfileRole())
    }

    @Test
    fun `only household members are enrollable`() {
        assertTrue(RecordingRole.TARGET_CHILD.isHouseholdMember)
        assertTrue(RecordingRole.SIBLING.isHouseholdMember)
        assertTrue(RecordingRole.PARENT.isHouseholdMember)
        assertTrue(!RecordingRole.MEDIA.isHouseholdMember)
        assertTrue(!RecordingRole.STRANGER.isHouseholdMember)
        assertTrue(!RecordingRole.ROOM_TONE.isHouseholdMember)
    }

    @Test
    fun `rejects an unknown role, listing the valid ones`() {
        val file = tempFile(
            "recordings.csv",
            "$recordingsHeader\n" +
                "a.wav,h1,x,grandparent,hebrew,false,spontaneous,1.0,table,quiet,kitchen,2026-08-01\n",
        )

        val failure = assertFailsWith<IllegalArgumentException> { Manifest.readRecordings(file) }

        assertTrue(failure.message!!.contains("grandparent"))
        assertTrue(failure.message!!.contains("target_child"), "should list valid roles")
    }

    @Test
    fun `rejects a missing required column`() {
        val file = tempFile("recordings.csv", "path,household_id\na.wav,h1\n")

        val failure = assertFailsWith<IllegalArgumentException> { Manifest.readRecordings(file) }

        assertTrue(failure.message!!.contains("speaker_id"))
    }

    @Test
    fun `rejects a manifest listing the same audio path twice`() {
        val row = "dup.wav,h1,maya,target_child,hebrew,false,spontaneous,1.0,table,quiet,kitchen,2026-08-01"
        val file = tempFile("recordings.csv", "$recordingsHeader\n$row\n$row\n")

        val failure = assertFailsWith<IllegalArgumentException> { Manifest.readRecordings(file) }

        assertTrue(failure.message!!.contains("dup.wav"), failure.message!!)
    }

    @Test
    fun `rejects a manifest with a header but no rows`() {
        val file = tempFile("recordings.csv", "$recordingsHeader\n")

        val failure = assertFailsWith<IllegalArgumentException> { Manifest.readRecordings(file) }

        assertTrue(failure.message!!.contains("no data rows"), failure.message!!)
    }

    @Test
    fun `a malformed number names the file and line`() {
        val file = tempFile(
            "recordings.csv",
            "$recordingsHeader\n" +
                "a.wav,h1,maya,target_child,hebrew,false,spontaneous,abc,table,quiet,kitchen,2026-08-01\n",
        )

        val failure = assertFailsWith<IllegalArgumentException> { Manifest.readRecordings(file) }

        assertTrue(failure.message!!.contains("recordings.csv"), failure.message!!)
        assertTrue(failure.message!!.contains("line 2"), failure.message!!)
    }

    // --- Events ---

    @Test
    fun `reads call events`() {
        val file = tempFile(
            "events.csv",
            "recording_path,start_ms,target_speaker_id\naudio/maya-01.wav,4200,maya\n",
        )

        val events = Manifest.readEvents(file)

        assertEquals(1, events.size)
        assertEquals("audio/maya-01.wav", events[0].recordingPath)
        assertEquals(4200L, events[0].startMs)
        assertEquals("maya", events[0].targetSpeakerId)
    }

    @Test
    fun `rejects an event with a blank speaker id`() {
        val file = tempFile("events.csv", "recording_path,start_ms,target_speaker_id\na.wav,100,\n")

        assertFailsWith<IllegalArgumentException> { Manifest.readEvents(file) }
    }

    @Test
    fun `rejects a negative event start`() {
        val file = tempFile(
            "events.csv",
            "recording_path,start_ms,target_speaker_id\na.wav,-1,maya\n",
        )

        assertFailsWith<IllegalArgumentException> { Manifest.readEvents(file) }
    }
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `./gradlew :harness:test --tests "app.presence.harness.ManifestTest"`
Expected: FAIL — unresolved reference `Csv`.

- [ ] **Step 3: Write the CSV reader**

`harness/src/main/kotlin/app/presence/harness/Csv.kt`:

```kotlin
package app.presence.harness

/**
 * A deliberately strict CSV reader: no quoting, no embedded commas, no escapes.
 *
 * Spec §9.1's fields are all simple tokens, and hand-rolled quote handling is a classic source
 * of silent corruption. This reader refuses anything ambiguous and names the offending line, so
 * a spreadsheet export that sneaks in a comma fails loudly instead of skewing the results.
 */
object Csv {

    fun parse(text: String): List<Map<String, String>> =
        parseWithLineNumbers(text).map { it.second }

    /**
     * Same as [parse], but pairs each row with its 1-based line number in the source text, so a
     * caller can name the line a human would find in their editor. `withIndex()` runs before the
     * blank-line filter precisely so those numbers stay true to the file.
     */
    fun parseWithLineNumbers(text: String): List<Pair<Int, Map<String, String>>> {
        val lines = text.lineSequence()
            .withIndex()
            .filter { (_, line) -> line.isNotBlank() }
            .toList()
        require(lines.isNotEmpty()) { "CSV is empty: expected a header row" }

        val headerLineNumber = lines.first().index + 1
        val header = splitRow(lines.first().value, headerLineNumber)

        // Duplicate names would make zip-to-map keep only the last value, losing a column with no
        // error at all — the exact silent mis-parse this reader exists to prevent. A spreadsheet
        // export with two identically-named columns is the realistic way this happens.
        val duplicates = header.groupingBy { it }.eachCount().filterValues { it > 1 }.keys
        require(duplicates.isEmpty()) {
            "line $headerLineNumber: duplicate column name(s) ${duplicates.joinToString()} — " +
                "each column must be unique or data is silently lost"
        }

        return lines.drop(1).map { (index, line) ->
            val lineNumber = index + 1
            val fields = splitRow(line, lineNumber)
            require(fields.size == header.size) {
                "line $lineNumber has ${fields.size} fields but the header has ${header.size}: $line"
            }
            lineNumber to header.zip(fields).toMap()
        }
    }

    private fun splitRow(line: String, lineNumber: Int): List<String> {
        require(!line.contains('"')) {
            "line $lineNumber contains a quote character; quoting is unsupported because fields " +
                "must not contain commas: $line"
        }
        return line.split(',').map { it.trim() }
    }
}
```

- [ ] **Step 4: Write the manifest reader**

`harness/src/main/kotlin/app/presence/harness/Manifest.kt`:

```kotlin
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

        val recordings = rows.map { (lineNumber, row) ->
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

        // Evaluation classifies a detection by its recording's role, so a path appearing twice
        // would let a false alarm resolve to the wrong row silently.
        val duplicatePaths = recordings.groupingBy { it.path }.eachCount().filterValues { it > 1 }.keys
        require(duplicatePaths.isEmpty()) {
            "$path lists these audio paths more than once: ${duplicatePaths.joinToString()} — " +
                "each recording must appear once"
        }
        return recordings
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

    /**
     * Adds file and line context to any validation failure, so a bad row in a several-hundred-row
     * manifest is findable. NumberFormatException extends IllegalArgumentException, so a malformed
     * numeric field is covered too.
     */
    private inline fun <T> withRowContext(path: Path, lineNumber: Int, build: () -> T): T =
        try {
            build()
        } catch (failure: IllegalArgumentException) {
            throw IllegalArgumentException(
                "${path.fileName}, line $lineNumber: ${failure.message}",
                failure,
            )
        }

    private fun requireColumns(present: Set<String>, required: List<String>, path: Path) {
        val missing = required - present
        require(missing.isEmpty()) { "$path is missing required columns: ${missing.joinToString()}" }
    }

    private fun <T : Enum<T>> parseEnum(raw: String, values: List<T>, column: String): T =
        values.firstOrNull { it.name.equals(raw.replace('-', '_'), ignoreCase = true) }
            ?: throw IllegalArgumentException(
                "unknown $column '$raw'. Valid values: " +
                    values.joinToString { it.name.lowercase() },
            )
}
```

- [ ] **Step 5: Run the test and verify it passes**

Run: `./gradlew :harness:test --tests "app.presence.harness.ManifestTest"`
Expected: PASS, 22 tests.

- [ ] **Step 6: Commit**

```bash
git add harness/src/main/kotlin/app/presence/harness/Csv.kt \
        harness/src/main/kotlin/app/presence/harness/Manifest.kt \
        harness/src/test/kotlin/app/presence/harness/ManifestTest.kt
git commit -m "feat: corpus manifest format per spec 9.1"
```

---

### Task 16: Metrics — the numbers that decide go/no-go

**Files:**
- Create: `harness/src/main/kotlin/app/presence/harness/Metrics.kt`
- Test: `harness/src/test/kotlin/app/presence/harness/MetricsTest.kt`

**Interfaces:**
- Consumes: `CallEvent` (Task 15).
- Produces: `data class Detection(recordingPath: String, timestampMs: Long)`; `enum class Verdict { SHIP, ITERATE, FALL_BACK }`; `data class Outcome(label: String, fireThreshold: Float, falseAlarms: Int, negativeHours: Double, detectedEvents: Int, totalEvents: Int, latenciesMs: List<Long>)` with computed `falseAlarmsPerHour`, `missRate`, `medianLatencyMs`, `verdict`; `object Metrics` with `scoreEvents(detections, events, latencyBudgetMs = 10_000): EventScoring` where `data class EventScoring(detected: Int, total: Int, latenciesMs: List<Long>)`. Task 17 uses all of it.

**Encoding §4.3 as code is the point of this task.** The verdict thresholds are: ship at ≤ 1 false alarm per 2 hours *and* ≥ 80% detection; iterate down to 2 per hour *or* ≥ 50% detection; fall back below that. `Outcome` computes its own verdict so no caller can quietly apply a friendlier rule.

Event matching rule: a detection counts for an event if it lands within the latency budget after the event's start. Each event matches at most one detection and each detection satisfies at most one event, earliest first — otherwise one lucky detection could "catch" several events at once.

- [ ] **Step 1: Write the failing test**

`harness/src/test/kotlin/app/presence/harness/MetricsTest.kt`:

```kotlin
package app.presence.harness

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class MetricsTest {

    private fun event(startMs: Long, path: String = "a.wav") = CallEvent(path, startMs, "maya")
    private fun detection(timestampMs: Long, path: String = "a.wav") = Detection(path, timestampMs)

    private fun outcome(
        falseAlarms: Int = 0,
        negativeHours: Double = 1.0,
        detected: Int = 10,
        total: Int = 10,
        latencies: List<Long> = listOf(1_000),
    ) = Outcome("ALL", 6f, falseAlarms, negativeHours, detected, total, latencies)

    // --- event matching ---

    @Test
    fun `a detection shortly after an event counts as detecting it`() {
        val scoring = Metrics.scoreEvents(listOf(detection(5_000)), listOf(event(3_000)))

        assertEquals(1, scoring.detected)
        assertEquals(listOf(2_000L), scoring.latenciesMs)
    }

    @Test
    fun `a detection beyond the latency budget does not count`() {
        val scoring = Metrics.scoreEvents(listOf(detection(14_000)), listOf(event(3_000)))

        assertEquals(0, scoring.detected)
        assertTrue(scoring.latenciesMs.isEmpty())
    }

    @Test
    fun `a detection exactly at the latency budget counts`() {
        val scoring = Metrics.scoreEvents(listOf(detection(13_000)), listOf(event(3_000)))

        assertEquals(1, scoring.detected)
        assertEquals(listOf(10_000L), scoring.latenciesMs)
    }

    @Test
    fun `a detection one millisecond past the budget does not count`() {
        val scoring = Metrics.scoreEvents(listOf(detection(13_001)), listOf(event(3_000)))

        assertEquals(0, scoring.detected)
    }

    @Test
    fun `a detection before the event does not count`() {
        assertEquals(0, Metrics.scoreEvents(listOf(detection(1_000)), listOf(event(3_000))).detected)
    }

    @Test
    fun `a detection in a different recording does not count`() {
        val scoring = Metrics.scoreEvents(
            listOf(detection(5_000, "other.wav")),
            listOf(event(3_000, "a.wav")),
        )

        assertEquals(0, scoring.detected)
    }

    @Test
    fun `one detection cannot satisfy two events`() {
        val scoring = Metrics.scoreEvents(
            listOf(detection(4_000)),
            listOf(event(3_000), event(3_500)),
        )

        assertEquals(1, scoring.detected)
        assertEquals(2, scoring.total)
    }

    @Test
    fun `each event is matched by its earliest eligible detection`() {
        val scoring = Metrics.scoreEvents(
            listOf(detection(4_000), detection(5_000)),
            listOf(event(3_000), event(4_500)),
        )

        assertEquals(2, scoring.detected)
        assertEquals(listOf(1_000L, 500L), scoring.latenciesMs)
    }

    @Test
    fun `no events means nothing detected and nothing missed`() {
        val scoring = Metrics.scoreEvents(listOf(detection(1_000)), emptyList())

        assertEquals(0, scoring.detected)
        assertEquals(0, scoring.total)
    }

    // --- derived rates ---

    @Test
    fun `computes false alarms per hour`() {
        assertEquals(4.0, outcome(falseAlarms = 2, negativeHours = 0.5).falseAlarmsPerHour)
    }

    @Test
    fun `false alarms per hour is zero when there is no negative audio`() {
        assertEquals(0.0, outcome(falseAlarms = 0, negativeHours = 0.0).falseAlarmsPerHour)
    }

    @Test
    fun `computes miss rate`() {
        assertEquals(0.25, outcome(detected = 6, total = 8).missRate)
    }

    @Test
    fun `miss rate is zero when there are no events`() {
        assertEquals(0.0, outcome(detected = 0, total = 0).missRate)
    }

    @Test
    fun `median latency of an even count averages the middle two`() {
        assertEquals(1_500L, outcome(latencies = listOf(1_000, 2_000)).medianLatencyMs)
    }

    @Test
    fun `median latency is null with no detections`() {
        assertNull(outcome(latencies = emptyList()).medianLatencyMs)
    }

    // --- verdict, per spec 4.3 ---

    @Test
    fun `ships at one false alarm per two hours with 80 percent detection`() {
        val result = outcome(falseAlarms = 1, negativeHours = 2.0, detected = 8, total = 10)

        assertEquals(Verdict.SHIP, result.verdict)
    }

    @Test
    fun `iterates when false alarms are too frequent even with perfect detection`() {
        val result = outcome(falseAlarms = 1, negativeHours = 1.0, detected = 10, total = 10)

        assertEquals(Verdict.ITERATE, result.verdict)
    }

    @Test
    fun `iterates when detection is below eighty percent even with no false alarms`() {
        val result = outcome(falseAlarms = 0, negativeHours = 2.0, detected = 6, total = 10)

        assertEquals(Verdict.ITERATE, result.verdict)
    }

    @Test
    fun `falls back when false alarms exceed two per hour`() {
        val result = outcome(falseAlarms = 5, negativeHours = 1.0, detected = 10, total = 10)

        assertEquals(Verdict.FALL_BACK, result.verdict)
    }

    @Test
    fun `iterates at exactly two false alarms per hour with exactly half detected`() {
        val result = outcome(falseAlarms = 2, negativeHours = 1.0, detected = 5, total = 10)

        assertEquals(2.0, result.falseAlarmsPerHour)
        assertEquals(0.5, result.detectionRate)
        assertEquals(Verdict.ITERATE, result.verdict, "both bounds are inclusive")
    }

    @Test
    fun `falls back just past either iterate bound`() {
        val tooManyAlarms = outcome(falseAlarms = 21, negativeHours = 10.0, detected = 10, total = 10)
        assertEquals(Verdict.FALL_BACK, tooManyAlarms.verdict, "2.1 per hour is past the bound")

        val tooFewDetected = outcome(falseAlarms = 0, negativeHours = 10.0, detected = 49, total = 100)
        assertEquals(Verdict.FALL_BACK, tooFewDetected.verdict, "49% is past the bound")
    }

    @Test
    fun `falls back when detection is below fifty percent`() {
        val result = outcome(falseAlarms = 0, negativeHours = 2.0, detected = 4, total = 10)

        assertEquals(Verdict.FALL_BACK, result.verdict)
    }
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `./gradlew :harness:test --tests "app.presence.harness.MetricsTest"`
Expected: FAIL — unresolved reference `Detection`.

- [ ] **Step 3: Write the implementation**

`harness/src/main/kotlin/app/presence/harness/Metrics.kt`:

```kotlin
package app.presence.harness

/** The engine fired at this position in this recording. */
data class Detection(val recordingPath: String, val timestampMs: Long)

/** Spec §4.3's three outcomes. */
enum class Verdict { SHIP, ITERATE, FALL_BACK }

data class EventScoring(
    val detected: Int,
    val total: Int,
    val latenciesMs: List<Long>,
)

/**
 * Results at one operating point, for one slice of the corpus.
 *
 * @param label which slice: "ENGLISH", "HEBREW", "CODE_SWITCHED", or "ALL". Spec §4.3 requires
 *   every criterion to be reported separately per language, never pooled.
 * @param negativeHours hours of audio containing no target speech; the denominator for false
 *   alarms.
 */
data class Outcome(
    val label: String,
    val fireThreshold: Float,
    val falseAlarms: Int,
    val negativeHours: Double,
    val detectedEvents: Int,
    val totalEvents: Int,
    val latenciesMs: List<Long>,
) {
    /** Detections fired in audio containing no target speech, per hour. Zero if there is none. */
    val falseAlarmsPerHour: Double
        get() = if (negativeHours <= 0.0) 0.0 else falseAlarms / negativeHours

    /**
     * Fraction of labelled events that went undetected. Zero for a corpus with no events, since
     * nothing was there to miss.
     *
     * Deliberately NOT `1 - detectionRate` in that case: [detectionRate] returns zero for an empty
     * corpus as a safety default, so a misconfigured or empty run falls back rather than reporting
     * a perfect score. The two are inconsistent only when there are no events at all, and that
     * asymmetry is the point.
     */
    val missRate: Double
        get() = if (totalEvents == 0) 0.0 else (totalEvents - detectedEvents).toDouble() / totalEvents

    /** Fraction of labelled events detected. Zero for an empty corpus, so the verdict falls back. */
    val detectionRate: Double
        get() = if (totalEvents == 0) 0.0 else detectedEvents.toDouble() / totalEvents

    val medianLatencyMs: Long?
        get() {
            if (latenciesMs.isEmpty()) return null
            val sorted = latenciesMs.sorted()
            val middle = sorted.size / 2
            return if (sorted.size % 2 == 1) sorted[middle] else (sorted[middle - 1] + sorted[middle]) / 2
        }

    /**
     * Spec §4.3, encoded so no caller can quietly apply a friendlier rule.
     *
     * Ship: ≤ 1 false alarm per 2 hours AND ≥ 80% detection.
     * Iterate: up to 2 false alarms per hour AND ≥ 50% detection.
     * Fall back: anything worse.
     */
    val verdict: Verdict
        get() = when {
            falseAlarmsPerHour <= 0.5 && detectionRate >= 0.80 -> Verdict.SHIP
            falseAlarmsPerHour <= 2.0 && detectionRate >= 0.50 -> Verdict.ITERATE
            else -> Verdict.FALL_BACK
        }
}

object Metrics {

    /** Spec §4.3: a real event must be caught within ten seconds to count. */
    const val DEFAULT_LATENCY_BUDGET_MS = 10_000L

    /**
     * Matches detections to labelled call events.
     *
     * Each event is satisfied by at most one detection and each detection satisfies at most one
     * event, earliest first — otherwise a single lucky detection could "catch" several events
     * and flatter the result.
     */
    fun scoreEvents(
        detections: List<Detection>,
        events: List<CallEvent>,
        latencyBudgetMs: Long = DEFAULT_LATENCY_BUDGET_MS,
    ): EventScoring {
        val unusedByRecording = detections
            .groupBy { it.recordingPath }
            .mapValues { (_, group) -> group.sortedBy { it.timestampMs }.toMutableList() }

        val latencies = ArrayList<Long>()
        for (event in events.sortedBy { it.startMs }) {
            val candidates = unusedByRecording[event.recordingPath] ?: continue
            val match = candidates.firstOrNull {
                it.timestampMs >= event.startMs && it.timestampMs - event.startMs <= latencyBudgetMs
            } ?: continue
            candidates.remove(match)
            latencies += match.timestampMs - event.startMs
        }

        return EventScoring(detected = latencies.size, total = events.size, latenciesMs = latencies)
    }
}
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `./gradlew :harness:test --tests "app.presence.harness.MetricsTest"`
Expected: PASS, 22 tests.

- [ ] **Step 5: Commit**

```bash
git add harness/src/main/kotlin/app/presence/harness/Metrics.kt \
        harness/src/test/kotlin/app/presence/harness/MetricsTest.kt
git commit -m "feat: Phase 0 metrics with spec 4.3 verdict thresholds"
```

---

### Task 17: Evaluation runner, report and CLI

**Files:**
- Modify: `shared/src/commonMain/kotlin/app/presence/engine/DetectionEngine.kt` — add an optional observation listener
- Modify: `shared/src/commonTest/kotlin/app/presence/engine/DetectionEngineTest.kt` — one test for it
- Create: `harness/src/main/kotlin/app/presence/harness/Evaluation.kt`
- Create: `harness/src/main/kotlin/app/presence/harness/Report.kt`
- Create: `harness/src/main/kotlin/app/presence/harness/Main.kt`
- Test: `harness/src/test/kotlin/app/presence/harness/EvaluationTest.kt`

**Interfaces:**
- Consumes: everything from Tasks 1–15.
- Produces: `data class Trace(recording: Recording, observations: List<Observation>)`; `object Evaluation` with `splitCorpus(recordings): CorpusSplit`, `replay(traces, events, threshold, label, filter): Outcome`, `sweep(traces, events, thresholds): List<Outcome>`; `object Report` with `markdown(outcomes, provenance): String` and `csv(outcomes): String`.

**The design decision that makes this fast enough to iterate on.** Embedding is by far the most expensive step, and sweeping twenty thresholds by re-running the engine twenty times would take twenty times as long. Instead the engine runs **once** per recording, emitting an observation trace, and each threshold is evaluated by replaying a fresh `EvidenceAccumulator` over the cached trace. Sweeping becomes nearly free, which is what makes the operating-point curve practical rather than an overnight job.

**Enrollment/evaluation split, stated explicitly to avoid leakage:**

| Purpose | Which recordings |
|---|---|
| Enrollment | `content_type = read`, household members only — §9.1's "short read passage in each language" is exactly this material |
| Evaluation positives | Target-child `calling` and `spontaneous` recordings that have labelled events |
| Evaluation negatives | Everything whose role is not `target_child` — siblings, parent, media, strangers, room tone — excluding anything used for enrollment |

Households are also split: `--holdout` names households touched only for the final measurement, per §9.1's rule to split by household and never by utterance.

- [ ] **Step 1: Add the observation listener to the engine**

In `shared/src/commonMain/kotlin/app/presence/engine/DetectionEngine.kt`, add a parameter to the constructor after `targetPersonId`:

```kotlin
    private val targetPersonId: String,
    private val config: DetectionEngineConfig = DetectionEngineConfig(),
    /**
     * Receives every scored observation before it reaches the accumulator.
     *
     * The harness uses this to cache traces so an operating-point sweep can replay accumulators
     * without re-embedding audio. Also useful for on-device diagnostics later.
     */
    private val observationListener: ((Observation) -> Unit)? = null,
```

and in `process`, immediately after building the observation, notify the listener:

```kotlin
            val observation = Observation(timestampMs, score.logLikelihoodRatio)
            observationListener?.invoke(observation)
            val evidence = accumulator.accept(observation)
```

replacing the previous line that constructed `Observation(...)` inline.

- [ ] **Step 2: Add a test for the listener**

Append to `shared/src/commonTest/kotlin/app/presence/engine/DetectionEngineTest.kt`:

```kotlin
    @Test
    fun `reports every observation to the listener`() {
        val seen = mutableListOf<Observation>()
        val subject = DetectionEngine(
            vadGate = VadGate(AlwaysSpeech()),
            embedder = FakeEmbedder(),
            matcher = HouseholdMatcher(maya, listOf(parent, sibling)),
            accumulator = EvidenceAccumulator(),
            targetPersonId = "maya",
            observationListener = { seen += it },
        )

        repeat(200) { subject.accept(targetFrame()) }
        subject.flush()

        assertTrue(seen.size >= 2, "expected several observations, got ${seen.size}")
        assertTrue(seen.all { it.logLikelihoodRatio > 0f }, "target audio should score positive")
    }
```

Run: `./gradlew :shared:jvmTest --tests "app.presence.engine.DetectionEngineTest"`
Expected: PASS, 8 tests.

- [ ] **Step 3: Write the failing evaluation test**

`harness/src/test/kotlin/app/presence/harness/EvaluationTest.kt`:

```kotlin
package app.presence.harness

import app.presence.domain.Language
import app.presence.engine.Observation
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class EvaluationTest {

    private fun recording(
        path: String,
        role: RecordingRole,
        contentType: ContentType,
        speakerId: String = "maya",
        language: Language? = Language.ENGLISH,
        codeSwitched: Boolean = false,
        household: String = "h1",
    ) = Recording(
        path = path,
        householdId = household,
        speakerId = speakerId,
        role = role,
        language = language,
        codeSwitched = codeSwitched,
        contentType = contentType,
        distanceMeters = 1.0,
        placement = "table",
        background = "quiet",
        room = "kitchen",
        date = "2026-08-01",
    )

    private fun trace(recording: Recording, vararg llr: Pair<Long, Float>) =
        Trace(recording, llr.map { (ts, value) -> Observation(ts, value) }, durationSeconds = 60.0)

    // --- corpus split ---

    @Test
    fun `read passages become enrollment material`() {
        val split = Evaluation.splitCorpus(
            listOf(
                recording("read.wav", RecordingRole.TARGET_CHILD, ContentType.READ),
                recording("call.wav", RecordingRole.TARGET_CHILD, ContentType.CALLING),
            ),
        )

        assertEquals(listOf("read.wav"), split.enrollment.map { it.path })
    }

    @Test
    fun `target calling and spontaneous audio become positives`() {
        val split = Evaluation.splitCorpus(
            listOf(
                recording("call.wav", RecordingRole.TARGET_CHILD, ContentType.CALLING),
                recording("chat.wav", RecordingRole.TARGET_CHILD, ContentType.SPONTANEOUS),
            ),
        )

        assertEquals(setOf("call.wav", "chat.wav"), split.positives.map { it.path }.toSet())
    }

    @Test
    fun `everything that is not the target becomes a negative`() {
        val split = Evaluation.splitCorpus(
            listOf(
                recording("sib.wav", RecordingRole.SIBLING, ContentType.SPONTANEOUS, "noam"),
                recording("tv.wav", RecordingRole.MEDIA, ContentType.SPONTANEOUS, "tv1"),
                recording("tone.wav", RecordingRole.ROOM_TONE, ContentType.ROOM_TONE, "room", null),
            ),
        )

        assertEquals(3, split.negatives.size)
        assertTrue(split.positives.isEmpty())
    }

    @Test
    fun `enrollment audio is never also evaluated`() {
        val split = Evaluation.splitCorpus(
            listOf(recording("read.wav", RecordingRole.SIBLING, ContentType.READ, "noam")),
        )

        assertTrue(split.negatives.isEmpty(), "enrollment material must not leak into evaluation")
    }

    // --- replay ---

    @Test
    fun `a positive trace above threshold detects its event`() {
        val positive = recording("call.wav", RecordingRole.TARGET_CHILD, ContentType.CALLING)
        val traces = listOf(trace(positive, 1_000L to 4f, 2_000L to 4f))
        val events = listOf(CallEvent("call.wav", 500, "maya"))

        val outcome = Evaluation.replay(traces, events, threshold = 6f, label = "ALL")

        assertEquals(1, outcome.detectedEvents)
        assertEquals(0, outcome.falseAlarms)
    }

    @Test
    fun `raising the threshold turns a detection into a miss`() {
        val positive = recording("call.wav", RecordingRole.TARGET_CHILD, ContentType.CALLING)
        val traces = listOf(trace(positive, 1_000L to 4f, 2_000L to 4f))
        val events = listOf(CallEvent("call.wav", 500, "maya"))

        val outcome = Evaluation.replay(traces, events, threshold = 20f, label = "ALL")

        assertEquals(0, outcome.detectedEvents)
        assertEquals(1, outcome.totalEvents)
    }

    @Test
    fun `a firing negative trace is a false alarm`() {
        val negative = recording("tv.wav", RecordingRole.MEDIA, ContentType.SPONTANEOUS, "tv1")
        val traces = listOf(trace(negative, 1_000L to 4f, 2_000L to 4f))

        val outcome = Evaluation.replay(traces, emptyList(), threshold = 6f, label = "ALL")

        assertEquals(1, outcome.falseAlarms)
        assertTrue(outcome.negativeHours > 0.0)
    }

    @Test
    fun `the language filter slices the corpus`() {
        val hebrew = recording("he.wav", RecordingRole.MEDIA, ContentType.SPONTANEOUS, "tv1", Language.HEBREW)
        val english = recording("en.wav", RecordingRole.MEDIA, ContentType.SPONTANEOUS, "tv2", Language.ENGLISH)
        val traces = listOf(
            trace(hebrew, 1_000L to 8f),
            trace(english, 1_000L to 8f),
        )

        val outcome = Evaluation.replay(traces, emptyList(), 6f, "HEBREW") {
            it.language == Language.HEBREW && !it.codeSwitched
        }

        assertEquals(1, outcome.falseAlarms, "only the Hebrew recording should count")
    }

    @Test
    fun `the sweep produces one outcome per threshold`() {
        val negative = recording("tv.wav", RecordingRole.MEDIA, ContentType.SPONTANEOUS, "tv1")
        val traces = listOf(trace(negative, 1_000L to 4f, 2_000L to 4f))

        val outcomes = Evaluation.sweep(traces, emptyList(), listOf(2f, 6f, 20f))

        assertEquals(3, outcomes.size)
        assertEquals(listOf(2f, 6f, 20f), outcomes.map { it.fireThreshold })
        assertTrue(
            outcomes.last().falseAlarms <= outcomes.first().falseAlarms,
            "false alarms must not increase as the threshold rises",
        )
    }

    // --- report ---

    @Test
    fun `the report names every language slice and the verdict`() {
        val outcomes = listOf(
            Outcome("ENGLISH", 6f, 1, 2.0, 8, 10, listOf(1_000)),
            Outcome("HEBREW", 6f, 4, 2.0, 4, 10, listOf(3_000)),
        )

        val markdown = Report.markdown(outcomes, provenance = "vad: x\nembedding: y")

        assertTrue(markdown.contains("ENGLISH"))
        assertTrue(markdown.contains("HEBREW"))
        assertTrue(markdown.contains("SHIP"))
        assertTrue(markdown.contains("FALL_BACK"))
        assertTrue(markdown.contains("embedding: y"), "provenance must appear in the report")
    }

    @Test
    fun `the report surfaces warnings that qualify the numbers`() {
        val outcomes = listOf(Outcome("ALL", 6f, 1, 2.0, 8, 10, listOf(1_000L)))

        val markdown = Report.markdown(outcomes, "vad: x", listOf("noam: enrolled partially"))

        assertTrue(markdown.contains("Warnings"))
        assertTrue(markdown.contains("noam: enrolled partially"))
    }

    @Test
    fun `the csv has a header and one row per outcome`() {
        val csv = Report.csv(listOf(Outcome("ALL", 6f, 1, 2.0, 8, 10, listOf(1_000))))

        val lines = csv.trim().lines()
        assertEquals(2, lines.size)
        assertTrue(lines[0].startsWith("label,threshold"))
    }
}
```

- [ ] **Step 4: Run the test and verify it fails**

Run: `./gradlew :harness:test --tests "app.presence.harness.EvaluationTest"`
Expected: FAIL — unresolved reference `Trace`.

- [ ] **Step 5: Write the evaluation logic**

`harness/src/main/kotlin/app/presence/harness/Evaluation.kt`:

```kotlin
package app.presence.harness

import app.presence.engine.AccumulatorConfig
import app.presence.engine.EvidenceAccumulator
import app.presence.engine.Observation

/**
 * One recording's scored observations, cached so a threshold sweep can replay accumulators
 * without re-embedding audio — the difference between a sweep taking seconds and taking hours.
 */
data class Trace(
    val recording: Recording,
    val observations: List<Observation>,
    val durationSeconds: Double,
)

/**
 * How the corpus divides. Stated explicitly because a leak between enrollment and evaluation
 * inflates results silently (spec §9.1).
 */
data class CorpusSplit(
    val enrollment: List<Recording>,
    val positives: List<Recording>,
    val negatives: List<Recording>,
)

object Evaluation {

    /**
     * Read passages enrol; the target's calling and spontaneous audio are positives; everything
     * else is a negative. Enrollment material never appears in evaluation.
     */
    fun splitCorpus(recordings: List<Recording>): CorpusSplit {
        val enrollment = recordings.filter {
            it.role.isHouseholdMember && it.contentType == ContentType.READ
        }
        val remainder = recordings - enrollment.toSet()
        return CorpusSplit(
            enrollment = enrollment,
            positives = remainder.filter { it.role == RecordingRole.TARGET_CHILD },
            negatives = remainder.filter { it.role != RecordingRole.TARGET_CHILD },
        )
    }

    /**
     * Replays a fresh accumulator over cached traces at one threshold.
     *
     * @param filter restricts which recordings count, for per-language slices. Spec §4.3 requires
     *   English, Hebrew and code-switched speech to be reported separately.
     */
    fun replay(
        traces: List<Trace>,
        events: List<CallEvent>,
        threshold: Float,
        label: String,
        filter: (Recording) -> Boolean = { true },
    ): Outcome {
        val selected = traces.filter { filter(it.recording) }
        val selectedPaths = selected.map { it.recording.path }.toSet()

        val detections = ArrayList<Detection>()
        var negativeSeconds = 0.0
        var falseAlarms = 0

        for (trace in selected) {
            val accumulator = EvidenceAccumulator(AccumulatorConfig(fireThreshold = threshold))
            // Role is known here, so a detection is classified where it is created rather than by a
            // second path lookup that duplicate paths could resolve to the wrong row — and this is
            // O(n) instead of O(detections x traces).
            val isTarget = trace.recording.role == RecordingRole.TARGET_CHILD
            for (observation in trace.observations) {
                if (accumulator.accept(observation) != null) {
                    detections += Detection(trace.recording.path, observation.timestampMs)
                    if (!isTarget) falseAlarms++
                }
            }
            if (!isTarget) negativeSeconds += trace.durationSeconds
        }

        val relevantEvents = events.filter { it.recordingPath in selectedPaths }
        val scoring = Metrics.scoreEvents(detections, relevantEvents)

        return Outcome(
            label = label,
            fireThreshold = threshold,
            falseAlarms = falseAlarms,
            negativeHours = negativeSeconds / 3_600.0,
            detectedEvents = scoring.detected,
            totalEvents = scoring.total,
            latenciesMs = scoring.latenciesMs,
        )
    }

    /** Replays every threshold. Cheap, because the embeddings are already computed. */
    fun sweep(
        traces: List<Trace>,
        events: List<CallEvent>,
        thresholds: List<Float>,
        label: String = "ALL",
        filter: (Recording) -> Boolean = { true },
    ): List<Outcome> = thresholds.map { replay(traces, events, it, label, filter) }
}
```

- [ ] **Step 6: Write the report**

`harness/src/main/kotlin/app/presence/harness/Report.kt`:

```kotlin
package app.presence.harness

object Report {

    fun markdown(
        outcomes: List<Outcome>,
        provenance: String,
        warnings: List<String> = emptyList(),
    ): String = buildString {
        appendLine("# Phase 0 accuracy report")
        appendLine()
        appendLine("Go / no-go criteria are spec §4.3. **Both languages must clear every criterion")
        appendLine("independently** — a system that passes in English and fails in Hebrew has not passed.")
        appendLine()
        appendLine("## Models")
        appendLine()
        appendLine("```")
        appendLine(provenance.trim())
        appendLine("```")
        appendLine()
        if (warnings.isNotEmpty()) {
            appendLine("## Warnings")
            appendLine()
            appendLine("These affect how much the numbers below can be trusted.")
            appendLine()
            for (warning in warnings) appendLine("- $warning")
            appendLine()
        }
        appendLine("## Results")
        appendLine()
        appendLine("| Slice | Threshold | FA/hour | Detection | Miss rate | Median latency | Verdict |")
        appendLine("|---|---:|---:|---:|---:|---:|---|")
        for (outcome in outcomes) {
            appendLine(
                "| ${outcome.label} | ${format(outcome.fireThreshold)} | " +
                    "${format(outcome.falseAlarmsPerHour)} | " +
                    "${percent(outcome.detectionRate)} | ${percent(outcome.missRate)} | " +
                    "${outcome.medianLatencyMs?.let { "$it ms" } ?: "—"} | ${outcome.verdict} |",
            )
        }
        appendLine()
        appendLine("## Reading this")
        appendLine()
        appendLine("- **SHIP** — ≤ 1 false alarm per 2 hours and ≥ 80% detection within 10 s.")
        appendLine("- **ITERATE** — up to 2 false alarms per hour and ≥ 50% detection.")
        appendLine("- **FALL_BACK** — worse. The fallback is the wake-phrase channel alone (spec §4.3).")
        appendLine()
        appendLine("EER is absent deliberately: error costs here are asymmetric, so the equal-error")
        appendLine("point is not the operating point we care about (spec §4.1).")
    }

    fun csv(outcomes: List<Outcome>): String = buildString {
        appendLine(
            "label,threshold,false_alarms,negative_hours,false_alarms_per_hour," +
                "detected_events,total_events,detection_rate,miss_rate,median_latency_ms,verdict",
        )
        for (o in outcomes) {
            appendLine(
                "${o.label},${o.fireThreshold},${o.falseAlarms},${format(o.negativeHours)}," +
                    "${format(o.falseAlarmsPerHour)},${o.detectedEvents},${o.totalEvents}," +
                    "${format(o.detectionRate)},${format(o.missRate)},${o.medianLatencyMs ?: ""},${o.verdict}",
            )
        }
    }

    private fun format(value: Number): String {
        val rounded = kotlin.math.round(value.toDouble() * 1000) / 1000
        return rounded.toString()
    }

    private fun percent(value: Double): String {
        val rounded = kotlin.math.round(value * 1000) / 10
        return "$rounded%"
    }
}
```

- [ ] **Step 7: Run the tests and verify they pass**

Run: `./gradlew :harness:test --tests "app.presence.harness.EvaluationTest"`
Expected: PASS, 12 tests.

- [ ] **Step 8: Write the CLI**

`harness/src/main/kotlin/app/presence/harness/Main.kt`:

```kotlin
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
    warnings: MutableList<String>,
): List<VoiceProfile> =
    enrollment.groupBy { it.speakerId }.mapNotNull { (speakerId, group) ->
        val clips = group.mapNotNull { recording ->
            val language = recording.language ?: return@mapNotNull null
            val samples = WavReader.read(root.resolve(recording.path))
            if (samples.size < SAMPLE_RATE_HZ) return@mapNotNull null
            EnrollmentClip(language, embedder.embed(samples), samples.size.toDouble() / SAMPLE_RATE_HZ)
        }
        if (clips.isEmpty()) {
            warnings += "$speakerId: no usable enrollment audio"
            return@mapNotNull null
        }
        // Require only the languages this person actually has material for. Demanding both would
        // flag every monolingual household — likely the common case — drowning real shortfalls in
        // expected noise.
        val policy = EnrollmentPolicy(
            minSecondsPerLanguage = 20.0,
            requiredLanguages = clips.map { it.language }.toSet(),
        )
        when (val result = EnrollmentService(policy).enroll(speakerId, group.first().role.toProfileRole(), clips)) {
            is EnrollmentResult.Complete -> result.profile
            is EnrollmentResult.Incomplete -> {
                warnings += "$speakerId: enrolled partially, short by ${result.shortfallSeconds}"
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

private val KNOWN_FLAGS = setOf(
    "help", "manifest", "models", "out", "holdout", "input-name", "output-name",
)

/**
 * `internal` so it can be tested directly, and throwing rather than exiting for the same reason —
 * `exitProcess` inside a parser would kill the test JVM.
 */
internal fun parseArgs(argv: Array<String>): Map<String, String> {
    val args = mutableMapOf<String, String>()
    var index = 0
    while (index < argv.size) {
        val token = argv[index]
        require(token.startsWith("--")) { "unexpected argument: $token" }
        val key = token.removePrefix("--")
        // A silently-ignored typo matters here: --holdout is the leakage guard, so a mistyped
        // --holdouts would let held-out households quietly rejoin the evaluation.
        require(key in KNOWN_FLAGS) {
            "unknown flag --$key. Known flags: ${KNOWN_FLAGS.sorted().joinToString { "--$it" }}"
        }
        if (key == "help") {
            args["help"] = "true"
            index += 1
            continue
        }
        require(index + 1 < argv.size) { "missing value for --$key" }
        val value = argv[index + 1]
        // A flag-shaped value almost always means the real value was forgotten, and would otherwise
        // swallow the following flag.
        require(!value.startsWith("--")) { "--$key expects a value but was followed by $value" }
        args[key] = value
        index += 2
    }
    return args
}
```

- [ ] **Step 9: Verify the CLI runs**

Run: `./gradlew :harness:run --args="--help"`
Expected: the usage text prints.

- [ ] **Step 10: Run the full suite**

Run: `./gradlew check`
Expected: PASS across `:shared` and `:harness`.

- [ ] **Step 11: Commit**

```bash
git add shared/src/commonMain/kotlin/app/presence/engine/DetectionEngine.kt \
        shared/src/commonTest/kotlin/app/presence/engine/DetectionEngineTest.kt \
        harness/src/main/kotlin/app/presence/harness/Evaluation.kt \
        harness/src/main/kotlin/app/presence/harness/Report.kt \
        harness/src/main/kotlin/app/presence/harness/Main.kt \
        harness/src/test/kotlin/app/presence/harness/EvaluationTest.kt
git commit -m "feat: Phase 0 evaluation runner, threshold sweep and report"
```

---

## After the plan: running Phase 0 for real

The code is the easy half. The measurement is the deliverable, and it needs the corpus from spec
§9.1, which is a data-collection effort rather than an engineering one. Sequence:

1. **Settle consent scope before recording anything** (spec §10, open question 6). Whether the
   recordings may train a shipped model or only evaluate it cannot be added to a consent form
   afterwards.
2. Collect per §9.1 — both languages, code-switching, varied distance and background, children's TV
   as negative material, at least three households with two held out.
3. Fill in `recordings.csv` and `events.csv`.
4. Run the sweep. Read `report.md`.
5. **If the verdict is FALL_BACK**, do not start tuning constants. Go to L3 (fine-tuning) first —
   it is the single largest lever, worth roughly an order of magnitude, and no amount of threshold
   fiddling substitutes for it.

## Deliberately not in this plan

Spec §9 lists five things Phase 0 must settle. Three are covered by Task 17's sweep — cross-lingual
performance, code-switching, and whether media-playback state suppresses television false alarms
(the `MEDIA` role gives the last one its measurement). Two are **not**, and are called out here so
the gap is a decision rather than an oversight:

- **Samrómur Children baseline.** The manifest pipeline is built around households — a target, a
  cohort, and labelled call events. Samrómur has none of those; it is thousands of children reading
  passages, and evaluating on it means generating same-speaker and different-speaker trial pairs and
  reporting EER for comparison with published work. That is a different harness sharing only the
  embedder, and it belongs in its own short plan alongside the L3 fine-tuning work, where its
  numbers are actually needed. **It is not a substitute for the household set**, which is what the
  go/no-go criteria are measured on (spec §4.3).
- **Picovoice Eagle benchmark.** A third-party SDK integration with its own licensing and API,
  serving as a build-versus-buy baseline (spec §4.1). Worth doing, but it is procurement plus an
  adapter, not a change to this pipeline — and it only matters if our own numbers disappoint.

Also worth knowing while running the sweep: **Task 17 sweeps the fire threshold, not the
accumulation window.** Window length is the other half of the core trade-off — longer is more
accurate and slower to react (spec §9) — and `AccumulatorConfig.windowMs` is a constructor
parameter, so exploring it means a second sweep loop over window lengths. Add it if the threshold
curve alone does not reach a satisfying operating point; it is a few lines in `Evaluation.sweep`.

## Follow-on plans

This plan is Phase 0 only, because spec §9 makes it the gate: *"Nothing else should be built
first."* The rest, in order, each its own plan:

| Plan | Scope | Gated on |
|---|---|---|
| **L3 model fine-tuning** | Python/WeSpeaker pipeline: Samrómur Children, ChildAugment over ivrit.ai, export to ONNX. Offline; ships as a better model file | Phase 0 numbers |
| **Android app** | Foreground service, screen-state gating, overlay intervention, staged permissions, enrollment and settings UI, SQLDelight storage | Phase 0 SHIP or ITERATE, plus the UI design |
| **L5 wake-phrase channel** | Query-by-Example keyword spotting; language-agnostic, so Hebrew is free | Phase 0; promoted to v1 candidate per spec §4.2 |
| **iOS app** | SwiftUI, AVAudioEngine, Live Activity and ducking (Tier 1); Family Controls shield (Tier 2) | Android shipping |

The engine built here is reused unchanged by every one of them — that is the whole point of putting
it in `commonMain`.

