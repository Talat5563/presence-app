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
- **Go / no-go thresholds (spec §4.3):** ship at ≤ 1 false alarm per 2 hours with ≥ 80% of real events detected within 10 s; iterate between that and 2 false alarms per hour or 50% detection; fall back below that.

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
  Main.kt                           CLI entry point

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

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
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
- Produces: `class Framer(frameSize: Int, hopSize: Int = frameSize)` with `push(samples: FloatArray): List<FloatArray>` and `reset()`. Task 11 and Task 15 feed audio through it.

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
Expected: PASS, 8 tests.

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
- Produces: `class Embedding` with `values: FloatArray`, `dimension: Int`, and factory `Embedding.ofNormalized(values: FloatArray): Embedding`; top-level `cosine(a: Embedding, b: Embedding): Float` and `centroidOf(embeddings: List<Embedding>): Embedding`. Tasks 4, 5, 6, 8 and 12 all depend on these exact names.

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
class Embedding private constructor(val values: FloatArray) {

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
    val accumulator = FloatArray(dimension)
    for (embedding in embeddings) {
        require(embedding.dimension == dimension) {
            "dimension mismatch in centroid input: expected $dimension, got ${embedding.dimension}"
        }
        for (i in 0 until dimension) accumulator[i] += embedding.values[i]
    }
    return Embedding.ofNormalized(accumulator)
}
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `./gradlew :shared:jvmTest --tests "app.presence.engine.EmbeddingTest"`
Expected: PASS, 13 tests.

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
        val dimensions = centroids.values.map { it.dimension }.distinct()
        require(dimensions.size == 1) {
            "profile '$personId' has centroids of differing dimensions: $dimensions"
        }
    }

    val languages: Set<Language> get() = centroids.keys
}
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `./gradlew :shared:jvmTest --tests "app.presence.domain.VoiceProfileTest"`
Expected: PASS, 6 tests.

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
        val merged = FloatArray(currentSampleCount)
        var offset = 0
        for (chunk in current) {
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
Expected: PASS, 10 tests.

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
- Produces: `data class EnrollmentClip(language: Language, embedding: Embedding, speechSeconds: Double)`; `data class EnrollmentPolicy(minSecondsPerLanguage: Double = 45.0, requiredLanguages: Set<Language> = setOf(ENGLISH, HEBREW))`; `sealed interface EnrollmentResult` with `Complete(profile)` and `Incomplete(profile, shortfallSeconds)`; `class EnrollmentService(policy)` with `enroll(personId, role, clips): EnrollmentResult`. Tasks 15 and 16 use this to build profiles from corpus audio.

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

        // Open-set gate. A stranger or the television resembles nobody in particular, but the
        // posterior alone would still crown a winner among the enrolled.
        if (z < config.minCohortZ) llr = minOf(llr, 0f)

        // "Not them" vetoes. A drifting profile should decay toward silence, never toward
        // firing on a sibling.
        val closestNegative = target.negatives.maxOfOrNull { cosine(observation, it) }
        if (closestNegative != null && closestNegative >= targetSimilarity) llr = minOf(llr, 0f)

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
            .coerceAtLeast(1e-4f)
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
Expected: PASS, 11 tests.

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

The threshold on the *sum* is therefore the real operating point of the whole product, and Task 16 sweeps it.

- [ ] **Step 1: Write the failing test**

`shared/src/commonTest/kotlin/app/presence/engine/EvidenceAccumulatorTest.kt`:

```kotlin
package app.presence.engine

import kotlin.math.abs
import kotlin.test.Test
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
    fun `rejects a non-positive window`() {
        kotlin.test.assertFailsWith<IllegalArgumentException> {
            AccumulatorConfig(windowMs = 0)
        }
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

    /** Returns the accumulated evidence if this observation triggers a detection, else null. */
    fun accept(observation: Observation): Float? {
        window.addLast(observation)

        val cutoff = observation.timestampMs - config.windowMs
        while (window.isNotEmpty() && window.first().timestampMs < cutoff) {
            window.removeFirst()
        }

        val firedAt = lastFiredAtMs
        if (firedAt != null && observation.timestampMs - firedAt < config.cooldownMs) return null

        var total = 0f
        for (entry in window) total += entry.logLikelihoodRatio
        if (total < config.fireThreshold) return null

        lastFiredAtMs = observation.timestampMs
        window.clear()
        return total
    }

    /** Clears accumulated evidence and any active cooldown. */
    fun reset() {
        window.clear()
        lastFiredAtMs = null
    }
}
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `./gradlew :shared:jvmTest --tests "app.presence.engine.EvidenceAccumulatorTest"`
Expected: PASS, 10 tests.

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
- Produces: `interface SpeakerEmbedder { val dimension: Int; fun embed(samples: FloatArray): Embedding }`; `data class DetectionEvent(personId: String, timestampMs: Long, accumulatedEvidence: Float, peakSimilarity: Float)`; `class DetectionEngineConfig(minEmbedDurationMs, embedWindowMs, embedHopMs)`; `class DetectionEngine(vadGate, embedder, matcher, accumulator, targetPersonId, config)` with `accept(frame: FloatArray): DetectionEvent?`, `flush(): DetectionEvent?`, `reset()`. Task 16 drives it.

This is the class spec §3 promised: it knows nothing about microphones, services, or screens, so the whole recognition pipeline is testable on the JVM.

One subtlety worth getting right: a long speech segment is cut into overlapping embedding windows, and **each window is a separate observation**, not averaged into one. Averaging would discard exactly the independent evidence the accumulator exists to sum.

- [ ] **Step 1: Write the failing test**

`shared/src/commonTest/kotlin/app/presence/engine/DetectionEngineTest.kt`:

```kotlin
package app.presence.engine

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
        // Cover the tail with a final window ending at the segment boundary, unless the last
        // window already reached it.
        val lastCovered = offset - hopSamples + windowSamples
        if (lastCovered < samples.size) {
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
Expected: PASS, 78 tests across all engine classes.

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
- Produces: `object WavReader` with `read(path: java.nio.file.Path): FloatArray`; `class UnsupportedAudioException(message: String) : Exception(message)`. Tasks 15 and 16 use it.

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

tasks.test {
    useJUnitPlatform()
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
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Path
import javax.sound.sampled.AudioSystem
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

        AudioSystem.getAudioInputStream(path.toFile()).use { stream ->
            val format = stream.format

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
            val bytes = stream.readAllBytes()
            val order = if (format.isBigEndian) ByteOrder.BIG_ENDIAN else ByteOrder.LITTLE_ENDIAN
            val shorts = ByteBuffer.wrap(bytes).order(order).asShortBuffer()

            val frameCount = shorts.limit() / channels
            val samples = FloatArray(frameCount)
            for (frame in 0 until frameCount) {
                var sum = 0f
                for (channel in 0 until channels) {
                    sum += shorts.get(frame * channels + channel) / FULL_SCALE
                }
                samples[frame] = (sum / channels).coerceIn(-1f, 1f)
            }
            return samples
        }
    }
}
```

- [ ] **Step 5: Run the test and verify it passes**

Run: `./gradlew :harness:test`
Expected: PASS, 6 tests.

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
- Modify: `README.md` — add a "Running Phase 0" section pointing at the scripts

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `class OnnxModelSession(modelPath: java.nio.file.Path) : AutoCloseable` with `inputNames: List<String>`, `outputNames: List<String>`, `describeSignature(): String`, `run(inputs: Map<String, Any>): Map<String, Any>`, and `close()`. Tasks 12 and 13 build on it.

**Read this before implementing.** The exact input and output tensor names differ between models and between versions of the same model — Silero VAD v4 takes separate `h` and `c` state tensors while v5 takes a single `state` tensor, and embedding models vary in whether they expect raw samples or precomputed features. **Do not guess.** Step 3 dumps the real signature and Tasks 12 and 13 are written against what it prints.

- [ ] **Step 1: Write the model-listing script**

`tools/list-models.sh`:

```bash
#!/usr/bin/env bash
# Lists the speaker-recognition models sherpa-onnx publishes, plus the Silero VAD asset.
# Note the upstream tag really is spelled "recongition".
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
echo "Pick one embedding model. Guidance:"
echo "  * 'wespeaker-' and '3dspeaker-' prefixes indicate the upstream toolkit."
echo "  * Prefer an English or multilingual model over a language-specific one."
echo "  * Note the model's size — it ships inside the app (spec §4)."
echo "  * CHECK ITS LICENCE before shipping (spec §10, open question 2)."
```

Make it executable and run it:

```bash
chmod +x tools/list-models.sh
./tools/list-models.sh
```

Record the chosen embedding model URL — the next step needs it.

- [ ] **Step 2: Write the fetch script**

`tools/fetch-models.sh`:

```bash
#!/usr/bin/env bash
# Downloads the models Phase 0 needs into models/. Binaries are gitignored (*.onnx).
#
# Usage: ./tools/fetch-models.sh <embedding-model-url>
#   Find candidate URLs with ./tools/list-models.sh
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "usage: $0 <embedding-model-url>" >&2
  echo "run ./tools/list-models.sh to see the choices" >&2
  exit 1
fi

EMBEDDING_URL="$1"
VAD_URL="https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx"
DEST="models"

mkdir -p "$DEST"
echo "Fetching Silero VAD..."
curl -fsSL "$VAD_URL" -o "$DEST/silero_vad.onnx"
echo "Fetching speaker embedding model..."
curl -fsSL "$EMBEDDING_URL" -o "$DEST/speaker_embedding.onnx"

# Record provenance: which model was used is part of every Phase 0 result.
cat > "$DEST/PROVENANCE.txt" <<PROVENANCE
vad: $VAD_URL
embedding: $EMBEDDING_URL
fetched: $(date -u +%Y-%m-%dT%H:%M:%SZ)
PROVENANCE

ls -lh "$DEST"
echo
echo "Recorded provenance in $DEST/PROVENANCE.txt — quote it in any results you report."
```

Run it:

```bash
chmod +x tools/fetch-models.sh
./tools/fetch-models.sh "<url you chose in step 1>"
```

- [ ] **Step 3: Write the failing test**

`shared/src/jvmTest/kotlin/app/presence/onnx/OnnxModelSessionTest.kt`:

```kotlin
package app.presence.onnx

import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * These tests need the real model files. Run ./tools/fetch-models.sh first; without them the
 * model-dependent tests skip rather than fail, so the suite stays green on a fresh clone.
 */
class OnnxModelSessionTest {

    private val vadModel = Path.of("../models/silero_vad.onnx")
    private val embeddingModel = Path.of("../models/speaker_embedding.onnx")

    @Test
    fun `reports a missing model file clearly`() {
        val failure = assertFailsWith<IllegalArgumentException> {
            OnnxModelSession(Path.of("../models/not-a-real-model.onnx"))
        }
        assertTrue(failure.message!!.contains("not-a-real-model.onnx"))
    }

    @Test
    fun `prints the VAD signature`() {
        if (!vadModel.exists()) {
            println("SKIP: ${vadModel.toAbsolutePath()} missing — run ./tools/fetch-models.sh")
            return
        }
        OnnxModelSession(vadModel).use { session ->
            println("=== Silero VAD signature ===")
            println(session.describeSignature())
            assertTrue(session.inputNames.isNotEmpty())
            assertTrue(session.outputNames.isNotEmpty())
        }
    }

    @Test
    fun `prints the embedding model signature`() {
        if (!embeddingModel.exists()) {
            println("SKIP: ${embeddingModel.toAbsolutePath()} missing — run ./tools/fetch-models.sh")
            return
        }
        OnnxModelSession(embeddingModel).use { session ->
            println("=== Speaker embedding signature ===")
            println(session.describeSignature())
            assertTrue(session.inputNames.isNotEmpty())
            assertTrue(session.outputNames.isNotEmpty())
        }
    }
}
```

- [ ] **Step 4: Run the test and verify it fails**

Run: `./gradlew :shared:jvmTest --tests "app.presence.onnx.OnnxModelSessionTest"`
Expected: FAIL — unresolved reference `OnnxModelSession`.

- [ ] **Step 5: Write the implementation**

`shared/src/jvmMain/kotlin/app/presence/onnx/OnnxModelSession.kt`:

```kotlin
package app.presence.onnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.file.Path
import kotlin.io.path.exists

/**
 * A loaded ONNX model.
 *
 * [describeSignature] exists because input and output tensor names vary by model and by version
 * — Silero VAD v4 takes separate `h` and `c` state tensors while v5 takes one `state` tensor.
 * Adapters are written against what this prints, never against an assumption.
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

    /** Human-readable dump of every input and output, with shapes. */
    fun describeSignature(): String = buildString {
        appendLine("inputs:")
        session.inputInfo.forEach { (name, info) -> appendLine("  $name: ${info.info}") }
        appendLine("outputs:")
        session.outputInfo.forEach { (name, info) -> appendLine("  $name: ${info.info}") }
    }

    /** Builds a float tensor of the given shape. Caller closes it, or uses [runTensors]. */
    fun floatTensor(data: FloatArray, shape: LongArray): OnnxTensor =
        OnnxTensor.createTensor(environment, java.nio.FloatBuffer.wrap(data), shape)

    /** Runs the model and hands the raw result to [block], closing everything afterwards. */
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

- [ ] **Step 6: Run the test and record the signatures**

Run: `./gradlew :shared:jvmTest --tests "app.presence.onnx.OnnxModelSessionTest" --info`
Expected: PASS, 3 tests. **Copy the printed signatures somewhere you can see them** — Tasks 12 and 13 are written against them.

- [ ] **Step 7: Document the workflow in the README**

Add to `README.md`, just before the `## Attribution` heading:

```markdown
## Running Phase 0

```bash
./tools/list-models.sh                          # see available embedding models
./tools/fetch-models.sh "<chosen model url>"    # download into models/ (gitignored)
./gradlew test                                  # engine + harness unit tests
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
- Produces: `class SileroVad(session: OnnxModelSession, frameSize: Int = 512) : VoiceActivityDetector, AutoCloseable`. Task 16 constructs it.

**Adapt to the signature you recorded in Task 11, Step 6.** The implementation below targets Silero VAD v5, whose signature is three inputs — `input` shaped `[batch, samples]`, `state` shaped `[2, batch, 128]`, and `sr` as an int64 scalar — and two outputs, `output` and `stateN`. **If your dump shows separate `h` and `c` inputs, you have v4:** replace the single state tensor with two `[2, batch, 64]` tensors and carry both across frames. The structure of the class is identical either way; only `run` changes.

Silero at 16 kHz expects exactly 512-sample frames. That is why `Framer` exists.

- [ ] **Step 1: Write the failing test**

`shared/src/jvmTest/kotlin/app/presence/onnx/SileroVadTest.kt`:

```kotlin
package app.presence.onnx

import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.math.PI
import kotlin.math.sin
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Needs models/silero_vad.onnx. Run ./tools/fetch-models.sh first; otherwise these skip so a
 * fresh clone stays green.
 */
class SileroVadTest {

    private val modelPath = Path.of("../models/silero_vad.onnx")

    private fun withVad(block: (SileroVad) -> Unit) {
        if (!modelPath.exists()) {
            println("SKIP: ${modelPath.toAbsolutePath()} missing — run ./tools/fetch-models.sh")
            return
        }
        SileroVad(OnnxModelSession(modelPath)).use(block)
    }

    private fun silence() = FloatArray(512)

    private fun tone(frequencyHz: Double = 220.0) = FloatArray(512) {
        (0.3 * sin(2 * PI * frequencyHz * it / 16_000.0)).toFloat()
    }

    private fun noise(seed: Int) = Random(seed).let { rng -> FloatArray(512) { rng.nextFloat() * 0.02f - 0.01f } }

    @Test
    fun `reports the frame size the model requires`() {
        withVad { vad -> assertTrue(vad.frameSize == 512) }
    }

    @Test
    fun `returns a probability in range for any input`() {
        withVad { vad ->
            listOf(silence(), tone(), noise(1)).forEach { frame ->
                val probability = vad.speechProbability(frame)
                assertTrue(probability in 0f..1f, "probability $probability outside [0,1]")
            }
        }
    }

    @Test
    fun `scores silence lower than a voiced tone`() {
        withVad { vad ->
            // Averaged over several frames: the recurrent state settles after the first few.
            val silenceScore = (1..10).map { vad.speechProbability(silence()) }.average()
            vad.reset()
            val toneScore = (1..10).map { vad.speechProbability(tone()) }.average()

            assertTrue(
                toneScore > silenceScore,
                "a voiced tone ($toneScore) should score above silence ($silenceScore)",
            )
        }
    }

    @Test
    fun `rejects a frame of the wrong size`() {
        withVad { vad ->
            assertFailsWith<IllegalArgumentException> { vad.speechProbability(FloatArray(256)) }
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
                kotlin.math.abs(first - afterReset) < 1e-4f,
                "expected $first after reset, got $afterReset",
            )
        }
    }
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `./gradlew :shared:jvmTest --tests "app.presence.onnx.SileroVadTest"`
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
 * Silero VAD via ONNX Runtime.
 *
 * Written against the v5 signature: inputs `input` [batch, samples], `state` [2, batch, 128],
 * and `sr` (int64 scalar); outputs `output` and `stateN`. If [OnnxModelSession.describeSignature]
 * shows separate `h` and `c` inputs the model is v4 — carry two [2, batch, 64] tensors instead.
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
            val probability = (result.get(0).value as Array<FloatArray>)[0][0]
            @Suppress("UNCHECKED_CAST")
            val nextState = result.get(1).value as Array<Array<FloatArray>>
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

Run: `./gradlew :shared:jvmTest --tests "app.presence.onnx.SileroVadTest"`
Expected: PASS, 5 tests (or all skipped with a printed SKIP if models are absent).

If output shapes do not match, print them and adjust the two casts — the `value` of an ONNX
output is a nested `Array` whose depth equals the tensor rank.

- [ ] **Step 5: Commit**

```bash
git add shared/src/jvmMain/kotlin/app/presence/onnx/SileroVad.kt \
        shared/src/jvmTest/kotlin/app/presence/onnx/SileroVadTest.kt
git commit -m "feat: Silero VAD adapter over ONNX Runtime"
```

---

### Task 13: OnnxSpeakerEmbedder — the real embedding model

**Files:**
- Create: `shared/src/jvmMain/kotlin/app/presence/onnx/OnnxSpeakerEmbedder.kt`
- Test: `shared/src/jvmTest/kotlin/app/presence/onnx/OnnxSpeakerEmbedderTest.kt`

**Interfaces:**
- Consumes: `Embedding`, `SpeakerEmbedder`, `cosine` (Tasks 3, 9); `OnnxModelSession` (Task 11).
- Produces: `class OnnxSpeakerEmbedder(session: OnnxModelSession, inputName: String, outputName: String) : SpeakerEmbedder, AutoCloseable`, with a `dimension` discovered on first use. Task 16 constructs it.

**Again, adapt to your recorded signature.** Most sherpa-onnx speaker models take raw 16 kHz samples shaped `[batch, samples]` and return `[batch, dimension]`. Some expect precomputed fbank features instead — if the input shape has three dimensions, that is what you have, and you must add a filterbank front end. Check before writing.

Input and output names are constructor parameters rather than hardcoded, precisely because they vary between models.

The interesting test here is the last one: **the same speaker's two utterances must score higher against each other than against a different speaker's.** That is the entire product hypothesis in one assertion. It uses synthetic signals, so it proves only that the plumbing is right, not that the model works on children — that is what Phase 0's real corpora are for.

- [ ] **Step 1: Write the failing test**

`shared/src/jvmTest/kotlin/app/presence/onnx/OnnxSpeakerEmbedderTest.kt`:

```kotlin
package app.presence.onnx

import app.presence.engine.cosine
import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class OnnxSpeakerEmbedderTest {

    private val modelPath = Path.of("../models/speaker_embedding.onnx")

    /** Replace with the names printed by OnnxModelSessionTest in Task 11. */
    private val inputName = "feats"
    private val outputName = "embs"

    private fun withEmbedder(block: (OnnxSpeakerEmbedder) -> Unit) {
        if (!modelPath.exists()) {
            println("SKIP: ${modelPath.toAbsolutePath()} missing — run ./tools/fetch-models.sh")
            return
        }
        OnnxSpeakerEmbedder(OnnxModelSession(modelPath), inputName, outputName).use(block)
    }

    /** A crude voice-like signal: a fundamental plus two harmonics. */
    private fun voice(fundamentalHz: Double, seconds: Double = 2.0): FloatArray {
        val count = (seconds * 16_000).toInt()
        return FloatArray(count) { i ->
            val t = i / 16_000.0
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
    fun `reports a plausible dimension`() {
        withEmbedder { embedder ->
            embedder.embed(voice(180.0))
            assertTrue(embedder.dimension in 64..1024, "unexpected dimension ${embedder.dimension}")
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
    fun `separates two different synthetic voices`() {
        withEmbedder { embedder ->
            // Same "speaker", two different utterance lengths.
            val speakerAFirst = embedder.embed(voice(120.0, seconds = 2.0))
            val speakerASecond = embedder.embed(voice(120.0, seconds = 3.0))
            val speakerB = embedder.embed(voice(260.0, seconds = 2.0))

            val sameSpeaker = cosine(speakerAFirst, speakerASecond)
            val differentSpeaker = cosine(speakerAFirst, speakerB)

            assertTrue(
                sameSpeaker > differentSpeaker,
                "same-speaker similarity $sameSpeaker should exceed cross-speaker $differentSpeaker",
            )
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

Run: `./gradlew :shared:jvmTest --tests "app.presence.onnx.OnnxSpeakerEmbedderTest"`
Expected: FAIL — unresolved reference `OnnxSpeakerEmbedder`.

- [ ] **Step 3: Write the implementation**

`shared/src/jvmMain/kotlin/app/presence/onnx/OnnxSpeakerEmbedder.kt`:

```kotlin
package app.presence.onnx

import app.presence.audio.SAMPLE_RATE_HZ
import app.presence.audio.millisToSamples
import app.presence.engine.Embedding
import app.presence.engine.SpeakerEmbedder

/**
 * A speaker-embedding model via ONNX Runtime.
 *
 * Tensor names are constructor parameters rather than constants because they differ between
 * models — check [OnnxModelSession.describeSignature] and pass what it prints.
 *
 * Assumes the model consumes raw 16 kHz samples shaped [batch, samples] and returns
 * [batch, dimension]. If your signature shows a rank-3 input the model wants precomputed
 * filterbank features instead, and a front end has to be added here.
 */
class OnnxSpeakerEmbedder(
    private val session: OnnxModelSession,
    private val inputName: String,
    private val outputName: String,
) : SpeakerEmbedder, AutoCloseable {

    private companion object {
        /** Below this the embedding is dominated by noise (spec §4.1). */
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
            "need at least $MIN_DURATION_MS ms (${minimumSamples} samples) at $SAMPLE_RATE_HZ Hz, " +
                "got ${samples.size}"
        }

        val tensor = session.floatTensor(samples.copyOf(), longArrayOf(1, samples.size.toLong()))

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

- [ ] **Step 4: Fix the tensor names in the test**

Replace `inputName` and `outputName` in the test with the actual names from Task 11 Step 6. If the run fails with an unknown-input error, ONNX Runtime lists the valid names in the exception — use those.

- [ ] **Step 5: Run the test and verify it passes**

Run: `./gradlew :shared:jvmTest --tests "app.presence.onnx.OnnxSpeakerEmbedderTest"`
Expected: PASS, 5 tests.

If `separates two different synthetic voices` fails, do not treat it as a plumbing bug — synthetic
tones are far outside the model's training distribution. Confirm the other four pass, then note it
and move on; Task 16 measures the model on real speech, which is the measurement that counts.

- [ ] **Step 6: Commit**

```bash
git add shared/src/jvmMain/kotlin/app/presence/onnx/OnnxSpeakerEmbedder.kt \
        shared/src/jvmTest/kotlin/app/presence/onnx/OnnxSpeakerEmbedderTest.kt
git commit -m "feat: ONNX speaker embedding adapter"
```

---

### Task 14: Manifest — the corpus metadata format

**Files:**
- Create: `harness/src/main/kotlin/app/presence/harness/Csv.kt`
- Create: `harness/src/main/kotlin/app/presence/harness/Manifest.kt`
- Test: `harness/src/test/kotlin/app/presence/harness/ManifestTest.kt`

**Interfaces:**
- Consumes: `Language`, `VoiceProfile.Role` (Task 4).
- Produces: `object Csv` with `parse(text: String): List<Map<String, String>>`; `enum class RecordingRole { TARGET_CHILD, SIBLING, PARENT, OTHER_HOUSEHOLD, MEDIA, STRANGER, ROOM_TONE }` with `isHouseholdMember: Boolean` and `toProfileRole(): VoiceProfile.Role`; `enum class ContentType { SPONTANEOUS, READ, CALLING, ROOM_TONE }`; `data class Recording(...)`; `data class CallEvent(recordingPath: String, startMs: Long, targetSpeakerId: String)`; `object Manifest` with `readRecordings(path: Path): List<Recording>` and `readEvents(path: Path): List<CallEvent>`. Tasks 15 and 16 consume these.

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

    fun parse(text: String): List<Map<String, String>> {
        val lines = text.lineSequence()
            .withIndex()
            .filter { (_, line) -> line.isNotBlank() }
            .toList()
        require(lines.isNotEmpty()) { "CSV is empty: expected a header row" }

        val header = splitRow(lines.first().value, lineNumber = lines.first().index + 1)

        return lines.drop(1).map { (index, line) ->
            val lineNumber = index + 1
            val fields = splitRow(line, lineNumber)
            require(fields.size == header.size) {
                "line $lineNumber has ${fields.size} fields but the header has ${header.size}: $line"
            }
            header.zip(fields).toMap()
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
    init { require(startMs >= 0) { "event start must not be negative, was $startMs" } }
}

object Manifest {

    private val REQUIRED_RECORDING_COLUMNS = listOf(
        "path", "household_id", "speaker_id", "role", "language", "code_switched",
        "content_type", "distance_m", "placement", "background", "room", "date",
    )

    private val REQUIRED_EVENT_COLUMNS = listOf("recording_path", "start_ms", "target_speaker_id")

    fun readRecordings(path: Path): List<Recording> {
        val rows = Csv.parse(path.readText())
        requireColumns(rows.first().keys, REQUIRED_RECORDING_COLUMNS, path)

        return rows.map { row ->
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

    fun readEvents(path: Path): List<CallEvent> {
        val rows = Csv.parse(path.readText())
        requireColumns(rows.first().keys, REQUIRED_EVENT_COLUMNS, path)

        return rows.map { row ->
            CallEvent(
                recordingPath = row.getValue("recording_path"),
                startMs = row.getValue("start_ms").toLong(),
                targetSpeakerId = row.getValue("target_speaker_id"),
            )
        }
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
Expected: PASS, 15 tests.

- [ ] **Step 6: Commit**

```bash
git add harness/src/main/kotlin/app/presence/harness/Csv.kt \
        harness/src/main/kotlin/app/presence/harness/Manifest.kt \
        harness/src/test/kotlin/app/presence/harness/ManifestTest.kt
git commit -m "feat: corpus manifest format per spec 9.1"
```

---

### Task 15: Metrics — the numbers that decide go/no-go

**Files:**
- Create: `harness/src/main/kotlin/app/presence/harness/Metrics.kt`
- Test: `harness/src/test/kotlin/app/presence/harness/MetricsTest.kt`

**Interfaces:**
- Consumes: `CallEvent` (Task 14).
- Produces: `data class Detection(recordingPath: String, timestampMs: Long)`; `enum class Verdict { SHIP, ITERATE, FALL_BACK }`; `data class Outcome(label: String, fireThreshold: Float, falseAlarms: Int, negativeHours: Double, detectedEvents: Int, totalEvents: Int, latenciesMs: List<Long>)` with computed `falseAlarmsPerHour`, `missRate`, `medianLatencyMs`, `verdict`; `object Metrics` with `scoreEvents(detections, events, latencyBudgetMs = 10_000): EventScoring` where `data class EventScoring(detected: Int, total: Int, latenciesMs: List<Long>)`. Task 16 uses all of it.

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
    val falseAlarmsPerHour: Double
        get() = if (negativeHours <= 0.0) 0.0 else falseAlarms / negativeHours

    val missRate: Double
        get() = if (totalEvents == 0) 0.0 else (totalEvents - detectedEvents).toDouble() / totalEvents

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
Expected: PASS, 18 tests.

- [ ] **Step 5: Commit**

```bash
git add harness/src/main/kotlin/app/presence/harness/Metrics.kt \
        harness/src/test/kotlin/app/presence/harness/MetricsTest.kt
git commit -m "feat: Phase 0 metrics with spec 4.3 verdict thresholds"
```

---

### Task 16: Evaluation runner, report and CLI

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

        for (trace in selected) {
            val accumulator = EvidenceAccumulator(AccumulatorConfig(fireThreshold = threshold))
            for (observation in trace.observations) {
                if (accumulator.accept(observation) != null) {
                    detections += Detection(trace.recording.path, observation.timestampMs)
                }
            }
            if (trace.recording.role != RecordingRole.TARGET_CHILD) {
                negativeSeconds += trace.durationSeconds
            }
        }

        val relevantEvents = events.filter { it.recordingPath in selectedPaths }
        val scoring = Metrics.scoreEvents(detections, relevantEvents)
        val falseAlarms = detections.count { detection ->
            selected.first { it.recording.path == detection.recordingPath }
                .recording.role != RecordingRole.TARGET_CHILD
        }

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

    fun markdown(outcomes: List<Outcome>, provenance: String): String = buildString {
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
Expected: PASS, 11 tests.

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
```

- [ ] **Step 9: Verify the CLI runs**

Run: `./gradlew :harness:run --args="--help"`
Expected: the usage text prints.

- [ ] **Step 10: Run the full suite**

Run: `./gradlew test`
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

Spec §9 lists five things Phase 0 must settle. Three are covered by Task 16's sweep — cross-lingual
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

Also worth knowing while running the sweep: **Task 16 sweeps the fire threshold, not the
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

