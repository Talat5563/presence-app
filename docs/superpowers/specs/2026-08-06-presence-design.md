# Presence — Design Specification

**Date:** 2026-08-06
**Status:** Draft, pending user approval
**Working name:** Presence (placeholder — rename freely; it appears only in docs, not code, at this stage)

---

## 1. The problem

A parent is on their phone. Their child says their name, asks a question, tries to show them
something — and it doesn't land, because the phone has the parent's attention.

Presence listens for *that specific child's voice* and softly redirects the parent's attention
back to them. It is not a screen-time app, not a usage limiter, and not a shaming tool. It
intervenes at exactly one moment: when the child who matters is speaking and the parent isn't
listening.

### Non-goals

- Monitoring or restricting the child's device. This app runs on the *parent's* phone.
- General screen-time reduction, app blocking by category, or usage quotas.
- Recording, transcribing, or understanding what anyone says. The app answers one question —
  *"is that my child's voice?"* — and nothing else.
- Detecting crying, distress, or generic child speech. Speaker identity is the trigger.

---

## 2. Product decisions (settled)

| Decision | Choice |
|---|---|
| Detection target | The parent's own child, by speaker identity |
| Default response | Dim the screen + mute audio + show the child's photo |
| Response configurability | Fully user-configurable per child |
| Listening window | Only while the phone is in use; optionally narrowed further to family-time schedules |
| Data handling | 100% on-device. No account, no server, no upload |
| Enrollment | Two modes — guided live recording, or import existing recordings |
| Accuracy correction | One-tap "not them" on the intervention screen |
| Household enrollment | Parent and siblings enrolled as non-target profiles — required, see §5 |
| Decision rule | Relative likelihood across the household, accumulated over segments — not an absolute per-utterance threshold |
| Metrics of record | False alarms per hour, and miss rate per real event. Not EER |
| Model strategy | Fine-tune on children's speech offline; ship the improved model in the app |
| Languages | English and Hebrew at first release. Bilingual enrollment is mandatory — see §4.2 |
| Layout | Bidirectional (LTR + RTL) from the first screen, not retrofitted |
| Codebase | Kotlin Multiplatform shared core, native UI per platform |
| Platform order | Android first, iOS second |

### Listening rule, stated precisely

**On Android**, the microphone is active only when the screen is on and unlocked. If the parent
enables family-time windows, the mic is active only when the screen is on *and* the current time
falls inside a window. The mic is never active while the screen is off. This is a hard invariant,
not a setting — an "always listening" option is deliberately absent.

**On iOS the invariant cannot be enforced in Tier 1**, because a backgrounded iOS app has no way to
read screen-on state (§6). The consequence is a deliberate divergence rather than a silent one:

| | Screen-on gating | Family-time windows |
|---|---|---|
| Android | Enforced | Optional |
| iOS Tier 1 (no entitlement) | **Not possible** | **Mandatory** — the only bound on listening |
| iOS Tier 2 (Family Controls) | Enforced via `DeviceActivityMonitor` | Optional |

So on iOS Tier 1 the mic may be active while the screen is off, within a window the parent
explicitly configured. Configuring at least one window is therefore a requirement to enable the app
on iOS Tier 1, and onboarding must state the difference plainly instead of implying Android's
guarantee. This is the strongest argument for pursuing the Tier 2 entitlement: it is what makes the
privacy story identical across platforms.

---

## 3. Architecture

The organizing principle: **the sensing engine is pure, portable logic; everything touching the OS
is a thin, swappable adapter.** The engine is the hard and valuable part; the OS integrations are
the parts that differ per platform. Splitting on that seam is what makes the multiplatform effort
pay for itself.

```
:shared  (Kotlin Multiplatform — commonMain)
  domain/
    ChildProfile          name, photo reference, enabled state
    VoiceProfile          embedding centroid + negative examples + threshold
    DetectionEvent        timestamp, childId, confidence, outcome
    InterventionConfig    intent level + per-channel toggles
    ListeningSchedule     day/time windows
  engine/
    DetectionEngine       pure state machine: audio frames in → DetectionEvent out
    VadGate               speech / not-speech
    EmbeddingMatcher      cosine similarity vs enrolled profiles
    Debouncer             hysteresis, cooldown, consecutive-hit logic
  enrollment/
    EnrollmentService     clips in → VoiceProfile out (shared by both enrollment modes)
    ClipQualityCheck      duration, SNR, clipping, speech-presence validation
  data/
    SQLDelight            profiles, events, settings
  platform/  (expect)
    AudioCapture          start/stop, emits 16 kHz mono Float frames
    ModelRunner           ONNX inference
    SecureStore           encrypted profile storage
    PhotoStore            child photos

:androidApp
  Compose UI
  ListeningService        foreground service, type=microphone
  ScreenStateReceiver     ACTION_SCREEN_ON / OFF / USER_PRESENT
  OverlayIntervention     TYPE_APPLICATION_OVERLAY — dim scrim + photo card
  AudioDucker             AudioManager / DND policy
  PermissionFlow          staged onboarding
  actual: AudioRecord, ONNX Runtime Android, EncryptedFile

:iosApp  (phase 2)
  SwiftUI
  ListeningSession        AVAudioSession + audio background mode
  LiveActivityIntervention  Dynamic Island / Lock Screen
  NotificationIntervention  Time-Sensitive, photo attachment
  ShieldIntervention        ManagedSettings (Tier 2 only)
  actual: AVAudioEngine, ONNX Runtime iOS, Keychain + Data Protection
```

`DetectionEngine` knows nothing about microphones, services, or screens. It consumes frames and
emits events. Consequence: **the entire recognition pipeline is unit-testable on the JVM against
recorded WAV fixtures** — no device, no emulator. For an app whose principal risk is accuracy
tuning, that is the single most important structural decision in this document.

---

## 4. The detection pipeline

```
mic (16 kHz mono, ring buffer)
  │
  ├─> VadGate — is anyone speaking at all?        [Silero VAD class, ~1 MB, runs continuously]
  │     └─> (silence: stop here — ~95% of frames)
  │
  ├─> Channel A: wake-phrase fast path            [optional, high precision]
  │     personalized keyword ("Mummy", "Abba", a name)
  │       └─> text-dependent speaker check on that phrase
  │             └─> high-confidence LLR contribution, fires fast
  │
  ├─> Channel B: text-independent speaker ID      [always on, lower per-segment confidence]
  │     embed the speech segment                  [~20 MB, speech only]
  │       └─> score vs EVERY enrolled household member (target, siblings, parent)
  │             └─> normalize against the household cohort
  │                   └─> relative log-likelihood ratio, not an absolute threshold
  │
  ├─> EvidenceAccumulator — sum LLR across segments in a rolling window
  │     fires when accumulated evidence crosses the operating point
  │     then enters cooldown
  │
  └─> DetectionEvent → InterventionRenderer
```

### Why embeddings rather than on-device training

A pre-trained speaker-embedding model turns any speech clip into a fixed-length vector.
Enrollment averages the vectors from the child's clips into a stored centroid. At runtime each
speech segment is embedded and compared by cosine similarity; above threshold, it's them.

Nothing is trained on the phone — enrollment is arithmetic over embeddings. That is why it
completes in seconds, works identically for live-recorded and imported audio, and supports
multiple children for free (one profile each). The similarity threshold becomes a user-facing
sensitivity slider.

Rejected alternatives: on-device few-shot classifier training (needs far more data than a parent
will supply, drifts as the child grows, no accuracy benefit); generic child-vs-adult voice
detection (no enrollment needed, but fires on the television, on visiting children, and on the
parent's own home videos).

### The two gates that make it livable

- **The VAD gate is what makes battery cost survivable.** The expensive embedding model wakes only
  for actual speech — roughly a 5% duty cycle in a normal room.
- **The evidence accumulator is what makes the app *accurate*, not merely tolerable.** See §4.1: it
  is the primary accuracy mechanism in this design, not a debouncing convenience.

---

## 4.1 Accuracy strategy

This section exists because the published numbers for children's speaker verification are bad
enough to kill a naive implementation, and because the reasons they are bad do not fully apply to
this product. Both halves matter.

### The honest baseline

An adult-trained ECAPA-TDNN applied to children with no adaptation ("zero-resource") measures:

| Condition | EER |
|---|---|
| CSLU Kids, scripted speech | 14.5% (boys) / 16.7% (girls) |
| MyST, spontaneous conversational speech | ~31% |

Spontaneous conversational speech is our condition. **~31% EER is unusable as a per-utterance
decision.** The ChildAugment authors additionally discarded single-word utterances — averaging 0.92
seconds — as too short to evaluate meaningfully. "Daddy!" is a single word.

Causes are anatomical and well established: children have higher fundamental and formant
frequencies from shorter vocal tracts, and the tract changes shape rapidly during growth, producing
much wider intra- and inter-speaker variation than adult speech.

### Why our problem is easier than the benchmark

Benchmark speaker verification is deliberately hard in ways this product is not: one short
utterance, an unknown open-set impostor population, an absolute threshold, a single shot. Presence
has four structural advantages, and the design must exploit all four.

**1. It is a closed household roster, not open-set verification.**
Enroll the parent and the siblings alongside the target child. The runtime question becomes *"which
household member does this most resemble"* — a relative comparison rather than an absolute
threshold. Sibling-vs-target is the hardest confusion in real households, and relative scoring
attacks it directly. Enrolled family members double as the impostor cohort that adaptive score
normalization needs and normally has to approximate from a generic set.

**2. Evidence accumulates across segments.**
No decision needs to rest on one 0.9-second word. A child seeking attention talks for tens of
seconds. Per-segment scores become log-likelihood-ratio contributions summed over a rolling window;
the decision fires when accumulated evidence crosses the operating point. The literature is
consistent that error falls as effective duration rises, and that segmenting long audio and
averaging embeddings is the standard way to exploit it. **Consequence: the accumulator is the
primary accuracy mechanism, not a UX nicety.** An earlier draft of this spec treated it as a
debouncer that counted hits; that was a design error.

**3. Error costs are asymmetric, so EER is the wrong metric.**
A false accept shows a photo the parent dismisses with one tap. A false reject misses a moment.
Neither is a security breach. The system should operate far from the equal-error point, trading
false alarms for recall. **The metrics of record are false alarms per hour of listening and miss
rate per real attention-seeking event.** EER appears in this spec only to compare against published
work; it is not a product target.

**4. Text-dependent verification is far more accurate on short utterances**, because the phonetic
content is known — the literature reports text-dependent systems outperforming text-independent
ones and tolerating much shorter enrollment. The most common real event is the child calling the
parent by name, which makes a high-precision second channel available cheaply.

### The accuracy ladder

| | Technique | Cost | Payoff | In v1? |
|---|---|---|---|---|
| **L0** | Pre-trained embedding, absolute threshold | none | 14–31% EER — unusable alone | baseline only |
| **L1** | Household cohort, relative scoring, score normalization | plain code | Large; attacks sibling confusion | **Yes** |
| **L2** | LLR accumulation across segments | plain code | Large; turns weak signal into confident decisions | **Yes** |
| **L3** | Fine-tune on children's speech; ChildAugment on adult data | days–weeks, one-time, offline | ~10× — the big lever | **Yes** (offline) |
| **L4** | Per-household adaptation: negative examples, drift re-enrollment | small | Moderate; essential over months | **Yes** |
| **L5** | Text-dependent wake-phrase channel | moderate | High precision on the most common event | Stretch |

L1 and L2 cost nothing but design thought. L3 is a one-time offline model-training cost that never
touches runtime complexity — it ships as a better model file.

### L3 in detail: fine-tuning

Fine-tuning the same ECAPA-TDNN architecture on children's speech reaches **2.9% EER** on Samrómur
Children — roughly a tenfold improvement over the zero-resource baseline, and the reason this
product is viable at all.

**Data.** [Samrómur Children](https://huggingface.co/datasets/language-and-voice-lab/samromur_children)
is **CC BY 4.0** on Hugging Face: 131 hours, 137,597 utterances, 3,175 speakers aged 4–17.
Commercial use is permitted with attribution. Two caveats to validate rather than assume:

- It is **Icelandic**, and our deployment language differs. Speaker embeddings transfer across
  languages substantially better than ASR does, but this must be measured, not asserted.
- It is **read** speech, while our runtime condition is **spontaneous**. This is the same mismatch
  that separates the 14% and 31% figures above, so it is the more serious of the two caveats.

**Supplementary augmentation (ChildAugment).** Adult training data can be warped to emulate
children's vocal tracts, yielding roughly 12% relative EER improvement on its own — useful as a
supplement, not a substitute for real child data. Published parameter ranges:

| Method | Parameters |
|---|---|
| Speed / pitch modification | α ∈ [0.9, 1.1] |
| VTLP | α ∈ [0.9, 1.1] |
| LPC-WP | α ∈ [0.7, 1.3] |
| LPC-SWP (formant-specific warping) | α₁ ∈ [0.6, 0.85], hierarchically constrained thereafter |
| BWP-FEP (bandwidth / energy perturbation) | β₁₋₄ ∈ [0.9, 1.1] |
| Application-agnostic | SpecAugment, additive noise, RIR |

**Toolchain.** WeSpeaker trains in PyTorch and exports to ONNX by design, which is exactly this
pipeline. [sherpa-onnx](https://k2-fsa.github.io/sherpa/onnx/speaker-identification/) already ships
WeSpeaker, 3D-Speaker, and NeMo embedding models as ONNX with a Kotlin API and working Android
speaker-identification demo APKs — so the fine-tune → export → on-device path is a well-trodden
one rather than novel integration work.

### L4 in detail: age drift

A child's voice changes as they grow, and this degrades a stored profile measurably. ICASSP 2025
work combining a Feature Transform Adapter with synthetic audio augmentation reduced EER by 19.4%
at a one-year age gap, 13.0% at two years, and **6.1% at three years**.

The declining returns are the finding that matters: **no technique solves drift.** Therefore:

- Re-enrollment is a first-class product feature, prompted on a schedule (younger children more
  often), not an obscure settings action.
- Profiles record their enrollment date; the UI surfaces staleness before accuracy visibly rots.
- The "not them" button stores rejected embeddings as negative examples, so a drifting profile
  degrades toward *silence* rather than toward *firing on a sibling*.
- Multiple enrollment sessions over time are retained and merged rather than overwritten, giving
  the profile a wider view of the child's voice.

### Build-versus-buy

[Picovoice Eagle](https://picovoice.ai/platform/eagle/) is a commercial on-device speaker
recognition engine covering Android and iOS, claiming meaningfully lower EER than SpeechBrain.
Worth benchmarking in Phase 0 as a baseline and a fallback. It is rejected as the primary path for
two reasons: it is proprietary with per-seat licensing, and **it cannot be fine-tuned on children's
speech** — which forfeits L3, the single largest lever available. If Phase 0 shows our fine-tuned
model losing to Eagle out of the box, that conclusion should be revisited honestly.

---

## 4.2 Bilingual operation — English and Hebrew

First release targets **English and Hebrew**. This is not a localization detail; it changes the
recognizer's design in three places.

### Language mismatch is a first-order accuracy problem

Cross-lingual speaker verification degrades **40–49%** when enrollment and test utterances are in
different languages. Critically, a bilingual *same-speaker* evaluation found that language mismatch
remains the main driver of that loss even with speaker identity held constant — it is domain shift
from the language itself, not merely inter-speaker variability.

The implication for a Hebrew/English household is direct and severe: **a child enrolled in Hebrew and
then heard speaking English is close to a different-domain trial.** Left unaddressed this would
plausibly cost more accuracy than everything L1 and L2 recover.

**Therefore bilingual enrollment is mandatory, not optional:**

- The guided enrollment session prompts in **both** languages, and does not complete until it has
  collected sufficient usable speech in each.
- `VoiceProfile` stores **per-language sub-centroids** rather than one pooled centroid, since pooling
  across a domain shift this large blurs the profile rather than broadening it. Scoring takes the
  best-matching sub-profile.
- Imported recordings are language-tagged, and the import flow reports per-language coverage so the
  parent can see which language is short.
- Households that are genuinely monolingual can enroll in one language; the app then states plainly
  that detection in the other language will be unreliable, rather than silently underperforming.
- Code-switching mid-sentence is normal in bilingual homes and is the untested case. Phase 0 must
  measure it explicitly (§9.1).

### Training data: Hebrew has no children's corpus

There is **no public Hebrew children's speech corpus**. This is the single hardest data gap in the
project, and it is why user-supplied recordings (§9.1) are load-bearing rather than merely helpful.

The L3 training mix therefore combines three sources:

| Source | Provides | Licence |
|---|---|---|
| [Samrómur Children](https://huggingface.co/datasets/language-and-voice-lab/samromur_children) | Real child speech at scale — teaches the model *child vocal-tract characteristics* generally | CC BY 4.0 |
| [ivrit.ai](https://arxiv.org/abs/2307.08720) (3,300+ h Hebrew adult speech), warped with ChildAugment | Hebrew-language *phonetics* with child-like vocal-tract parameters | Free; verify terms |
| User-supplied household recordings (§9.1) | Real bilingual children, real rooms — the evaluation set, and possibly a small fine-tuning set | Consent-gated, never public |

The reasoning behind the split: **what transfers across languages and what does not are different
things.** Learning "what a child's voice looks like in embedding space" generalizes from Icelandic
perfectly well — that is why Samrómur is usable despite the language gap. Learning Hebrew phonetic
realizations does not transfer, which is what the ChildAugment-over-ivrit.ai path supplies. Phase 0
must verify this reasoning rather than assume it; it is a hypothesis, not a result.

### The wake-phrase channel becomes language-agnostic

L5 originally looked expensive because a trained wake-word model is language-specific, and a Hebrew
one would have to be built. **Query-by-Example keyword spotting removes that problem entirely:** the
user enrolls the keyword by recording a few examples, and detection is acoustic similarity to those
recordings. No Hebrew model, no per-language work — *"אבא"*, *"אמא"*, *"Daddy"*, *"Mummy"*, or a
nickname all work identically.

Two consequences:

1. **L5 is promoted from stretch goal to a v1 candidate**, reassessed after Phase 0.
2. **The recordings serve double duty.** Asking the child to call the parent a few times the way they
   normally do produces both QbyE keyword templates *and* speaker-ID enrollment audio of exactly the
   utterance type that matters most. One recording session, two purposes, and a more natural
   enrollment prompt than reading a word list.

### Bidirectional UI

Hebrew is right-to-left. The UI must be bidirectional from the first screen — layout mirroring,
Hebrew typography, and mixed-direction strings, since a child's name may be in either script while
the surrounding sentence is in the other. Retrofitting RTL is far more expensive than designing for
it, and the intervention overlay — the most layout-sensitive screen in the app — is the one that
would break worst.

---

## 4.3 Go / no-go criteria for Phase 0

Measured on the household-simulation set (§9.1), with L1 and L2 active:

- **Ship:** ≤ 1 false alarm per 2 hours of active listening, at ≥ 80% detection of real
  attention-seeking events within 10 seconds.
- **Iterate:** ≤ 2 false alarms per hour **and** ≥ 50% detection — both required. Either dimension falling below its floor means fall back, however good the other is: 5 false alarms an hour is unusable whatever the detection rate, and 40% detection is useless however quiet.
- **Fall back:** worse than that after L3. The fallback is Channel A alone — a personalized
  wake-phrase detector with a speaker check, which is a narrower but genuinely useful product, and
  a far more honest one than a text-independent detector that cries wolf.

**Both languages must clear the bar independently.** A system that hits the target in English and
misses it in Hebrew has not passed — it has one working language and one that will erode the parent's
trust in the whole app. Report every criterion separately for English, Hebrew, and code-switched
speech.

---

## 5. Enrollment

Both modes converge on the same `EnrollmentService`; only clip acquisition differs.

**Mode A — guided live setup.** The child repeats a handful of short prompts, targeting ~60 seconds
of usable speech **per language** — so a bilingual enrollment is roughly two minutes of usable
speech in total, since §4.2 makes both languages mandatory unless the household is monolingual.
The UI shows live waveform feedback and per-clip quality results. Best quality, requires the
child's cooperation for a couple of minutes.

**Mode B — import existing recordings.** The parent selects existing videos or voice notes. Audio
is extracted, VAD-segmented, and the parent confirms which segments are the child. No cooperation
needed, but quality varies and some imports will be rejected.

`ClipQualityCheck` gates both: minimum total speech duration, adequate signal-to-noise, no
clipping, single dominant speaker. Rejected clips give a specific reason ("too much background
noise", "we couldn't hear speech here") rather than a generic failure.

**Duration is not a detail.** Published results show error falling as enrollment duration rises, and
utterances under about a second being near-worthless. `ClipQualityCheck` therefore enforces a
minimum *aggregate speech* duration rather than a minimum clip count, and the UI reports progress in
seconds of usable speech collected, not clips recorded.

### Household enrollment — required, not optional

This is a direct consequence of §4.1. **Enrolling only the target child is a design error.** The
parent and any siblings are enrolled too, as labelled non-target profiles, because:

1. Relative scoring across the household is what makes sibling discrimination tractable.
2. Enrolled family members form the impostor cohort for score normalization.
3. The parent's own voice is the single most common non-target voice the microphone will hear, since
   the phone is in their hand.

Non-target enrollment is deliberately lighter than target enrollment — shorter sessions, and import
is fine — because these profiles only need to be good enough to *subtract*. Onboarding must frame
this honestly and in the parent's terms: *"Record yourself and their brother or sister too. It's how
we learn to tell them apart."* A parent who understands the reason will do it; one who is asked to
record three people for no stated reason will abandon setup.

**Television and media audio** are the other major false-alarm source, and cannot be enrolled the
same way — the voices are unbounded. Mitigations available without new machinery: media playback is
usually detectable through platform audio state, and detections that coincide with active media
output can be scored more conservatively. Worth measuring in Phase 0 before building anything
special for it.

---

## 6. Intervention

`InterventionConfig` declares **intent**, not mechanism:

- `Notify` — least intrusive. A notification with the child's photo.
- `Soften` — the default. Make the phone less compelling: dim, mute/duck, show their face.
- `Block` — hard stop until acknowledged.

Each platform renders the intent at the strongest level it is permitted to.

### Android rendering

| Channel | Mechanism | Permission |
|---|---|---|
| Dim | Semi-transparent overlay scrim | `SYSTEM_ALERT_WINDOW` |
| Photo | Card in the same overlay window | `SYSTEM_ALERT_WINDOW` |
| Mute | `AudioManager` volume + DND | Notification policy access |
| Notify | Standard notification | `POST_NOTIFICATIONS` |
| Block | Full-opacity overlay, dismiss required | `SYSTEM_ALERT_WINDOW` |

Dim and photo share one overlay window — one component, not two.

**Forcing the display to grayscale is not achievable.** It requires `WRITE_SECURE_SETTINGS`, which
is ADB/root-only and unavailable to installed apps. Dimming via overlay scrim replaces it.

**The green microphone indicator cannot be suppressed.** Android shows it for any microphone
access. Onboarding addresses it directly rather than hoping it goes unnoticed: *"You'll see a green
dot when we're listening. We never record or save anything."*

### iOS rendering

iOS has no equivalent of `SYSTEM_ALERT_WINDOW` and no way to dim the screen from the background.
The product is delivered through three native mechanisms instead, split across two tiers.

**Tier 1 — no special entitlement, ships immediately:**

| Channel | Mechanism |
|---|---|
| "Mute" | `AVAudioSession` with `.duckOthers` — actively ducks the video or podcast playing |
| Photo | **Live Activity** — the child's face in the Dynamic Island / Lock Screen, over any app |
| Notify | Time-Sensitive local notification with photo attachment; pierces Focus modes |
| Dim | Not available |
| Block | Not available |

Ducking replaces muting, and the Dynamic Island replaces the overlay. This is a close translation
of the Android experience, not a degraded stub.

**Tier 2 — requires Apple's Family Controls entitlement:**

`ManagedSettingsStore.shield.applications` blocks the app the parent is currently in. Requesting
the entitlement unlocks two things simultaneously:

1. The `Block` intent becomes real on iOS.
2. `DeviceActivityMonitor` becomes available — **iOS's only viable signal for "the parent is using
   their phone right now."**

Point 2 matters more than point 1. Without the entitlement, a backgrounded iOS app cannot detect
screen-on state at all, so Tier 1 must fall back to **schedule-based listening only**. Tier 2
restores full parity with Android's "only while I'm using my phone" rule.

### Platform risks on record

- Background audio for an app that is not a media app attracts App Store review scrutiny. The
  review notes must make the case explicitly.
- Apple grants Family Controls selectively. A self-directed attention app is a plausible grantee
  but not a certainty. **Tier 1 must stand alone as a shippable product**, and the design assumes
  it does.
- On iOS, an incoming call or another app claiming the audio session interrupts listening. The
  session must be re-established on interruption-ended.

---

## 7. Permissions and onboarding

Android requires four separate trips into system Settings before every feature works. Presenting
that as a wall on first launch would lose most users.

| Permission | Enables | Friction |
|---|---|---|
| `RECORD_AUDIO` | Everything | Standard prompt |
| `FOREGROUND_SERVICE_MICROPHONE` | Mic while in another app | Manifest only |
| `POST_NOTIFICATIONS` | Required service notification | Standard prompt |
| `SYSTEM_ALERT_WINDOW` | Dim + photo overlay | Settings trip |
| Notification policy access | Muting / DND | Settings trip |
| Battery optimization exemption | Service survival | Settings trip |

**Onboarding is staged, not front-loaded.** Microphone plus notifications produce a working
notify-level app on day one. Overlay and DND are offered later, in context, as upgrades to the
response — each requested at the moment the parent configures the feature that needs it. The
settings screen shows which response channels are unlocked and what each still needs.

---

## 8. Error handling and failure modes

| Failure | Handling |
|---|---|
| Mic permission revoked while running | Service stops, persistent notification explains, one-tap re-grant |
| Foreground service killed by OEM battery manager | Restart on next screen-on; if repeated, surface the OEM-specific allowlist instructions |
| Overlay permission revoked | Response silently degrades to `Notify`; settings shows the downgrade |
| Model fails to load | App runs in a disabled state with a clear error; never silently pretends to listen |
| Enrollment clips all rejected | Specific per-clip reasons, offer to switch enrollment mode |
| Phone call active | Listening suspends; audio focus is never contested during a call |
| Another app holds exclusive mic | Detected and surfaced; listening resumes when released |
| No child profile enrolled | App cannot be enabled; onboarding routes to enrollment |
| iOS audio session interrupted | Re-establish on `interruptionEnded`; log gaps for diagnostics |

**Cross-cutting rule: the app never claims to be listening when it is not.** Every degraded state
is visible in the UI. A parent trusting a silent, broken app is worse than no app.

---

## 9. Testing strategy

### Phase 0 — accuracy spike, before any UI

Nothing else should be built first. Phase 0 evaluates **the system, not the model** — with household
cohort scoring (L1) and evidence accumulation (L2) active — because measuring a bare pre-trained
embedding against an absolute threshold would reproduce the ~31% EER from the literature and tell us
nothing we do not already know.

It reports the **product metrics from §4.1** — false alarms per hour of listening, and miss rate per
real attention-seeking event within a 10-second latency budget — as a curve across operating points,
not a single number. EER is computed only to compare against published work.

**Evaluation sets, in increasing order of realism:**

1. **Samrómur Children** (CC BY 4.0) — a large, clean, public baseline for child-vs-child
   discrimination. Establishes whether the model works at all, and quantifies the fine-tuning gain.
2. **Household simulation** — target child, sibling, parent, and television audio mixed at realistic
   room levels and distances. This is the set the go/no-go criteria are measured on. It is what
   Samrómur cannot tell us, because the benchmark contains no siblings and no TV.
3. **Real household recordings** — small, consented, and never committed to a public repository
   (§11). The only set that exposes spontaneous-speech and room-acoustic reality.

**Phase 0 must also settle these, since each is load-bearing for the plan:**

- Zero-resource baseline versus fine-tuned model, on the same sets — does L3 deliver its promised
  order of magnitude here, or only on Icelandic read speech?
- How far the Icelandic-to-target-language and read-to-spontaneous mismatches actually cost.
- Picovoice Eagle benchmarked as an out-of-the-box baseline (§4.1, build-versus-buy).
- The accumulation window length and cooldown that hit the latency budget — a longer window is more
  accurate and slower, and this trade is the core tuning decision of the product.
- Whether media-playback state is a usable signal for suppressing television false alarms.
- **Cross-lingual performance (§4.2):** enrol-Hebrew / test-English and the reverse, versus
  matched-language, versus bilingual enrollment with per-language sub-centroids. This quantifies the
  40–49% degradation risk on our actual data and validates the mitigation.
- **Code-switching**, which no benchmark covers and every bilingual household does.
- Whether the ChildAugment-over-ivrit.ai path (§4.2) measurably beats Samrómur alone on Hebrew.

---

## 9.1 Data collection protocol for supplied recordings

Real household recordings are the only way to evaluate the conditions that decide this product, and
for Hebrew they are the only child data that exists at all. This section specifies what to collect so
the effort is not wasted — a re-collection because the metadata was thin is expensive and annoying.

### Consent — settle this before recording anything

Recording children creates real obligations. Written parental consent per child, stating what is
recorded, what it is used for, where it is stored, how long it is kept, and how to withdraw. If the
data will ever train a shipped model rather than only evaluate it, say so explicitly in the consent —
retrofitting that permission later is not possible. Recordings never enter the public repository
(§11) and are stored encrypted with access limited to named people.

### Who

Target **5–10 households** if achievable; even 3 is far better than none. Per household:

- **1–2 target children.** Spread ages across roughly 3–12; younger voices are the hard case and are
  worth over-sampling.
- **Every sibling**, since sibling-vs-target is the hardest confusion the product faces.
- **Both parents** where applicable — the parent's voice is the one the microphone hears most.

### What, per speaker

Roughly **5 minutes of usable speech**, split deliberately rather than recorded in one block:

| Content | Why |
|---|---|
| Spontaneous conversation, **Hebrew** | The realistic condition; read speech flatters the recognizer |
| Spontaneous conversation, **English** | Required to measure and fix cross-lingual mismatch (§4.2) |
| **Code-switched** speech, both languages in one utterance | Normal in bilingual homes, absent from every benchmark |
| The child **calling the parent** — *"אבא!"*, *"Mummy!"* — repeated naturally, ~10× | The single most important event type; also QbyE keyword templates (§4.2) |
| A short **read passage** in each language | Comparability with published benchmarks |

### Conditions — vary them on purpose

Recognition failures come from acoustics, not just voices. Capture the same speaker across:

- **Distance:** ~1 m and ~3 m from the phone.
- **Phone placement:** on a table, and in a pocket or face-down.
- **Background:** quiet; television on; kitchen or general household noise.
- **Rooms:** at least two, ideally with different reverberation.

### Negative material — as valuable as the voices

- **Television and streaming audio** in Hebrew and English, including children's programming, which
  is the worst adversary because it *is* children's voices.
- **Unrelated children** — visitors, cousins, a playground — as open-set impostors.
- **Silence and room tone** for each room, to calibrate the VAD.

### Longitudinal follow-up

Re-record the same children **after 3–6 months**. Age drift is a documented, measurable degradation
(§4.1, L4), and without a second timepoint the re-enrollment policy is guesswork. This costs almost
nothing at collection time and cannot be recovered retroactively.

### Format and labelling

- **16 kHz or higher, mono, uncompressed WAV.** Never lossy-compressed — codec artefacts corrupt
  exactly the spectral detail speaker embeddings rely on.
- Record with a phone, not a studio microphone. The target device's own hardware is the deployment
  condition.
- Per file: speaker ID (pseudonymous), age, sex, household ID, language, code-switched yes/no,
  content type, distance, phone placement, background condition, room, and date. Date matters — it is
  what makes the longitudinal comparison possible.

### Held-out discipline

Split by **household**, never by utterance. Evaluating on a household whose voices appeared in
training inflates results dramatically and silently. At least two households must be held out and
touched only for final measurement.

- **Engine unit tests (JVM, no device):** `DetectionEngine`, `VadGate`, `EmbeddingMatcher`,
  `Debouncer` against WAV fixtures. Deterministic, fast, the bulk of the suite.
- **Fixture corpus:** target child, sibling, parent, television audio, background noise, silence.
  Committed only with consent, and never real family audio in a public repository — see §11.
- **Enrollment tests:** clip quality acceptance and rejection, centroid math, negative examples.
- **Android instrumentation:** service lifecycle across screen on/off, permission revocation
  mid-run, overlay rendering, DND acquisition and release.
- **Battery measurement:** a measured target, not an aspiration. Budget defined during planning.
- **Manual field testing:** real households, since the failure modes that matter — siblings, TV,
  video calls — do not reproduce on a desk.

---

## 10. Open questions for planning

1. Battery budget target: what percentage per hour of screen-on listening is acceptable?
2. Specific model selection and license verification for the VAD and embedding models. sherpa-onnx
   redistributes WeSpeaker, 3D-Speaker, and NeMo models under their respective upstream licenses —
   each candidate must be checked individually before it ships, not assumed permissive.
3. Multi-child support in v1, or single child with the schema ready for more? Note that §5 requires
   multi-*profile* support regardless, since siblings and the parent are enrolled as non-targets — so
   the marginal cost of multi-child is now small.
4. Minimum Android API level.
5. Distribution intent: personal use, open source, or App/Play Store release? This determines
   whether the Family Controls entitlement request and store-review preparation are in scope.
**Resolved 2026-08-07:** deployment languages are **English and Hebrew** (see §4.2), and
**real child recordings will be supplied** by the project (collection protocol in §9.1). These
answers raised the following in their place:

6. **Consent scope for supplied recordings** — evaluation only, or training too? This must be settled
   *before* collection, because it cannot be added to a consent form retroactively (§9.1). It decides
   whether user-supplied Hebrew child speech can enter the L3 fine-tuning mix or only measure it.
7. **How many households** can realistically be recruited? The Phase 0 held-out split needs at least
   two households untouched by tuning (§9.1), which sets a hard floor of roughly three.
8. **ivrit.ai licence terms** — confirm they permit the derivative use in §4.2 (vocal-tract warping to
   produce child-like Hebrew training data), and whether commercial use is included.
9. **Who writes and reviews the Hebrew copy?** The app's tone is its most fragile quality (see the UI
   prompt), and tone does not survive machine translation. A native Hebrew speaker needs to write it,
   not check it afterwards.

---

## 11. Privacy commitments

These are product requirements, not aspirations, and each is testable:

1. No audio ever leaves the device. No network permission is required by the core feature; if the
   app declares `INTERNET` at all, it must be justified in this document first.
2. No raw audio is persisted. Enrollment clips are converted to embeddings and the source audio is
   deleted unless the parent explicitly opts to retain it for re-enrollment.
3. Voice profiles are stored in platform-encrypted storage (Android `EncryptedFile`, iOS Keychain
   with Data Protection).
4. Detection events log timestamp, child, and confidence — never audio.
5. Real family audio is never committed to version control. `.gitignore` enforces this; test
   fixtures live in a designated directory with consent documented.
6. Uninstalling deletes everything. No residue, no cloud copy to forget about.

---

## 12. What is explicitly deferred

- iOS implementation (phase 2; the shared core is designed for it now)
- Family Controls entitlement request (blocked on distribution intent)
- ~~L5 wake-phrase channel — stretch goal~~ **Promoted to a v1 candidate** (§4.2): Query-by-Example
  keyword spotting is language-agnostic, so Hebrew costs nothing extra, and the enrollment recordings
  serve double duty. Reassessed after Phase 0. It remains the designated fallback product if
  text-independent accuracy proves inadequate
- Feature Transform Adapter architecture changes for age robustness (§4.1, L4) — the simpler drift
  mitigations come first
- Any analytics or telemetry
- Multi-device or multi-parent sync
- Statistics and history beyond a simple event log

---

## 13. References

Accuracy figures and techniques in §4.1 come from the following. Numbers quoted in this spec are as
reported by the authors on their own evaluation sets, and are not directly comparable to each other.

**Children's speaker verification — the problem**
- [ChildAugment: Data augmentation methods for zero-resource children's speaker verification](https://arxiv.org/html/2402.15214)
  (JASA 155(3), 2024) — zero-resource EERs on CSLU Kids and MyST; vocal-tract augmentation methods
  and parameter ranges; the ~0.92s short-utterance exclusion.
- [Children's speaker verification in low and zero resource conditions](https://www.sciencedirect.com/science/article/abs/pii/S1051200421001548)
  (Digital Signal Processing, 2021).

**Fine-tuning — the main lever**
- [Deep Speaker Embeddings for Speaker Verification of Children](https://link.springer.com/chapter/10.1007/978-3-031-70566-3_6)
  (TSD 2024) — x-vector / ECAPA-TDNN / ResNet-TDNN compared with and without fine-tuning;
  ECAPA-TDNN at 2.9% EER fine-tuned, on Samrómur Children.

**Age drift**
- [Enhancing Age-Related Robustness in Children Speaker Verification](https://arxiv.org/abs/2502.10511)
  (ICASSP 2025) — Feature Transform Adapter and synthetic audio augmentation; 19.4% / 13.0% / 6.1%
  relative EER reduction at one-, two-, and three-year age gaps.
- [An Age-Agnostic System for Robust Speaker Verification](https://arxiv.org/html/2508.01637).

**Scoring and aggregation**
- [Trainable Adaptive Score Normalization for Automatic Speaker Verification](https://arxiv.org/abs/2504.04512)
  — AS-norm and its trainable extension; the basis for household-cohort normalization in §4.1.
- [Incorporation of Speech Duration Information in Score Fusion of Speaker Recognition Systems](https://arxiv.org/pdf/1608.02272)
  — duration-dependent scoring.

**Cross-lingual verification (§4.2)**
- [Disentangling Speaker and Language Effects in Cross-Lingual Speaker Verification](https://arxiv.org/abs/2607.01161)
  — bilingual same-speaker evaluation showing language mismatch, not inter-speaker variability, is the
  main driver of cross-lingual loss.
- [Spoken language mismatch in speaker verification (NIST-SRE / CRSS Bi-Ling)](https://ieeexplore.ieee.org/document/7078603/)
  — the 40–49% degradation figure.
- [Cross-Lingual Speaker Verification with Domain-Balanced Hard Prototype Mining and Language-Dependent Score Normalization](https://arxiv.org/pdf/2007.07689)
  — per-language score normalization, relevant to the sub-centroid design.

**Text-dependent and keyword spotting (L5)**
- [Speaker Verification: Text-Dependent vs. Text-Independent](https://www.microsoft.com/en-us/research/project/speaker-verification-text-dependent-vs-text-independent/)
  (Microsoft Research) — the short-utterance advantage underpinning L5.
- [QbyE-MLPMixer: Query-by-Example Open-Vocabulary Keyword Spotting](https://arxiv.org/pdf/2206.13231)
  and [Metric Learning for User-defined Keyword Spotting](https://arxiv.org/pdf/2211.00439) — the
  language-agnostic enrollment approach that makes Hebrew wake-phrases free.
- [Personalized Keyword Spotting for User-Defined Keywords Leveraging Text-Independent Speaker Verification](https://arxiv.org/html/2606.20106)
  — combining the keyword and speaker checks, which is exactly Channel A.

**Hebrew speech data (§4.2)**
- [ivrit.ai](https://arxiv.org/abs/2307.08720) — 3,300+ hours of Hebrew speech, freely usable;
  **adult** speech, intended here as the base for ChildAugment warping.
- [HEBDB](https://arxiv.org/pdf/2407.07566) — spontaneous Hebrew dialogue, publicly available.
- [Building an Accurate Open-Source Hebrew ASR System through Crowdsourcing](https://www.isca-archive.org/interspeech_2025/marmor25_interspeech.pdf)
  (Interspeech 2025) — 314 h transcribed Hebrew.
- **No public Hebrew children's speech corpus was found.** This gap is why §9.1 exists.

**Data**
- [Samrómur Children](https://huggingface.co/datasets/language-and-voice-lab/samromur_children) —
  CC BY 4.0, 131h, 137,597 utterances, 3,175 speakers aged 4–17.
  [Corpus paper (LREC 2022)](https://aclanthology.org/2022.lrec-1.105/). **Attribution is a license
  condition — it must appear in the app's about screen and in this repository.**
- MyST (My Science Tutor) — 1,371 students, grades 3–5, ~470h conversational speech. Access terms to
  be confirmed.
- CSLU Kids — ~1,100 children, grades K–10. LDC-licensed, paid.
- [CryCeleb](https://arxiv.org/pdf/2305.00969) — infant cry speaker verification; out of scope here,
  relevant only if the product ever targets pre-verbal children.

**Toolchain**
- [sherpa-onnx speaker identification](https://k2-fsa.github.io/sherpa/onnx/speaker-identification/)
  — ONNX embedding models with Kotlin API and Android demo APKs.
- [WeSpeaker](https://arxiv.org/pdf/2210.17016) — PyTorch training with ONNX/JIT export; the intended
  fine-tuning path.
- [3D-Speaker](https://github.com/modelscope/3D-Speaker) — models and ONNX Runtime inference.
- [Picovoice Eagle](https://picovoice.ai/platform/eagle/) — commercial on-device baseline (§4.1).
