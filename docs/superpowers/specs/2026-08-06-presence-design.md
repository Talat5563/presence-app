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
| Codebase | Kotlin Multiplatform shared core, native UI per platform |
| Platform order | Android first, iOS second |

**Listening rule, stated precisely:** the microphone is active only when the screen is on and
unlocked. If the parent enables family-time windows, the mic is active only when the screen is on
*and* the current time falls inside a window. The mic is never active while the screen is off.
This rule is a hard invariant, not a setting — the "always listening" option is deliberately absent.

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
  ├─> EmbeddingMatcher — embed the speech segment [ECAPA-TDNN class, ~20 MB, speech only]
  │     └─> cosine similarity vs each enabled VoiceProfile
  │
  ├─> Debouncer — N hits within a window, then cooldown
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
- **The debouncer is what makes the app tolerable to live with.** A single-frame trigger would fire
  constantly. Detection requires N consecutive positive segments inside a rolling window, followed
  by a cooldown period during which no further intervention fires.

### Known accuracy risk

Speaker-embedding models are trained predominantly on adult speech. Children's voices — higher
pitch, less stable formants, rapidly changing with age — are materially harder. This is the
project's biggest technical unknown. Mitigations, in order:

1. Validate against real recordings of children **before** building any UI (see §9, Phase 0).
2. Ship the sensitivity slider as a first-class control, not a hidden advanced setting.
3. The "not them" button stores the rejected embedding as a negative example; the matcher requires
   a candidate to be closer to the positive centroid than to any negative example.
4. Prompt for re-enrollment periodically, since a child's voice drifts over months.

If Phase 0 shows accuracy is unacceptable, the fallback is generic child-voice detection with a
manual confirm step — a materially different product, and a decision point worth reaching early
and cheaply.

---

## 5. Enrollment

Both modes converge on the same `EnrollmentService`; only clip acquisition differs.

**Mode A — guided live setup.** The child repeats a handful of short prompts (~60 seconds total).
The UI shows live waveform feedback and per-clip quality results. Best quality, requires the
child's cooperation for one minute.

**Mode B — import existing recordings.** The parent selects existing videos or voice notes. Audio
is extracted, VAD-segmented, and the parent confirms which segments are the child. No cooperation
needed, but quality varies and some imports will be rejected.

`ClipQualityCheck` gates both: minimum total speech duration, adequate signal-to-noise, no
clipping, single dominant speaker. Rejected clips give a specific reason ("too much background
noise", "we couldn't hear speech here") rather than a generic failure.

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

**Phase 0 — accuracy spike, before any UI.** A JVM harness that runs the pipeline over a corpus of
real child and adult recordings, reporting false-accept and false-reject rates across thresholds.
This validates or kills the core premise for a few days' work. Nothing else should be built first.

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
2. Specific model selection and license verification for the VAD and embedding models.
3. Multi-child support in v1, or single child with the schema ready for more?
4. Minimum Android API level.
5. Distribution intent: personal use, open source, or App/Play Store release? This determines
   whether the Family Controls entitlement request and store-review preparation are in scope.

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
- Any analytics or telemetry
- Multi-device or multi-parent sync
- Statistics and history beyond a simple event log
