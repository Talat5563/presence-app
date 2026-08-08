# Design ↔ engine integration gaps

**Date:** 2026-08-08
**Sources:** the design handoff in this folder, the spec, and the engine as built and merged in Phase 0.

Phase 6 (the Android app) starts here. Nothing below is a criticism of the visual design — these are
places where the design and the codebase currently disagree about what is possible, or where one
assumes something the other does not provide.

**First, the expected part.** The design references `ChildProfile`, `InterventionConfig`,
`ListeningSchedule`, `ClipQualityCheck` and persistent storage, none of which exist. That is correct
and planned: spec §3 lists them, and Phase 0 deliberately built only the sensing engine. They are
Phase 6 scope, not gaps. What follows is everything that is *not* explained by that.

---

## Decisions needed before Phase 6 can be planned

### D1 — Multi-child in v1: yes or no?

**The design assumes yes throughout.** `home.listeningForMany`, `overlay.hedgedCallSecondChild`,
`overlay.notEither`, and screen 15's `soften-multi` state all exist.

**The engine cannot do it.** `DetectionEngine` carries a single `targetPersonId` and
`HouseholdMatcher` a single `target`; a second `TARGET_CHILD` profile lands in the *cohort*, where it
actively suppresses the first child's score. Two enrolled children would make the app worse at
detecting either one.

Spec §10 open question 3 asked this and it was never answered. It cannot stay open any longer —
the two answers produce materially different work:

- **No:** drop those four strings and one state from the design; the roster still enrolls siblings,
  but only as non-targets. Cheapest, and defensible for a first release.
- **Yes:** the engine needs N target profiles with per-target accumulators, and the matcher needs to
  exclude other targets from the cohort rather than scoring against them. Not enormous, but it is
  engine work with its own accuracy implications, and Phase 2's measurement would need to cover it.

### D2 — Android "mute": DND, or ducking?

Spec §6 renders Android's mute as `AudioManager` volume plus Do-Not-Disturb, and §7 counts
notification-policy access as one of the four Settings trips. **The design has dropped it**: screen
18's only audio channel is `response.channelDuck` ("Turn the sound down") as an unlocked toggle, and
no screen, state or string covers the DND permission.

Ducking is the iOS mechanism. On Android it lowers media volume but does not silence a ringtone or
a notification, which is what a parent asking for "mute" probably means. Either:

- **Adopt ducking on both platforms** — simpler, one fewer Settings trip, weaker intervention. Then
  spec §6 and §7 need correcting.
- **Keep DND on Android** — matches the spec, but screen 18 needs a locked row and screen 03 needs a
  fourth permission hand-off.

---

## Engine gaps — real, and Phase 0's to own

### E1 — "Not them" has no data path (Critical)

`VoiceProfile.negatives` and `MatcherConfig.negativeVetoMargin` both exist and work. But **the
embedding that caused a detection is discarded**: `DetectionEvent` carries `peakSimilarity` and
nothing else, and `MatchScore` carries no embedding either.

So the correction button — which the design calls the most important control after the photo, and
which spec §4.1 relies on to make a drifting profile decay toward silence rather than toward firing
on a sibling — **cannot be implemented against the current engine.**

Fix is small: carry the observation's `Embedding` on `DetectionEvent`, or expose it through the
existing observation listener. It must be decided before Phase 6, because it changes a type the
whole app binds to.

### E2 — Additive re-enrollment cannot work (Critical)

Spec §4.1 L4: *"Multiple enrollment sessions over time are retained and merged rather than
overwritten."* Screen 12 offers exactly that (`household.topUpAdditive`).

`EnrollmentService.enroll` builds a **fresh** profile from the clips it is given, with
`negatives = emptyList()`. Centroids are plain means with no retained sample counts, so a later
session cannot be weighted and merged — and topping up would silently discard every "not them"
correction the parent had made.

Needs either retained per-language counts on `VoiceProfile` (to merge means correctly) or retained
clip embeddings. The first is cheaper and sufficient.

### E3 — Cooldown: 45 s in the design, 60 s in the engine

`tokens.json` and screen 14 say 45 000 ms; `AccumulatorConfig.cooldownMs` is 60 000. Reconcile.

**More importantly, the design misdescribes the behaviour.** Firing clears the evidence window *and*
starts the cooldown, and observations during cooldown are dropped entirely — so the real gap before a
second intervention is the cooldown **plus** a fresh accumulation window. A child still talking
cannot re-fire the moment the cooldown lapses; that was a deliberate choice (see the execution
notes), and the design's timing copy should reflect it.

Also: cooldown starts at **fire**, not at dismissal. The design's claim that "a dismissal does not
let it immediately reappear" has no mechanism behind it — dismissal is not wired to the accumulator
at all. If that behaviour is wanted, it is engine work.

### E4 — Detection latency floor is ~4.5 s

The design budgets 10 s and adds ~700 ms of entrance animation. The earliest possible fire is about
4.5 s of near-continuous speech: `maxSegmentMs` 3000 with `embedHop` 1500 yields two observations,
and `fireThreshold` 6.0 against `maxAbsLlr` 4.0 needs two. Comfortably inside budget — worth knowing
when tuning the entrance so it does not feel like an alarm catching up.

### E5 — "A moment" has no wall clock

`DetectionEvent.timestampMs` is a position in the audio stream, reset by `reset()` and by any service
restart. Screen 23's day grouping, `home.momentsToday`, and the `{time}` in screens 14 and 26 all
need wall-clock timestamps and a persistence layer. Both are Phase 6 scope, but the event type needs
the field.

### E6 — Smaller engine mismatches

- `EnrollmentPolicy.minSecondsPerLanguage` is global at 45 s, but screen 06 asks 60 s per language
  for targets and roughly half, with a relaxed quality bar, for non-targets. Needs parameterising by
  role.
- Nothing distinguishes "monolingual by choice" (screen 06c's *standing fact*) from "enrollment
  incomplete". Both produce `EnrollmentResult.Incomplete`, so the app cannot tell a settled state
  from an unfinished one.
- `Language` has no code-switched value, though spec §4.2 treats code-switching as normal.
- `DetectionEvent` has no `outcome` field (spec §3 specifies one); history needs
  went-to-them / dismissed / corrected.
- No service-kill counter for screen 03d's "≥2 confirmed kills" trigger.

---

## Design defects — the designer's to fix

**Critical**

- **`overlay.blockReason` is a concatenation fragment**: *"might be calling you. You asked me to stop
  you here."* — designed to be joined to the name. `screens.md` forbids exactly this ("Never
  concatenate… never `{name}` + `" might be calling you"`"), and in Hebrew a writer cannot repair it,
  because the verb must agree with a subject the fragment does not contain.

**Important**

- **Notification denial is said to block listening.** On Android 13+ a denied `POST_NOTIFICATIONS`
  does not stop a foreground service. Spec §7 also says microphone plus notifications produce a
  working app — it never claims denial is fatal.
- **Home (13) is listed as a mechanical iOS port**, but it carries the persistent notification's
  reassurance job on iOS, and `home.listeningUntil` / `listeningNoWindow` encode the Android
  screen-on guarantee that iOS Tier 1 cannot make. It belongs in the rework table alongside 02, 15,
  18 and 20.
- **`notification.degradedTitle` hardcodes "Presence"**, contradicting the handoff's own
  name-dependence section.
- **Spec §8's "iOS audio session interrupted" has no screen** and is absent from the iOS divergence
  table.

**Minor** — two referenced keys do not exist (`common.turnOn`, `notification.appName`); 49 of 298
keys are unreferenced, including whole sets (`enrollMode.nonTarget*`, `response.channel*IOSImpossible`,
`onboarding.privacyTitleIOS`); three trees contain hard-coded English; several enumerated states are
never designed (15 `soften-multi`, 21 confirmations, 22 confirmations, 07 six of eight, 13
`outside-window`); 15 components are used but unspecified — most notably **`TextField`/`FieldLabel`**,
which has no spec, no focus or error state, and no bidi rules, despite the child's name being the
document's central bidirectional case.

---

## What this changes about the roadmap

Phase 6's plan must open with D1 and D2 answered, and with E1–E3 fixed in the engine — they are
small, but each changes a type or a constant the whole app binds to, and finding them mid-build
would be expensive. E1 in particular gates a control the design treats as central.
