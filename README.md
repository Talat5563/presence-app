# Presence

An app that helps parents notice their kids.

A parent is on their phone. Their child says their name, asks a question, tries to show them
something — and it doesn't land. Presence listens for *that specific child's voice* and gently
redirects the parent's attention back to them: the screen dims, audio ducks, and the child's face
appears.

Everything happens on the device. No account, no server, no audio ever uploaded.

> **Working name.** "Presence" is a placeholder. It appears only in documentation at this stage, so
> renaming costs nothing.

---

## Status

**Design phase.** No implementation code yet — deliberately.

The next milestone is a *Phase 0 accuracy spike*: a JVM harness that measures speaker-recognition
accuracy on real recordings of children before any UI is built. Speaker-embedding models are trained
mostly on adult speech, and children's voices are materially harder. That spike validates or kills
the core premise for a few days' work, and nothing else should be built ahead of it.

## What's here

| Path | Contents |
|---|---|
| `docs/superpowers/specs/2026-08-06-presence-design.md` | The full design specification — architecture, detection pipeline, per-platform intervention design, permissions, failure modes, testing strategy |
| `docs/design/UI_DESIGN_PROMPT.md` | A self-contained, ready-to-paste prompt for designing the complete UI. Hand it to a fresh Claude session |

## Planned shape

**Kotlin Multiplatform shared core, native UI per platform.** Android first, iOS second.

The sensing engine is pure portable logic; everything touching the OS is a thin adapter. That seam
matters because the engine is the hard, valuable part while the OS integrations are what differ per
platform — and it means the whole recognition pipeline is unit-testable on the JVM against recorded
WAV fixtures, with no device and no emulator.

```
mic (16 kHz mono)
  └─> VAD gate — is anyone speaking?          ~1 MB, always on
        └─> speaker embedding                 ~20 MB, speech only (~5% duty cycle)
              └─> similarity vs enrolled profile
                    └─> debounce + cooldown
                          └─> intervention
```

The VAD gate keeps battery cost survivable. The debouncer keeps the app tolerable to live with.

**Platform reality:** Android can dim and draw over other apps. iOS cannot — there, the same product
is delivered through a Live Activity in the Dynamic Island, a Time-Sensitive notification, and
`AVAudioSession` ducking. See §6 of the spec for the full mapping and the two-tier iOS plan.

## Privacy commitments

Product requirements, not aspirations — each one testable:

- No audio leaves the device, ever
- No raw audio is persisted; enrollment clips become embeddings and the source is deleted
- Voice profiles live in platform-encrypted storage
- Event logs record timestamp, child, and confidence — never audio
- Real family audio never enters version control (enforced by `.gitignore`)
- Uninstalling deletes everything

## License

Not yet chosen. Pick one before making this repository public.
