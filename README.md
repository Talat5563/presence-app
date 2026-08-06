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

The next milestone is a *Phase 0 accuracy spike* (spec §9): a JVM harness that measures the detection
system — not just the model — against public child-speech benchmarks and a household simulation set,
before any UI is built.

That ordering isn't caution for its own sake. Recognizing a specific child's voice is the whole
product and it is genuinely hard: adult-trained speaker models measure around **31% EER on children's
spontaneous speech**, which is unusable per-utterance. Fine-tuning on children's speech reaches
**2.9% EER**, and the design adds household-cohort scoring and cross-segment evidence accumulation on
top. Spec §4.1 lays out that accuracy ladder with sources, and §9 defines explicit go / no-go numbers.

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
              └─> score vs every enrolled household member
                    └─> accumulate log-likelihood across segments
                          └─> intervention
```

The VAD gate keeps battery cost survivable. **The accumulator is what makes it accurate** — no single
0.9-second word has to carry the decision, and scoring against the whole household (siblings and the
parent included) turns an absolute threshold into a relative comparison. Enrolling the other people in
the house is therefore a requirement, not a nice-to-have.

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

## Attribution

If the shipped model is fine-tuned on **Samrómur Children** (spec §4.1), that corpus is licensed
**CC BY 4.0** and attribution is a license condition — it must appear both here and in the app's about
screen:

> Samrómur Children, Language and Voice Lab, Reykjavík University — CC BY 4.0.
> https://huggingface.co/datasets/language-and-voice-lab/samromur_children

Third-party model licenses (WeSpeaker, 3D-Speaker, NeMo, Silero VAD) must each be verified
individually before shipping — see spec §10, open question 2.

## License

Not yet chosen. Pick one before making this repository public.
