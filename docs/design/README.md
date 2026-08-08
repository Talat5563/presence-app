# Presence — design handoff

Design artifacts for the Android build (iOS to follow). Working name; see the name-dependence note at the end of `screens.md`.

| File | What it is |
|---|---|
| `tokens.json` | Colours for both themes, type scale, spacing, radii, elevation, motion durations and easings. Machine-readable — generate the Compose theme from this rather than eyeballing values. |
| `strings.json` | Every copy string, keyed, with `en` and `he` values. Strings carrying an `intent` note need a native Hebrew writer, not a translator. |
| `screens.md` | Per screen: purpose, every state, component tree, and the string keys it uses. Also the bidirectional layout rules and the iOS divergence notes. |
| `components.md` | Component inventory with variants, states and specifications. |
| `mockups.html` | The visual reference. Open in any browser — all 31 screens, phone-framed, light and dark, English and Hebrew on the key screens. |

## Read these first

**The Hebrew in `strings.json` is a placeholder.** It was written to size the layout, not to ship. Tone does not survive translation and tone is this app's most fragile quality, so every string with an `intent` note must be rewritten by a native Hebrew speaker who reads the intent and reaches for the feeling. Strings without an `intent` note are mechanical and a good translator is enough.

**One rule outranks everything else in these documents:** the app never claims to be listening when it is not. Every degraded state is visible on the home screen *and* in the persistent notification, at the moment it happens.

**Three copy invariants that must survive any rewrite:**
- The hedged form ships. *"Maya might be calling you"*, never *"is calling you"* — the unhedged form is a lie roughly one time in five, and it is a lie the parent will personally catch.
- Detected moments may be counted. **Missed moments must never be counted, estimated, hinted at, or given reserved space.**
- Never a confidence score or percentage, anywhere.

## Known gaps

- Photos are placeholders. Real family photos drop into the `PhotoSlot` / `PhotoCard` components unchanged.
- Hebrew copy needs the native-writer pass described above.
- iOS Tier 1 has four screens flagged as real rework rather than a mechanical port — listed at the end of `screens.md`. The Live Activity intervention has no mockup yet and needs its own design pass.
- Dynamic colour (Material You) is not yet mapped. The warm palette is deliberate and should win where the two conflict; the sage listening state and burnt-amber degraded state must not be recoloured by the system.
