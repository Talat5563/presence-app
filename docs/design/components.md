# Presence — Component inventory

Token names refer to `tokens.json`. Every component below is specified for both themes, both text directions, and font scaling to 200%.

**Three rules that apply to every component here:**

1. **Minimum touch target 48dp.** The overlay correction control is 56dp.
2. **State is never conveyed by colour alone.** Every stateful component carries at least two of: shape, glyph, position, text.
3. **No fixed-height text containers.** Everything sizes to content, so 200% font scale grows the layout rather than clipping it.

---

## Buttons

Six types. **One primary per screen, never two.**

### Primary
The single forward action.

| | |
|---|---|
| Height | 52dp |
| Padding | 0 28dp |
| Radius | `radius.full` |
| Type | `typography.label` (sans 500 / 15) |
| Fill | `color.{theme}.accent` |
| Label | `color.{theme}.onAccent` |

**States.** `default` · `pressed` (`accentPressed`) · `disabled` (`surfaceSunken` fill, `disabledInk` label — always paired with a caption naming the unlock condition) · `loading` (16dp ring spinner, label retained, never replaced by a bare spinner).

**Variant — with hand-off.** Trailing `open_in_new` at 17dp, 9dp gap. Used on every action that leaves for system Settings, so the trip is never a surprise.

### Tonal
The equal-weight alternative, for when two paths are genuinely equal.

Height 52dp · `accentSoft` fill · `onAccentSoft` label · pressed `accentSoftPressed`.

### Outline
Secondary on a card, and the shape of every "Open Settings" hand-off.

Height 52dp (48dp in footers) · 1.5dp `outlineStrong` border · transparent fill · `onBackground` label · pressed fills `background`.

### Quiet
Text only, still 48dp tall. Skips, "later", "we only speak one language at home".

Height 48dp · padding 0 18dp · transparent · `onBackgroundSubtle` label · pressed fills `surfaceMuted`.

### Offered hand
**The "not them" correction, and only that.**

| | |
|---|---|
| Height | 56dp |
| Width | Full, or flex 1 alongside one sibling |
| Type | `typography.labelLarge` (16dp) |
| Fill | `dark.overlaySurfaceAlpha` — `rgba(255,255,255,0.14)` |
| Pressed | `rgba(255,255,255,0.22)` |
| Icon | **None** |

Sits directly beneath the dismiss action at **identical size**, differing only in fill. Small is how you design something you hope nobody presses — this is designed to be pressed. One tap, no dialog, no "are you sure", no penalty, no confirmation toast.

### Destructive
Height 52dp · 1.5dp `status.destructiveOutline` border · transparent fill · `status.destructive` label. Confirmation is a **single** sheet stating what is lost — never a maze. Appears only on screen 22.

---

## The two lock affordances

These must **never** look alike. Reuse one for both and you get a parent tapping forever at something that will never open.

### LockedRowOpenable — an ungranted permission

```
Row (surface, outline, radius card, 48dp min)
  Icon lock_open_right (22dp, status.locked)
  Column: Title (body, full contrast) · Subtitle (caption) — names the permission
  OutlineChip (40dp, "Turn on")
```
The whole row is the tap target. Full-contrast text. **A control on the trailing side.**

### LockedRowUnavailable — a platform impossibility

```
Row (surfaceMuted — sunken, radius card)
  Icon block (22dp, status.unavailable)   ← a DIFFERENT GLYPH, not a padlock
  Column: Title (status.unavailable) · Subtitle (status.unavailable)
          — names the platform limit and absolves the parent
  [nothing]
```
Dimmed text, sunken background, and **no control at all**. Nothing to tap means nobody taps forever.

| | Openable | Unavailable |
|---|---|---|
| Glyph | `lock_open_right` | `block` |
| Text contrast | Full | Dimmed |
| Background | `surface` | `surfaceMuted` |
| Trailing control | Yes | **None** |
| Row tappable | Yes | No |

---

## Cards

### Card (E1)
`surface` fill · 1dp `outline` · `radius.cardLarge` (16) · padding 14–16dp · no shadow. List rows, settings groups.

### ChoiceCard
A whole-card tap target offering one of two or more paths.

```
Card
  Row: Icon (24dp) · Title (titleSmall)
  Body
  TagRow (optional)
```
**States.** `default` (1dp `outline`) · `recommended` (1.5dp `accent` border — marks the likelier path without demoting the other) · `pressed` (fill `background`).

Used on 05, 30. On the non-target variant of 05 the `recommended` border moves to the *import* card — the emphasis swap is the reduction made visible.

### PhotoCard
The emotional centre. One face, named.

| Context | Size | Radius |
|---|---|---|
| Overlay `soften` | 216dp | `radius.sheet` (24) |
| Overlay `block` | 248dp | 28 |
| Overlay multi (lead) | 200dp | 24 |
| Overlay multi (second) | 78dp | 18 |
| Notification | 76dp | 16 |
| Enrollment complete | 132dp | 24 |
| Re-enrollment | 104dp | 22 |
| Add a child (empty) | 158dp | 24, dashed `outlineStrong` |

Elevation `e3` on the overlay. Placeholder fill is a 150° gradient `photoPlaceholderFrom → photoPlaceholderTo`. **The photo must be the most saturated element on any screen it appears on** — every other colour here is chosen to sit quietly underneath it.

Never a group photo, never a carousel. The photo never mirrors in RTL.

### InsetNote
A fact attached to the person or setting it belongs to, inside their card.

```
Panel (surfaceSunken, radius card, padding 11–12dp)
  Icon (19dp, onBackgroundSubtle)
  Column: Fact (bodySmall) · FixAction (label, accent)
```
Used for staleness and the one-language limitation on 21. Not a badge, not amber, not an error — a fact, stated once, with the fix attached.

### InfoPanel / WarningPanel / RelaxationPanel

| Variant | Fill | Icon colour | Use |
|---|---|---|---|
| `info` | `surfaceMuted` | `onBackgroundSubtle` | Neutral facts, the green/orange dot notes |
| `warning` | `status.warningSurface` | `status.warning` | Degraded states, rejection reasons |
| `relaxation` | `status.listeningSurface` | `status.listening` | The non-target reductions, the Android screen-off guarantee |
| `reassure` | `surface` + `outline` | `status.listening` | "Nothing you've built is lost" |

All are icon + text, radius `card`, padding 12–16dp. The warning variant **always** carries a glyph — never colour alone.

---

## List rows

### TargetPersonCard
```
Card (surface, outline, radius cardLarge, padding 12–13dp)
  Avatar 52dp (avatarFill / avatarInk, serif initial)
  Column
    Name (titleSmall, 18dp medium)
    StatusLine (14dp) — sage + check_circle when ready
    LimitationLine + FixAction (when applicable)
  Trailing: chevron_right (roster) | more_vert (manage)
```

### NonTargetRow
```
Row (surfaceMuted, radius card, padding 9–12dp)
  Avatar 38dp (avatarFillMuted / avatarInkMuted)
  Column
    Name (body, 16dp regular)
    Supporting (caption) — states the lighter ask in words
  Trailing: chevron_right | more_vert
```

**The weight difference is the message.** 52 vs 38dp avatar, 18dp medium vs 16dp regular name, bordered card vs flat tinted row. The lighter row *is* the signal that it is lighter work — and its supporting line says so in words too, because visual weight alone is not a promise.

### MomentRow (history)
```
Card (surface, outline, radius card)
  Avatar 42dp
  Column: "{name}, {time}" (body) · Outcome (caption)
```
No icon on the outcome, no colour coding. A correction is logged in exactly the same neutral voice as a success.

### ToggleRow
```
Row (surface or surfaceMuted, radius card, 48dp min)
  Column: Label (body) · StateLine (caption) — states the consequence IN ITS CURRENT POSITION
  Toggle (48×28 or 52×30)
```
The state line is what makes the trade legible without tapping: *"Off — they'll be deleted"* / *"On — a top-up builds on them"*.

---

## Controls

### ListeningToggle
The primary on/off control, 72×40dp, thumb 32dp, radius full.

| State | Track | Thumb | Paired signals |
|---|---|---|---|
| `on` | `status.listening` | white, trailing | Solid dot in the ring, the word "Listening" |
| `off` | `outline` / `surfaceSunken` | `surface`, leading | Hollow dot, the word "Paused" |

Three redundant signals at once — dot fill, colour, words — so state survives colour blindness and a half-second glance.

### Toggle (settings)
48×28dp (52×30 on emphasis rows), thumb 22–24dp. `status.listening` on, `outline` off.

### StatusRing
The home screen's single largest element. Three concentric circles: 132dp outer, 96dp mid, 56dp core.

| State | Outer | Mid | Core |
|---|---|---|---|
| `listening` | `status.listeningSurface` | mid-tint | `status.listening` fill + filled `graphic_eq` |
| `paused` | `background` step | `surfaceSunken` | 2dp `paused` ring + `pause` glyph |
| `degraded` | warning-tinted | warning-tinted | `status.warning` glyph, no fill |
| `mic-busy` | neutral | neutral | `phone_in_talk`, no fill |

The filled icon variant is used here and **nowhere else** in the product.

### Slider (sensitivity)
Track 8dp `surfaceSunken` · fill `accent` · thumb 32dp `surface` with 2dp `accent` border and `e2` shadow · 34dp touch row.

**No numbers, no ticks, no percentage, no `recommended` marker.** End labels state the trade in words; a live sentence beneath restates the parent's current position in plain language. The slider has no correct setting and the UI never implies one. Flips direction in RTL.

### DayPicker
Seven 30dp circles. Selected: `accent` fill, `onAccent` label. Unselected: `surfaceSunken`, `onBackgroundFaint`. Order reverses in RTL; day letters are localised.

### Chip

| Variant | Fill | Use |
|---|---|---|
| `outline` | transparent + 1.5dp `outlineStrong` | Auto-resume options, photo source |
| `accentSoft` | `accentSoft` | Enrollment tags on the emphasised card |
| `neutral` | `surfaceMuted` | Enrollment tags on the secondary card |
| `sage` | `status.listeningSurface` | The collected-duration confirmation |

Height 44dp when tappable, 26–30dp when informational. Radius `full` (tappable) or `chip` (informational).

### PresetTile
Three side by side on the response builder, flex 1 each.

```
Tile (radius card, padding 11dp 8dp, centred column)
  Icon 20dp
  Label (13.5dp medium)
```
**States.** `unselected` (`surface` + `outline`) · `selected` (`accentSoft` + 1.5dp `accent`) · `unavailable` (`surfaceMuted`, `block` glyph, `status.unavailable` label, not tappable — matches LockedRowUnavailable exactly).

---

## Enrollment components

### Waveform
Live input display. 12 bars at 5dp wide, 3dp radius, 4dp gap, container 96–104dp tall. The callout variant uses 6 bars at 6dp.

**Amplitude → colour**, not just height: `surfaceSunken` (silence) → mid-tint → `accent` (speech). Bars animate on amplitude only; under reduced motion the bars hold a static representative shape rather than animating.

**Does not mirror in RTL** — a mirrored waveform reads as a different sound.

### LanguageProgressPair / LanguageProgressStack
```
Card (radius chip, padding 10dp 12dp)
  Row: LanguageName (script-matched family) · Value
  Track 7–10dp (surfaceSunken) · Fill (accent | status.listening when complete)
  [optional] StatusWord (caption) — "still short" | "nearly there"
```

**States.** `active` (1.5dp `accent` border) · `inactive` · `complete` (sage fill + `check`, border dropped).

**The value is always seconds of usable speech, per language** — never "4 of 6 clips". Short clips are near-worthless to the recognizer, so a clip count would actively mislead. **Bars only ever go up**; a rejected clip removes nothing, and every rejection reason says so.

### RecordButton
84dp circle, `accent` fill, `onAccent` 28dp rounded-square stop glyph, with a 10dp (14dp on the callout) glow ring at 12–16% accent alpha.

**States.** `idle` (mic glyph) · `recording` (stop glyph, glow ring breathes at 2s) · `processing` (ring spinner). Under reduced motion the glow ring is static.

### SegmentRow
```
Card (surface, outline, radius card, padding 12dp 13dp)
  PlayButton (play_circle 30dp, accent)
  Column
    Row: TimeRange (14.5dp medium) · LanguageTag (chip, script-matched)
    MiniWaveform (8 bars, 3dp, 16dp tall)
  Actions: Accept (44dp circle, listeningSurface, check) · Reject (44dp, surfaceMuted, close)
```
**States.** `default` · `playing` (pause glyph, waveform animates) · `accepted` (accent-tinted border) · `rejected` (`surfaceMuted`, struck range, reason line, **Undo chip**).

Rejection is always undoable. Language tags are per-segment because coverage is reported per language.

### RejectionCard
```
Card (surface + outline dark / warningSurface light, radius card)
  Icon 20dp (status.warning)
  Column: Reason (15dp medium) · Fix (14dp)
```
Every rejection names a specific, non-technical cause **and** what to do about it, and reassures that nothing already collected was lost. Amber, never red.

---

## Notification components

### PersistentNotification
```
Notification (surfaceRaised, radius 22)
  HeaderRow: AppMark 14dp · AppName · StatusPill
  Title (16dp medium) · Body (14dp)
  Divider
  ActionRow: two 44dp alpha-fill pills
```

**States.**

| | Fill | Header | Sound | Importance |
|---|---|---|---|---|
| `listening` | `surfaceRaised` | sage dot + "Listening" | Silent | LOW |
| `paused` | `surfaceRaised` | hollow dot + "Paused" | Silent | LOW |
| `degraded` | `warningSurface` + `warningOutline` | `error` + "Not listening" | **Once** | DEFAULT |

The degraded state is the only time this notification ever makes a noise, and the only time it re-sorts to the top of the shade. For a parent who rarely opens the app, this row may be the only place they would learn it broke.

### InterventionNotification
```
Notification (radius sheet, e3)
  HeaderRow: AppMark · AppName · "now"
  Row: Photo 76dp (radius 16) · Column[Name serif 26dp · HedgedLine 16dp]
  Divider
  ActionRow: dismiss · notThem
```
Both are notification actions, so the correction works without unlocking or opening the app. Sound is a single soft two-note chime, never the default alert tone.

---

## Overlay components

### Scrim
`color.{theme}.scrim` — warm brown, **never neutral grey**. Fades in over 520ms on `gentleIn`. Tappable to dismiss (announced in the accessible label).

### OverlayColumn
```
Column (centred)
  PhotoCard
  Name (serif 44dp / 48dp block)
  HedgedLine (bodyLarge 20dp)
  DuckedChip (optional)
```
Centre-aligned by deliberate choice, so it survives RTL mirroring almost unchanged.

### OverlayButtonColumn
Buttons **stack** rather than sitting side by side, so their order does not flip in RTL and muscle memory holds across languages. `soften`: two 56dp buttons + a caption. `block`: one 56dp primary, then a row of two 56dp (correction leading, return trailing), + a caption.

### DuckedChip
26dp pill, 7% white alpha fill, `volume_down` 15dp + caption. States what already happened, so the parent is asked to do nothing.

---

## Progress & guidance

### ProgressRow (roster)
Track 8dp `surfaceSunken`, fill `accent`, radius full, with a "{done} of {total}" caption. **Appears only on screen 11**; its absence is part of what distinguishes 21.

### SettingsPreview / MockRadioGroup
A drawing of the system destination the parent is being sent to.

```
Panel (surfaceMuted or surfaceSunken, radius card)
  Label — what to look for
  MockRow / MockRadioGroup (surface, outline) — the shape of the real screen
  Caption — the wording may differ on your phone
```
Used on 03c and 03d. The battery destination is named differently on every manufacturer's phone, so we **show** the shape rather than promise a label. The selected mock option carries a 5dp `accent` ring and a "pick this one" hint.

---

## Structural

### PhoneFrame *(documentation only — not a product component)*
340×716dp inner, 9dp bezel. Android radius 38, iOS radius 44. Exists only in `mockups.html`.

### StatusBar
Android: 13dp medium, `onBackgroundSubtle`, right-aligned glyph cluster. iOS: 14dp semibold, `onBackground`. The OS microphone dot is drawn illustratively in the mockups — **the real dot cannot be styled or hidden**, on either platform.

### AppBar
`min-height: 44dp` leading affordance + optional 18dp medium title. Android: `arrow_back`. iOS: `arrow_back_ios` + a text label in `accent`. The back glyph mirrors in RTL.

### BottomSheet
`radius.sheetTop` (24 24 0 0) · `surface` · `sheetLift` shadow · 38×4dp grabber, centred. Scrim `scrimSheet`. Used for the overlay permission ask (03c) and the one-language exit (06c).

### AppMark
A ring with a soft halo — readable as an ear, a ripple, an aperture, or a face, and committed to none, because it must outlive the name. **Drawn, never lettered**, so nothing needs redrawing on a rename.

| Size | Container radius | Ring | Halo |
|---|---|---|---|
| 88dp | 26 | 40dp, 3dp stroke | 8dp spread |
| 48dp | 15 | 20dp, 2.5dp | 5dp |
| 32dp | 10 | 13dp, 2dp | — |
| 14dp | 4 | 6dp, 1.5dp | — |

Three fills: `accent` on light, `accentSoft` inverse, `surface` on dark. Never mirrors.

---

## Iconography

`Material Symbols Rounded`, weight 300, optical size 24, fill 0. Rounded because the geometry is friendlier and matches the pill buttons; weight 300 because the default 400 is too assertive next to this type.

**Filled variants are used for exactly one thing:** the active listening indicator on `StatusRing`.

| Size | Use |
|---|---|
| 15dp | Inline with caption text |
| 19–20dp | Row leading, panel leading |
| 22–24dp | Standard, app bar |
| 30–34dp | Feature icons, empty states |

**Set used.** `graphic_eq` `mic` `mic_off` `pause` `pause_circle` `notifications` `person` `person_add` `person_off` `group` `schedule` `tune` `history` `translate` `lock_open_right` `block` `error` `check` `check_circle` `open_in_new` `arrow_back` `arrow_back_ios` `arrow_forward` `chevron_right` `close` `add` `add_circle` `add_a_photo` `photo_camera` `photo_library` `perm_media` `play_circle` `replay` `volume_down` `volume_off` `music_note` `record_voice_over` `battery_alert` `hearing_disabled` `phone_in_talk` `phonelink_off` `shield` `autorenew` `delete` `more_vert` `bedtime` `bolt` `brightness_low` `pan_tool` `fiber_manual_record` `info`

**RTL.** Mirror: `arrow_back` `arrow_back_ios` `arrow_forward` `chevron_right`. Never mirror: `mic` `mic_off` `graphic_eq` faces `photo_camera` `play_circle` media controls, the app mark.

**No icon ever carries meaning on its own** — every one has a visible label beside it or an accessible name.

---

## Accessibility contract

Applies to every component above.

- **Contrast.** WCAG AA in both themes. All pairs in `tokens.json` carry their measured ratio.
- **Targets.** ≥48dp everywhere. The overlay correction is 56dp.
- **Font scale.** To 200% without breaking layout. No fixed-height text containers; icon+label rows wrap to a column above 1.6×; on the overlay the **photo** shrinks (216 → 160dp) before any text scales, and text never scales down.
- **Reduced motion.** All transitions become a 160ms cross-fade. No scale, no translate, no breathing glow, no waveform animation. The overlay still **fades** rather than cutting in — an instant cut is the one thing that would feel like an alarm.
- **Screen reader.** The intervention overlay is an assertive live region announcing `overlay.a11yOverlay`. Focus lands on dismiss. Every icon-only control has an accessible name. Toggles announce their state as words, never as colour. Progress announces seconds, matching the visible label.
- **Colour independence.** Permission locks differ by glyph and by the presence or absence of a control. Listening status carries dot fill, colour, and words simultaneously.
