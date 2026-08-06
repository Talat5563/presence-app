# UI Design Prompt — Presence

**How to use this file:** paste everything below the divider into a fresh Claude session (Claude
Code, claude.ai, or Claude with the Figma MCP connected). It is self-contained — it does not assume
the reader has seen this repository or any prior conversation.

**Keep it current.** If product decisions change in
`docs/superpowers/specs/2026-08-06-presence-design.md`, update this prompt to match. A stale design
prompt produces confidently wrong screens.

---

# Design the UI for "Presence", a mobile app that helps parents notice their kids

You are designing the complete user interface for an Android app (with an iOS version to follow).
Read the whole brief before producing anything.

## The product in one paragraph

A parent is on their phone. Their child says their name, asks a question, tries to show them
something — and it doesn't land. Presence listens for that specific child's voice and gently
redirects the parent's attention back to them: the screen dims, audio ducks, and the child's face
appears. All voice processing happens on the device; nothing is ever uploaded. The parent trains
the app on their own child's voice once, either by recording a short session or by importing
existing videos and voice notes.

## Who this is for

Parents who already feel bad about phone use around their kids. That single fact should govern
every design decision. They are not looking to be scolded, scored, graded, or shown a chart of
their failures. They downloaded this because they *want* to be more present and need help
noticing the moment.

## Tone — the hardest and most important constraint

**This is not a productivity app and it is not a screen-time app.** It must not feel like either.

Get this wrong and the app is uninstalled within a week, no matter how good the engineering is.

| Do | Don't |
|---|---|
| Warm, calm, human | Clinical, gamified, corporate |
| "Maya is calling you" | "ATTENTION VIOLATION DETECTED" |
| The child's photo as the emotional center | Streak counters, scores, guilt metrics |
| Soft, unhurried transitions | Alarm-red, buzzing urgency |
| Neutral language about missed moments | Judgment, shame, "you failed" framing |

The interruption should feel like a gentle tap on the shoulder from someone who loves you — never
like a notification from a compliance system.

## Screens to design

### Onboarding
1. **Welcome / value proposition** — what this does, in one screen, emotionally rather than
   technically.
2. **Privacy explainer** — a genuine selling point. Everything stays on the phone; nothing is
   recorded or saved; no account. Must include a plain-language note that Android shows a green
   microphone dot while listening and that this is normal and unavoidable.
3. **Staged permission requests** — microphone and notifications first (these alone produce a
   working app). Overlay and Do-Not-Disturb access are requested *later, in context*, when the
   parent configures the response that needs them. Each request needs a screen explaining what it
   unlocks in the parent's terms, and each involves sending the user into system Settings — design
   the hand-off and the return.

### Setting up a child
4. **Add a child** — name and photo. The photo is central to the whole experience; make choosing it
   feel meaningful, not like filling a form field.
5. **Choose enrollment mode** — record a short session together, or import existing recordings.
6. **Live enrollment session** — the child repeats a few short prompts, about 60 seconds total.
   Needs live waveform feedback, clear progress, per-clip pass/fail, and a retry path. Design this
   as something a parent and child do together and might even enjoy.
7. **Import enrollment** — pick videos or voice notes, then confirm which detected speech segments
   are the child. Needs a segment list with playback and accept/reject.
8. **Enrollment quality feedback** — a clip can be rejected for too much background noise, no
   speech found, or clipping. Show specific, non-technical reasons and what to do about it. Progress
   is reported as **seconds of usable speech collected**, not clips recorded — short clips are
   near-worthless to the recognizer, so a "4 of 6 clips done" bar would actively mislead.
9. **Enrollment complete** — confirmation with a warm payoff, not a checkmark on a form.

### Household setup — do not skip this flow

The recognizer distinguishes the target child far more reliably when it also knows what the *other*
people in the house sound like. Enrolling the parent and any siblings is a **requirement**, not an
optional extra — sibling-vs-target is the hardest confusion in real homes, and the parent's own voice
is the one the microphone hears most, since the phone is in their hand.

This flow is where setup most likely gets abandoned, so it is a genuine design problem: you are
asking a parent to record three people instead of one.

10. **Why we need everyone's voice** — the honest explanation, in the parent's terms, not the
    engineer's. Something in the spirit of *"Record yourself and their brother or sister too. It's
    how we learn to tell them apart."* A parent who understands the reason will do it; one who is
    asked for no stated reason will quit.
11. **Household roster** — who's been added, who's still missing, whose voice profile is getting
    stale. Must convey that non-target profiles are *lighter* work than the target child's: shorter
    sessions, importing is fine, "good enough" genuinely is good enough. The visual weight of these
    entries should be lower than the target child's.
12. **Re-enrollment prompt** — children's voices drift as they grow, and accuracy quietly decays with
    it. Design a gentle, periodic nudge that surfaces staleness *before* the app starts failing.
    Never alarming, and never framed as the parent's fault.

### Everyday use
13. **Home / status** — is it listening right now, and why or why not. Today's moments. A prominent
    on/off control. This screen is seen most; it should be calm and mostly empty.
14. **The intervention overlay** — *the most important screen in the app.* Appears over whatever
    the parent is using: a dimming scrim, the child's photo, their name, a dismiss action, and a
    small "not them" correction button. Design the `Soften` (default) and `Block` variants. It must
    be unmissable but never alarming, and dismissible in one tap without hunting.
15. **Notification variant** — for parents who haven't granted overlay permission. Carries the
    child's photo.
16. **Paused / disabled state** — clearly not listening, easy to resume, no nagging.

### Settings
17. **Response builder** — per child, compose the reaction: dim, mute, show photo, notification
    sound. Channels that require an ungranted permission must be visibly locked with a one-tap path
    to unlock. Three preset intents (`Notify`, `Soften`, `Block`) with custom adjustment underneath.
18. **Sensitivity** — a slider trading false alarms against missed moments. This is a first-class
    control, not buried in an advanced section. Explain the trade-off in plain words. Note that the
    honest framing is *"more reminders, some wrong"* versus *"fewer reminders, some missed"* — the
    recognizer genuinely cannot be perfect on children's voices, and the UI should not pretend the
    slider has a correct setting.
19. **Family time schedule** — optional day and time windows that further narrow when the app
    listens. Show clearly that on Android the mic *never* runs while the screen is off, regardless
    of schedule. (On iOS this guarantee does not hold in the entitlement-free tier, where a schedule
    is the only bound on listening — so design this screen's explanatory copy as
    platform-conditional rather than a single universal claim.)
20. **Manage household** — every enrolled person: the target child or children, siblings, and the
    parent. Edit, re-enroll, disable, delete. Target children and non-target profiles need visibly
    different treatment. Surface profile staleness here too.
21. **History** — a simple, non-judgmental log of moments. Deliberately not a dashboard, not a
    score, not a trend line. If it starts to feel like a report card, redesign it.

### Error and edge states
22. Microphone permission revoked while running.
23. Background service killed by the phone's battery manager, with recovery guidance.
24. Voice model failed to load — app disabled with a clear explanation.
25. No child enrolled yet — the app cannot be turned on.
26. Listening blocked because another app is using the microphone, or a call is active.
27. Empty history, first run.

## Design system to produce

Deliver a coherent system, not 27 unrelated mockups:

- **Color** — light and dark themes, both complete. Warm and calm; avoid the alarm-red /
  productivity-blue reflex. The child's photo must be the most colorful thing on any screen it
  appears on.
- **Type scale** — legible at a glance by a distracted person holding a phone one-handed.
- **Components** — buttons, cards, list rows, sliders, toggles, permission rows with lock states,
  waveform display, segment player, the photo card.
- **Iconography** — a consistent set; specify the family used.
- **Motion** — specify the intervention's entrance and exit precisely. Duration and easing carry
  most of the emotional weight here. It should breathe in, not snap.
- **Spacing and elevation** rules.

## Hard constraints from the engineering design

Do not design around capabilities that do not exist:

- **Forcing the screen to grayscale is impossible** on Android for an installed app. Dimming is
  done with a translucent overlay. Do not design a grayscale effect.
- **The green microphone dot cannot be hidden.** Acknowledge it in onboarding.
- **A persistent notification is mandatory** while listening. It is not optional chrome — design it
  as a deliberate surface, ideally a reassuring one.
- **The overlay must be dismissible without the parent hunting for a target.** It sits over other
  apps; a trapped user is a furious user.
- **iOS cannot dim the screen and cannot draw over other apps.** The iOS equivalents are a Live
  Activity in the Dynamic Island, a Time-Sensitive notification, and audio ducking. Design the
  Android UI first, but flag which screens will need genuinely different iOS treatment rather than
  a mechanical port.

## Design for an imperfect recognizer

This is a constraint on the *whole* design, not one screen.

Recognizing a specific child's voice is genuinely hard — published research puts error rates on
children's spontaneous speech far above adult benchmarks, because children's vocal tracts are smaller
and change rapidly as they grow. The engineering design fights this hard, but the app **will** be
wrong sometimes: it will occasionally show a photo when the child didn't call, and occasionally miss
when they did.

A design that assumes accuracy is a design that betrays the parent. So:

- **False alarms must be cheap to dismiss and impossible to resent.** One tap, no penalty, no dialog,
  no "are you sure". The "not them" button is a gift the parent gives the app, and it should feel
  that way rather than like error reporting.
- **Never present a confidence score or percentage.** It invites arguing with the machine.
- **Never claim certainty in copy.** "Maya might be calling you" is more honest than "Maya is calling
  you" — though test both, because hedged language can also feel weak. This is a real tension, and
  worth resolving deliberately rather than by accident.
- **The sensitivity slider is the parent's steering wheel** for this trade-off. Treat it as a primary
  control, and explain it without engineering vocabulary.
- **Missed moments must never be surfaced as failures**, by either party. No "you missed 4 moments"
  counters. That would be a guilt metric, which §Tone forbids.

## Accessibility requirements

- WCAG AA contrast minimum in both themes.
- Full screen-reader labelling, especially on the intervention overlay.
- Respect the system font scale up to 200% without breaking layout.
- Respect reduced-motion preferences.
- Never rely on color alone to convey state — permission locks and listening status especially.
- Touch targets at least 48dp.

## Platform conventions

Material 3 for Android, following Android navigation patterns and using dynamic color where it
doesn't fight the warm palette. Note where the iOS version should follow Apple's HIG instead of
mirroring Android.

## What to deliver

1. A design rationale — the palette, type scale, and motion choices, with the reasoning. Lead with
   how the tone constraint drove the decisions.
2. All screens above, with their states.
3. The component inventory with specifications.
4. Explicit notes on which screens need different iOS treatment and why.

**Format:** produce a single self-contained HTML page showing every screen as a phone-framed
mockup, with a theme toggle for light and dark, plus the design system documented on the same page.
If the Figma MCP is connected, build it in Figma as a proper component library instead.

## Where to start

Design the intervention overlay first, before anything else. It is the whole product in one screen.
If it doesn't feel like a loving nudge, nothing else in the app matters — and everything else should
follow the visual language you establish there.
