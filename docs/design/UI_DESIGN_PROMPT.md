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

**"Presence" is a working name, not a decided brand.** It may well be replaced before launch, so do
not build the identity on the word itself — no wordmark the product cannot survive losing, no
visual pun on presence, no copy that only makes sense if the app is called this. Design a mark and a
palette that would still hold together under a different name, and say in your rationale which
pieces are name-dependent.

## The product in one paragraph

A parent is on their phone. Their child says their name, asks a question, tries to show them
something — and it doesn't land. Presence listens for that specific child's voice and gently
redirects the parent's attention back to them: the screen dims, audio ducks, and the child's face
appears. All voice processing happens on the device; nothing is ever uploaded. To do this the parent
teaches the app who the child is — and, just as importantly, who *everyone else in the house* is, so
it can tell them apart — either by recording a short session together or by importing existing
videos and voice notes. Because children's voices change as they grow, those profiles are refreshed
every so often rather than set once and forgotten.

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
| "Maya might be calling you" | "ATTENTION VIOLATION DETECTED" |
| The child's photo as the emotional center | Streak counters, scores, guilt metrics |
| Soft, unhurried transitions | Alarm-red, buzzing urgency |
| Neutral language about missed moments | Judgment, shame, "you failed" framing |

The interruption should feel like a gentle tap on the shoulder from someone who loves you — never
like a notification from a compliance system.

## Screens to design

### Onboarding
1. **Welcome / value proposition** — what this does, in one screen, emotionally rather than
   technically.
2. **Privacy explainer** — a genuine selling point. Everything stays on the phone; no account; no
   upload, ever. Be precise here rather than sweeping. While the app is *listening* it never records
   or saves audio — but enrollment is recording, plainly, and those clips are deleted once the voice
   profile is built unless the parent chooses to keep them for a future re-enrollment (screen 22).
   Say both halves. A blanket "nothing is ever recorded" is a promise the very next flow visibly
   breaks, and a parent who catches you at it stops believing the rest of the screen. Must include a
   plain-language note that Android shows a green microphone dot while listening and that this is
   normal and unavoidable.

   This screen also carries the **platform divergence**, and it must not be buried: on Android the
   microphone is off whenever the screen is off — a hard guarantee, not a setting. On iOS without
   Apple's Family Controls entitlement (Tier 1, see the constraints section) that guarantee cannot be
   enforced at all, and the family-time schedule becomes the only bound on listening. An iOS parent
   must be shown the iOS truth, not the Android promise with a footnote.
3. **Staged permission requests** — microphone and notifications first (these alone produce a
   working app). Overlay access, Do-Not-Disturb access, and battery-optimization exemption are
   requested *later, in context*: the first two when the parent configures the response that needs
   them, the third when the app has evidence it is being killed in the background. Each request
   needs a screen explaining what it unlocks in the parent's terms, and each involves sending the
   user into system Settings — design the hand-off and the return. The battery one is the ugliest of
   them: the destination screen is named differently on every manufacturer's phone, so the
   hand-off needs to show the parent what they are looking for rather than promise a specific
   label, and the return has to verify rather than assume it worked.

### Setting up a child
4. **Add a child** — name and photo. The photo is central to the whole experience; make choosing it
   feel meaningful, not like filling a form field.
5. **Choose enrollment mode** — record a short session together, or import existing recordings.
6. **Live enrollment session** — the child speaks a few short prompts. The target is roughly **60
   seconds of *usable speech* per language** — not sixty seconds of wall clock, which with a
   five-year-old in a real living room is a wildly different number, and not sixty seconds total.
   Progress must be counted in the same currency screen 8 uses. Needs live waveform feedback,
   clear progress, per-clip pass/fail, and a retry path. Design this as something a parent and child
   do together and might even enjoy.

   Four things make this screen harder than it looks:
   - **By default it collects both Hebrew and English** (see the bilingual section below), so
     progress is tracked *per language* and the session isn't complete until both are covered. Make
     switching languages feel like part of the game, not a second chore.
   - **A genuinely monolingual household must be able to finish.** Give the screen a "we only speak
     one language at home" exit, and make it an honest one: it closes the session, and it states
     once — calmly, without drama or a warning triangle — that the app will not reliably notice this
     child in the other language, with an offer to add it whenever they like. This is not a
     dismissible tooltip. The choice creates a standing fact on the profile that stays visible on
     the roster, the home screen, and manage-household (screens 11, 13, 21), because the worst
     outcome here is a parent who believes they are covered in both languages and isn't. Without
     this exit the flow simply cannot end for a monolingual family, which is a bug, not a nudge.
   - **One prompt is "call me the way you normally do"** — the child says *"אבא!"* or *"Mummy!"* a
     handful of times. This is the most valuable recording in the whole session and the most natural
     thing to ask for. Give it its own moment rather than burying it in a word list.
   - A one-minute attention span is optimistic for a five-year-old, and you are asking for two
     languages' worth. Design for interruption, partial completion, and resuming later without
     losing progress.
7. **Import enrollment** — pick videos or voice notes, then confirm which detected speech segments
   are the child. Needs a segment list with playback and accept/reject. Segments are **language-
   tagged**, and the screen reports per-language coverage so the parent can see which language is
   short. The same one-language exit as screen 6 applies, and reaches the same standing state.
8. **Enrollment quality feedback** — a clip can be rejected for too much background noise, no
   speech found, clipping, or **more than one person talking over the child**. Show specific,
   non-technical reasons and what to do about it. Progress is reported as **seconds of usable speech
   collected**, not clips recorded — short clips are near-worthless to the recognizer, so a "4 of 6
   clips done" bar would actively mislead.
9. **Enrollment complete** — confirmation with a warm payoff, not a checkmark on a form. This is
   also the natural moment to ask, once and in plain words, whether to keep the recordings so a
   future top-up can build on them (screen 22). Ask; don't assume. The default is to delete them.

### Household setup — do not skip this flow

The recognizer distinguishes the target child far more reliably when it also knows what the *other*
people in the house sound like. Enrolling the parent and any siblings is a **requirement**, not an
optional extra — sibling-vs-target is the hardest confusion in real homes, and the parent's own voice
is the one the microphone hears most, since the phone is in their hand.

This flow is where setup most likely gets abandoned, so it is a genuine design problem: you are
asking a parent to record three people instead of one.

Screens 5–9 were written for the target child. **Every one of them needs a reduced non-target
variant**, and the reduction has to be *visible* rather than merely shorter, or the parent will
assume they are signing up for the full ordeal three times. What relaxes: roughly half the usable
speech; importing is the encouraged path rather than the tolerated one; the quality bar drops, so
clips that would trigger a retry for the target child are simply accepted; and — say this out loud
on the screen — **non-target profiles do not need both languages.** One language is fine. These
profiles only have to be good enough for the recognizer to *subtract* the wrong people, not good
enough to identify the right one, and a parent left guessing whether they cut a corner will either
over-invest or quit.

10. **Why we need everyone's voice** — the honest explanation, in the parent's terms, not the
    engineer's. Something in the spirit of *"Record yourself and their brother or sister too. It's
    how we learn to tell them apart."* A parent who understands the reason will do it; one who is
    asked for no stated reason will quit.
11. **Household roster** — the *setup* view: who's been added, who's still missing, whose voice
    profile is getting stale, and any standing limitation on a profile. A child enrolled in one
    language only carries a quiet permanent line here — *"English only — we may miss her in
    Hebrew"* — with a one-tap way to fix it. Not a red badge, not an error: a fact about the
    profile, stated once. Must convey that non-target profiles are *lighter* work than the target
    child's: shorter sessions, importing is fine, one language is fine, "good enough" genuinely is
    good enough. The visual weight of these entries should be lower than the target child's.
    (Screen 21 is the ongoing counterpart. The split: **this screen answers "what's left to set
    up", 21 answers "change something that already exists".** If your design can't hold that line
    visibly, merge the two and say so in your rationale — two near-identical lists is worse than
    one honest one.)
12. **Re-enrollment prompt** — children's voices drift as they grow, and accuracy quietly decays with
    it. Design a gentle, periodic nudge that surfaces staleness *before* the app starts failing.
    Never alarming, and never framed as the parent's fault. Two things the copy must carry: a new
    session is **added to** what's already there, never replacing it — a parent who fears wiping two
    years of learning will not press the button — and younger children come round more often than
    older ones, which should read as "her voice is changing fastest right now", not as the app
    pestering.

### Everyday use
13. **Home / status** — is it listening right now, and why or why not. A prominent on/off control.
    This screen is seen most; it should be calm and mostly empty. It also carries any standing
    limitation, honestly and without alarm — *"listening in English only"* for a one-language
    profile — because a parent who believes they are covered in both is the one this app fails
    worst.

    On counting: **detected** moments may be shown — *"3 moments today"* is the app working, and
    seeing it is quietly reassuring. **Missed** moments must never be counted, estimated, hinted
    at, or given reserved space. That is a guilt metric and the Tone section forbids it. A
    zero-count day is an unremarkable state, not an empty state with a sad illustration.
14. **The listening notification** — Android requires a persistent notification the whole time the
    app is listening, so this is not chrome to be minimized; it is one of the most-seen surfaces in
    the product, sitting in the shade next to the parent's messages. Design it as reassurance:
    what's on, who it's watching for, one tap to pause. It is also where degraded states have to
    shout, because for a parent who rarely opens the app it may be the only place they would notice
    that the microphone permission is gone or the service was killed.
15. **The intervention overlay** — *the most important screen in the app.* Appears over whatever
    the parent is using: a dimming scrim, the child's photo, their name, a dismiss action, and a
    "not them" correction.

    Three response intents run through the rest of this brief, so define them here:
    - `Notify` — the least intrusive. A notification carrying the child's photo.
    - `Soften` — **the default.** Make the phone less compelling: dim the screen, duck or mute the
      audio, show their face.
    - `Block` — a hard stop until the parent acknowledges it.

    Design the `Soften` (default) and `Block` variants. It must be unmissable but never alarming,
    and dismissible in one tap without hunting.

    The correction control is not a "small button" — small is how you design something you hope
    nobody presses. It should be *quiet* but genuinely easy to hit first time: one tap, no dialog,
    no penalty. A parent tapping "not them" is doing the app a favour, and the control should read
    as an offered hand rather than a bug report form.

    Multi-child: if two children are enrolled, show the one the evidence favours, named. Never a
    group photo, never a carousel, never *"someone is calling you"* — the entire emotional mechanism
    is one specific face. If a second child crosses the bar while the overlay is already up, fold
    them into the overlay that exists rather than stacking a second one on top of it.
16. **Notification variant** — for parents who haven't granted overlay permission. Carries the
    child's photo.
17. **Paused / disabled state** — clearly not listening, easy to resume, no nagging.

### Settings
18. **Response builder** — per child, compose the reaction: dim, mute, show photo, notification
    sound. Three preset intents (`Notify`, `Soften`, `Block`) with custom adjustment underneath.

    Two different kinds of unavailable live here, and they must not look alike. A channel blocked by
    an **ungranted permission** is visibly locked with a one-tap path to unlock. A channel the
    **platform cannot deliver at all** — `Dim` and `Block` on iOS Tier 1 — needs a locked state with
    nothing to tap and an explanation that doesn't imply the parent did something wrong. Reuse one
    lock affordance for both and you get a parent tapping forever at something that will never open.

    One more state: if overlay permission is revoked after setup, the response silently degrades to
    `Notify`. That downgrade must surface here as a banner on the affected child's response —
    *"showing a notification instead of dimming, because overlay access was turned off"* — with the
    re-grant path attached. Quietly doing less than the parent configured is exactly the failure
    this app cannot afford.
19. **Sensitivity** — a slider trading false alarms against missed moments. This is a first-class
    control, not buried in an advanced section. Explain the trade-off in plain words. Note that the
    honest framing is *"more reminders, some wrong"* versus *"fewer reminders, some missed"* — the
    recognizer genuinely cannot be perfect on children's voices, and the UI should not pretend the
    slider has a correct setting.
20. **Family time schedule** — day and time windows that narrow when the app listens. **Optional on
    Android; required on iOS Tier 1, where the app cannot be enabled at all until at least one
    window exists.** That is a gate, not a nag, and it has to read as a reasonable condition rather
    than a punishment — the parent is being asked to draw a boundary the platform will not draw for
    them. Show clearly that on Android the mic *never* runs while the screen is off, regardless of
    schedule; show equally clearly that on iOS Tier 1 that guarantee does not hold, which is
    precisely why the window is mandatory there. Write this screen's explanatory copy as
    platform-conditional rather than a single universal claim.
21. **Manage household** — the *ongoing* counterpart to the roster (11): every enrolled person, the
    target child or children, siblings, and the parent. Edit, re-enroll, disable, delete. Target
    children and non-target profiles need visibly different treatment. Surface profile staleness
    here, plus any one-language limitation and the way to add the missing language. Say on the
    re-enroll control itself that a new session **adds to** the profile rather than replacing it —
    the fear of erasing what already works is what stops people re-enrolling. Per-person control
    over retained enrollment clips lives here too, mirroring screen 22.
22. **Recordings and data** — the one screen that answers *"what is actually stored about my
    family?"*. Enrollment clips become a voice profile and are then deleted; the parent may opt to
    keep them so a future top-up can build on them instead of starting from nothing. Per person,
    defaulted to off, with the trade stated in one line each way. Also here: delete a person's
    clips, delete a profile, delete everything. No dark patterns and no confirmation mazes — but do
    say plainly what deleting a profile costs, because it does not come back.
23. **History** — a simple, non-judgmental log of moments. Deliberately not a dashboard, not a
    score, not a trend line. If it starts to feel like a report card, redesign it.
24. **About and licences** — version, and the open-source attributions. This is not filler you can
    stub out: the speech models this app is built on are licensed on condition that they are
    credited inside the app, so the screen is a legal requirement with a design problem attached.
    Make it legible and unembarrassed rather than a grey wall of scrolling legalese.

### Error and edge states
25. **Microphone permission revoked while running** — the app stops listening; say so plainly and
    offer the one-tap re-grant.
26. **Background service killed by the phone's battery manager** — with recovery guidance, and the
    battery-exemption hand-off if it keeps happening.
27. **Voice model failed to load** — app disabled with a clear explanation. It must never pretend.
28. **No child enrolled yet** — the app cannot be turned on; route warmly into enrollment.
29. **Listening blocked** — another app holds the microphone, or a call is active.
30. **Every enrollment clip rejected** — the total-failure case, where a whole session produced
    nothing usable. Give the reasons in the parent's terms and offer a way out rather than a bare
    retry: try again somewhere quieter, or switch to importing existing recordings instead. A parent
    who has just failed at this in front of their child needs an exit, not a wall.
31. **Empty history, first run.**

## Design system to produce

Deliver a coherent system, not 31 unrelated mockups:

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
- **The app never claims to be listening when it is not.** Every degraded state — permission
  revoked, service killed by the battery manager, model failed to load, another app holding the
  microphone, outside a family-time window — is visible in the UI, on the home screen and in the
  persistent notification, at the moment it happens. A parent quietly trusting a silent, broken app
  is worse than no app at all. This outranks every aesthetic preference in this brief.
- **iOS comes in two tiers, and the difference is user-visible.** *Tier 1* is what ships without
  asking Apple for anything: audio ducking instead of muting, a Live Activity in the Dynamic Island
  instead of an overlay, and a Time-Sensitive notification. On Tier 1, `Dim` and `Block` are **not
  available at all** — iOS cannot dim the screen from the background and cannot draw over other
  apps — and the app cannot tell whether the parent is using their phone, which is why the
  family-time schedule is mandatory there. *Tier 2* requires Apple's Family Controls entitlement,
  which may or may not be granted; it makes `Block` real by shielding the app the parent is in, and
  restores the "only while I'm actually using my phone" rule that Android gets for free. Design
  Android first and Tier 1 as a complete product rather than a degraded one — but design the locked
  states for what Tier 1 genuinely cannot do, and flag which screens need real iOS rework rather
  than a mechanical port.

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
- **Never claim certainty in copy.** This brief resolves the tension rather than leaving it to
  chance: the hedged form wins. *"Maya might be calling you"* is what ships, and it is what the
  tone table and the bidirectional examples below both use. Hedged language can read as weak, so
  fight that with warmth and typography, not by promoting a guess to a fact — the unhedged form is
  a lie roughly one time in five, and it is a lie the parent will personally catch. Test the
  alternative if you like, but do not let certainty back in by accident.
- **It arrives late, and it must not arrive twice.** The recognizer earns its confidence by
  accumulating evidence over several seconds of speech; the budget is to catch four out of five
  real moments *within ten seconds* of the child starting. So the overlay routinely appears after
  the child has been talking a while, and sometimes after they have given up and wandered off. Two
  consequences you cannot retrofit. First, the copy has to survive the delay: a present-tense "Maya
  is calling you *right now*" reads as broken when it lands eight seconds late, while "Maya might be
  calling you" still holds — another reason the hedged form wins. Second, the entrance motion should
  feel like something arriving gently, never like an alarm catching up on lost time. After firing,
  the app goes quiet for a cooldown rather than re-triggering while the child keeps talking, so one
  episode is one overlay, not a stutter — and if the parent dismisses it while the child is still
  going, it does not immediately reappear.
- **The sensitivity slider is the parent's steering wheel** for this trade-off. Treat it as a primary
  control, and explain it without engineering vocabulary.
- **Missed moments must never be surfaced as failures**, by either party. No "you missed 4 moments"
  counters. That would be a guilt metric, which the Tone section forbids.

## Bilingual and bidirectional — English and Hebrew from day one

The first release ships in **English and Hebrew**. Hebrew is right-to-left, so this is a layout
constraint on every screen, not a translation task at the end.

- **Design RTL and LTR as equal first-class layouts.** Mirror the layout, not the child's photo.
  Directional icons (back arrows, progress, chevrons) flip; logos, faces, and media controls do not.
  Retrofitting RTL is far more expensive than designing for it, and the intervention overlay — the
  most layout-sensitive screen in the app — breaks worst.
- **Mixed-direction strings are the normal case, not an edge case.** A child's name may be in Hebrew
  script inside an English sentence or the reverse. *"מאיה might be calling you"* must not scramble.
  Show how you handle this; it is the detail that most often ships broken.
- **Specify a Hebrew type family** alongside the Latin one, and check that the two are visually
  compatible at the same optical size. Hebrew has no case distinction and different vertical metrics,
  so a type scale tuned only on English will look wrong.
- **Copy length differs.** Hebrew is often more compact than English; layouts must not depend on
  either holding a particular width, and neither may go slack when the other is the shorter one.
- **Tone does not survive machine translation**, and tone is this app's most fragile quality. Treat
  the Hebrew copy as something a native speaker writes, not something a translator checks. Where a
  phrase carries emotional weight, note the *intent* so a Hebrew writer can reach for it
  independently rather than translating word-for-word.

Deliver the intervention overlay, home, and at least one enrollment screen in **both languages and
both directions**. The rest can be English-only in the mockups provided the RTL rules are documented.

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
5. The bidirectional layout rules, plus the key screens rendered in both English and Hebrew.

**Format:** produce a single self-contained HTML page showing every screen as a phone-framed
mockup, with a theme toggle for light and dark, plus the design system documented on the same page.
If the Figma MCP is connected, build it in Figma as a proper component library instead.

## Where to start

Design the intervention overlay first, before anything else. It is the whole product in one screen.
If it doesn't feel like a loving nudge, nothing else in the app matters — and everything else should
follow the visual language you establish there.
