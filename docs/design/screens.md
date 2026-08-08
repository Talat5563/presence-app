# Presence — Screen specifications

Working name. Nothing below depends on the word "Presence"; see §Name dependence.

Every screen is listed with its purpose, its states, its component tree, and the `strings.json` keys it references. Token names are from `tokens.json`. Components are specified in `components.md`.

**The rule that outranks every other rule in this document:** the app never claims to be listening when it is not. Every degraded state is visible on the home screen *and* in the persistent notification at the moment it happens.

---

## Reading this document

- `→` means navigation.
- **States** are enumerated exhaustively. A state not listed here should not exist.
- Component tree indentation is containment.
- String keys are given as `namespace.key`.
- `[ANDROID]` / `[iOS-T1]` / `[iOS-T2]` mark platform-conditional content.

---

# Onboarding

## 01 · Welcome

**Purpose.** Establish the emotional premise before asking for anything. One idea, one screen, no carousel of value props, no sign-up.

**States.** Single state. First launch only; never reachable again.

**Tree.**
```
Screen (bg: background)
  StatusBar
  AppMark (48dp, radius 15, accent fill)
  Headline (serif, headline)          onboarding.welcomeHeadline
  BodyLarge                            onboarding.welcomeBody1
  BodyLarge                            onboarding.welcomeBody2
  Spacer (flex)
  PrimaryButton                        onboarding.welcomeCta        → 02
  QuietButton                          onboarding.welcomePrivacy    → 02
  HomeIndicator
```

**Notes.** The headline names the parent's experience, not the feature. It contains no product name and survives a rename. There is no "skip" — both actions go forward, one via the privacy screen.

---

## 02 · Privacy explainer

**Purpose.** A genuine selling point, stated precisely rather than sweepingly. This screen carries the platform divergence and must not bury it.

**States.**

| State | Trigger | Difference |
|---|---|---|
| `android` | Android build | Carries the screen-off microphone guarantee; green-dot note |
| `ios-tier1` | iOS without Family Controls | **No screen-off guarantee exists.** Carries the mandatory-schedule explanation and the orange-dot note |
| `ios-tier2` | Family Controls granted | As Android: the "only while I'm using my phone" rule is restored |

**Tree — `android`.**
```
Screen (bg: background)
  StatusBar
  BackAffordance (min 44dp)
  Headline (serif)                     onboarding.privacyTitleAndroid
  ClaimList
    Claim (icon: check, status.listening)      onboarding.privacyNoAccount
    Claim (icon: check, status.listening)      onboarding.privacyNotRecording
    Claim (icon: fiber_manual_record, status.warning)
                                               onboarding.privacyEnrollmentDoes
    Claim (icon: check, status.listening)      onboarding.privacyMicOffScreenOff
  InfoPanel (surfaceMuted, leading green dot)  onboarding.privacyGreenDot
  Spacer (flex)
  PrimaryButton                        onboarding.privacyAck        → 03a
```

**Tree — `ios-tier1`.** As above, minus `privacyMicOffScreenOff`, plus:
```
  WarningPanel (status.warningSurface)
    Title (icon: info)                 onboarding.privacyIOSNoUsageDetection
    Body                               onboarding.privacyIOSScheduleReason
  InfoPanel (leading orange dot)       onboarding.privacyOrangeDot
```

**Why the amber claim exists.** The third claim is deliberately *not* a check mark. A blanket "nothing is ever recorded" is a promise the very next flow visibly breaks, and a parent who catches you at it stops believing the rest of the screen. Both halves are stated in the same breath, with the recording half visually distinguished so it cannot be skimmed past.

**Why the platform split is structural, not a footnote.** An iOS parent must be shown the iOS truth. The screen-off guarantee is a *hard* Android guarantee — not a setting — and it cannot be enforced at all on Tier 1. On Tier 1 the family-time schedule becomes the only bound on listening, which is why this screen introduces it: screen 20's gate must not arrive as a surprise.

---

## 03 · Staged permissions

**Purpose.** Ask for each capability at the moment it earns its keep, in the parent's terms, and design both the hand-off into system Settings and the return.

**Staging.**

| Permission | When asked | Unlocks |
|---|---|---|
| Microphone | Onboarding, immediately | Everything. Without it there is no app |
| Notifications | Onboarding, immediately | The persistent notification and the `Notify` response |
| Overlay | From the response builder, when the parent selects `Soften`/`Block` | Dimming and the on-screen face |
| Do Not Disturb | From the response builder, when the parent enables muting | Audio muting rather than ducking |
| Battery exemption | **Only after the service has actually been killed**, with evidence | Survival in the background |

Microphone and notifications alone produce a working app. Everything else is contextual, and every one of them is declinable to a working — if lesser — configuration.

### 03a · Microphone

**States.** `explainer` · `system-dialog` (Android's, unstyleable) · `granted` → 03b · `denied` → soft re-ask with the consequence named, never a loop.

**Tree.**
```
Screen
  FeatureIcon (64dp circle, accentSoft, mic)
  Headline (serif)                     onboarding.permMicTitle
  BodyLarge                            onboarding.permMicBody
  BodyLarge                            onboarding.permMicChoose
  Spacer (flex)
  PrimaryButton                        onboarding.permMicCta
  QuietButton                          onboarding.permNotYet
  [overlay] SystemDialogScrim → Android's dialog
```

**Note.** The system dialog cannot be styled, so our screen does the persuading and *names the option to pick* before it appears. A parent should never be guessing inside someone else's UI.

### 03b · Notifications

**States.** `explainer` · `system-dialog` · `granted` · `denied` (the persistent notification is mandatory, so denial means the app cannot listen at all — stated plainly, with the re-grant path).

**Tree.**
```
Screen
  FeatureIcon (notifications)
  Headline (serif)                     onboarding.permNotifTitle
  ExplainerRow (icon: graphic_eq)      onboarding.permNotifQuietName / QuietBody
  ExplainerRow (icon: person)          onboarding.permNotifFaceName / FaceBody
  Spacer (flex)
  PrimaryButton                        onboarding.permNotifCta
  QuietButton                          onboarding.permNotYet
```

**Note.** The mandatory notification is framed as a benefit — it is how the parent pauses without opening anything — rather than apologised for.

### 03c · Overlay access

**States.** `sheet` · `handed-off` · `returned-granted` → 03e verified · `returned-denied` → 03e failed · `declined` → response silently set to `Notify`, *surfaced* on 18 as a banner.

**Tree.**
```
Scrim (scrimSheet)
  BottomSheet (radius sheetTop, surface, sheetLift)
    Grabber
    Headline (serif)                   onboarding.permOverlayTitle
    Body                               onboarding.permOverlayBody
    SettingsPreview (surfaceSunken)
      Label                            onboarding.permOverlayLookFor
      MockSettingsRow (app mark, name, off switch)
      Caption                          onboarding.permOverlayFindUs
    PrimaryButton (trailing open_in_new)  onboarding.permOverlayCta
    QuietButton                        onboarding.permOverlayDecline
```

**Note.** `permOverlayDecline` is a real, named, dignified choice, not a dead end. It lands the parent on a working `Notify` response.

### 03d · Battery exemption — the ugliest hand-off

**Trigger.** Only after ≥2 confirmed service kills. Never speculative.

**States.** `explainer` · `handed-off` · `returned-verified` · `returned-unverified` (offers a second attempt and a "live with it" path).

**Tree.**
```
Screen
  Headline (serif)                     onboarding.permBatteryTitle
  BodyLarge                            onboarding.permBatteryBody   {count}{manufacturer}
  GuidancePanel (surfaceMuted)
    Label                              onboarding.permBatteryVaries
    MockRadioGroup (surface, outline)
      Option (selected, accent ring)   "Unrestricted" + permBatteryPickThis
      Option (dimmed)                  "Optimised"
    Caption                            onboarding.permBatteryWordingDiffers
  Spacer (flex)
  PrimaryButton (trailing open_in_new) onboarding.permBatteryCta
  QuietButton                          onboarding.permNotNow
```

**Why a drawing rather than a label.** The destination screen is named differently on every manufacturer's phone. We *show* the parent the shape of what they are looking for and explicitly say the wording may differ, rather than promising a label we cannot guarantee. The body names the real kill count and the real manufacturer, so the ask is evidently warranted.

### 03e · The return — verified, never assumed

**Purpose.** Re-check the permission on resume. The app must never say "all set" on faith.

**States.** `verified` · `unverified`.

**Tree — `verified`.**
```
Card (surface, outline)
  Row (icon: check_circle, status.listening)  onboarding.returnVerifiedTitle
  Body                                        onboarding.returnVerifiedBody
  PrimaryButton                               onboarding.returnCarryOn
```

**Tree — `unverified`.**
```
Card (status.warningSurface)
  Row (icon: error, status.warning)           onboarding.returnFailedTitle
  Body                                        onboarding.returnFailedBody
  PrimaryButton (trailing open_in_new)        onboarding.returnTryAgain
  QuietButton                                 onboarding.returnUseNotification
```

**Note.** Failure blames the list, not the parent ("it's easy to miss in that list"), and always offers the lesser response as a way out. Never a dead-end retry.

---

# Setting up a child

## 04 · Add a child

**Purpose.** The photo is the emotional centre of the entire product, so it is the screen's subject rather than a field on a form.

**States.** `empty` · `photo-chosen` · `named` (CTA enables) · `editing` (from 21).

**Tree.**
```
Screen
  BackAffordance
  Headline (serif)                     addChild.title
  Body                                 addChild.photoPrompt
  PhotoSlot (158dp, radius sheet, dashed outlineStrong)   addChild.photoEmpty
  Row: OutlineChip photo_library / OutlineChip photo_camera
                                       addChild.fromPhotos / fromCamera
  FieldLabel                           addChild.nameLabel
  TextField (56dp, radius card)
  Caption                              addChild.nameHelp
  Spacer (flex)
  PrimaryButton                        addChild.next               → 05
```

**Note.** The photo slot sits *above* the name field so the screen reads as "choose a face", not "fill in a record". At ≥1.6× font scale the slot shrinks before any text does.

---

## 05 · Choose enrollment mode

**Purpose.** Offer recording and importing as genuine equals, and — in the non-target variant — make the reduced ask *visible* rather than merely shorter.

**States.**

| State | Difference |
|---|---|
| `target` | Recording carries the accent border. Tags: 2–4 minutes, both languages |
| `non-target` | **Importing carries the accent border.** A sage relaxation panel states all three reductions. Tags: 30 seconds, one language fine |

**Tree — `target`.**
```
Screen
  BackAffordance
  Headline (serif)                     enrollMode.titleTarget
  Body                                 enrollMode.subtitleTarget
  ChoiceCard (accent border, icon mic)
    Title / Body                       enrollMode.recordName / recordBody
    TagRow (accentSoft)                enrollMode.tagDuration / tagBothLanguages
  ChoiceCard (outline, icon perm_media)
    Title / Body                       enrollMode.importName / importBody
    TagRow (surfaceMuted)              enrollMode.tagNothingToRecord / tagStaysOnPhone
```

**Tree — `non-target`.** Same skeleton, with the cards' emphasis swapped, and inserted above them:
```
  RelaxationPanel (status.listeningSurface, icon bolt)
                                       enrollMode.nonTargetRelaxations
```

**Why the reduction must be visible.** Screens 05–09 all need a reduced non-target variant, because a parent who cannot see the difference will assume they are signing up for the full ordeal three times. What relaxes: roughly half the usable speech; importing becomes the *encouraged* path; the quality bar drops so clips that would trigger a retry for the target child are simply accepted; and **non-target profiles do not need both languages** — said out loud, on the screen. A parent left guessing whether they cut a corner will either over-invest or quit.

---

## 06 · Live enrollment session

**Purpose.** Collect roughly **60 seconds of usable speech per language** — not 60 seconds of wall clock, which with a five-year-old in a real living room is a wildly different number, and not 60 seconds total. Designed as something a parent and child do together.

**States.**

| State | Notes |
|---|---|
| `idle` | Prompt shown, not recording |
| `recording` | Live waveform, accent bars |
| `clip-accepted` | Per-clip verdict + seconds credited |
| `clip-rejected` | Reason + retry, progress unchanged |
| `language-complete` | That language's card flips to sage + check |
| `callout` | The "call me the way you normally do" moment, its own layout |
| `paused` | Resumable, nothing lost |
| `session-complete` | → 09 |
| `one-language-exit` | Sheet; → standing profile limitation |

**Tree — `recording`.**
```
Screen (bg: dark.background — dark by default)
  TopRow: close · QuietButton         enrollSession.pauseAndResume
  LanguageProgressPair
    LanguageCard (active: accent border)   he · seconds
    LanguageCard (complete: sage)          en · check
  PromptLabel                          enrollSession.sayThisIn
  PromptText (serif, 30dp, script-matched family)
  Waveform (live, 12 bars)
  ClipVerdict (icon check_circle, sage) enrollSession.clipGood
  Spacer (flex)
  RecordButton (84dp, accent, 10dp glow ring)
  Row: QuietChip replay / QuietChip arrow_forward
                                       enrollSession.tryAgain / nextPrompt
```

**Tree — `callout`.**
```
Screen (radial gradient, accent-tinted)
  Overline (accent-tinted)             enrollSession.calloutLabel
  Headline (serif, 32dp, centred)      enrollSession.calloutPrompt   {name}
  BodyLarge (centred)                  enrollSession.calloutBody
  Waveform (6 bars, larger)
  ScriptChipRow                        "אמא!" "אבא!" "Mummy!"
  RecordButton (84dp, 14dp glow)
  Caption                              enrollSession.calloutProgress
```

**Four things that make this screen harder than it looks.**

1. **Two languages by default.** Progress is tracked *per language* and the session is not complete until both are covered. Switching language is a move in the game, not a second chore — the two language cards sit side by side and the active one is simply the one with the accent border.
2. **A monolingual household must be able to finish.** `enrollSession.oneLanguageExit` is always available. Without it the flow cannot end for a monolingual family, which is a bug, not a nudge.
3. **The callout prompt gets its own moment.** *"Call me the way you normally do"* produces the most valuable recording in the whole session and is the most natural thing to ask for. It is addressed to the child by name, and the script chips show that either language counts.
4. **Designed for interruption.** A one-minute attention span is optimistic for a five-year-old, and this asks for two languages' worth. "Pause & finish later" is permanently in the corner; progress bars only ever go up; nothing is lost.

**The one-language exit (`06c`).**
```
Scrim (scrimSheet)
  BottomSheet (dark.surface)
    Grabber
    Headline (serif)                   enrollSession.oneLanguageExitTitle
    BodyLarge                          enrollSession.oneLanguageExitBody
    InsetPanel (surfaceRaised, icon translate)
                                       enrollSession.oneLanguageExitCaveat
    PrimaryButton                      enrollSession.oneLanguageConfirm
    QuietButton                        enrollSession.oneLanguageReconsider
```
Not amber. No warning triangle. Not a dismissible tooltip — a calm sheet the parent must answer. It creates a **standing fact** on the profile that then appears on 11, 13 and 21, because the worst outcome this app can produce is a parent who believes they are covered in both languages and isn't.

---

## 07 · Import enrollment

**Purpose.** Pick media, then confirm which detected speech segments are the child. Segments are language-tagged and coverage is reported per language.

**States.** `picking` · `analysing` · `segment-list` · `segment-playing` · `segment-accepted` · `segment-rejected` (undoable) · `no-speech-found` · `one-language-exit` (same destination as 06c) · `complete` → 09.

**Tree.**
```
Screen
  BackAffordance
  Headline (serif)                     enrollImport.title
  Body                                 enrollImport.subtitle
  LanguageCoveragePair                 he: 12s "still short" · en: 54s "nearly there"
  SegmentList
    SegmentRow (play_circle, range, language tag, mini-waveform, accept/reject 44dp)
    SegmentRow …
    SegmentRow (rejected: struck range, reason, Undo)
                                       enrollImport.droppedAdult / undo
  Spacer (flex)
  PrimaryButton                        enrollImport.addAnother
  QuietButton                          enrollSession.oneLanguageExit
```

**Note.** Per-language coverage sits *above* the list so the parent can see which language is short before they start judging segments. Rejection is always undoable.

---

## 08 · Enrollment quality feedback

**Purpose.** Explain rejections in specific, non-technical terms, and report progress in the only honest currency.

**Rejection reasons.** Each pairs a plain-language cause with an actionable fix.

| Reason | Key | Fix key |
|---|---|---|
| More than one person talking over the child | `reasonOverlap` | `reasonOverlapFix` |
| Too much background noise | `reasonNoise` | `reasonNoiseFix` |
| No speech found | `reasonNoSpeech` | `reasonNoSpeechFix` |
| Clipping | `reasonClipping` | `reasonClippingFix` |

**Tree.**
```
Screen (dark)
  Close
  Headline (serif)                     enrollQuality.title
  LanguageProgressStack                enrollSession.progressPerLanguage (per language)
  Overline                             enrollQuality.rejectedHeading
  RejectionList
    RejectionCard (icon record_voice_over, status.warning)  reasonOverlap + fix
    RejectionCard (icon graphic_eq)                          reasonNoise + fix
    RejectionCard (icon volume_off)                          reasonNoSpeech + fix
  Spacer (flex)
  PrimaryButton                        enrollQuality.carryOnIn
  QuietButton                          enrollQuality.finishLater
```

**Why seconds, not clips.** Short clips are near-worthless to the recognizer, so a "4 of 6 clips done" bar would actively mislead. Progress is **seconds of usable speech collected**, per language, in the same currency screen 06 uses. Bars only ever go up — a rejected clip removes nothing, and every rejection reason says so.

---

## 09 · Enrollment complete

**Purpose.** A warm payoff, and the one honest ask about retaining recordings.

**States.** `both-languages` · `one-language` (body names the covered language and the gap) · `non-target` (shorter, no language claim).

**Tree.**
```
Screen (centred column)
  PhotoCard (132dp, radius sheet, e2)
  Headline (serif, 30dp, centred)      enrollComplete.title
  BodyLarge (centred)                  enrollComplete.bodyBoth | bodyOne
  SageChip (icon check_circle)         enrollComplete.collected
  AskCard (surface, outline)
    Title                              enrollComplete.keepAskTitle
    Body                               enrollComplete.keepAskBody
    ToggleRow (default OFF)            enrollComplete.keepToggle / keepOffState
  Spacer (flex)
  PrimaryButton                        enrollComplete.ctaHousehold   → 10
```

**Note.** The retention question is asked **once**, in plain words, with the trade stated both ways, and **defaults to delete**. Ask; don't assume. The forward action leads into the household flow rather than dropping the parent on home, because household setup is a requirement and this is the moment of most goodwill.

---

## 30 · Every clip rejected

**Purpose.** The total-failure case. A parent who has just failed at this in front of their child needs an exit, not a wall.

**States.** Single state, with the reason list populated from the actual failures.

**Tree.**
```
Screen
  Close
  Headline (serif)                     enrollQuality.allRejectedTitle
  BodyLarge                            enrollQuality.allRejectedBody
  ReasonList
    ReasonPanel (warningSurface, graphic_eq)
    ReasonPanel (warningSurface, record_voice_over)
  Overline                             enrollQuality.allRejectedWaysForward
  ChoiceCard (accent border, perm_media)  allRejectedImportName / Body   → 07
  ChoiceCard (outline, bedtime)           allRejectedQuietName / Body    → 06
```

**Note.** No bare "retry" anywhere on the screen. "That's on us, not on {name}" is the load-bearing line and must survive any rewrite. The room is the subject of the headline — not the child, not the parent.

---

# Household setup

## 10 · Why we need everyone's voice

**Purpose.** Give the honest reason before the ask. A parent who understands will do it; one asked with no stated reason will quit.

**States.** Single state, with the sibling's name interpolated when known.

**Tree.**
```
Screen
  BackAffordance
  Headline (serif)                     household.whyTitle
  BodyLarge                            household.whyBody1
  BodyLarge                            household.whyBody2
  RelaxationPanel (listeningSurface)
    Label                              household.whyLighterHeading
    Row (schedule)                     household.whyLighterTime
    Row (translate)                    household.whyLighterLang
    Row (perm_media)                   household.whyLighterImport
  Spacer (flex)
  PrimaryButton                        household.whyCta
  QuietButton                          household.whyDefer
```

**Note.** Enrolling the parent and siblings is a **requirement**, not an optional extra — sibling-vs-target is the hardest confusion in real homes, and the parent's own voice is the one the microphone hears most, since the phone is in their hand. But the requirement is never stated as a rule; deferring is a real option and the roster keeps the thread.

---

## 11 · Household roster — *the setup view*

**Purpose.** Answer "**what's left to set up**". This screen has a finish line and stops existing once everyone is added.

**States.** `incomplete` · `complete` (the progress row and the add-prompt rows disappear; the screen becomes a summary and the CTA changes to "Start listening").

**Tree.**
```
Screen
  BackAffordance
  Headline (serif)                     household.rosterTitle
  ProgressRow (track surfaceSunken, fill accent)   household.rosterProgress
  Overline                             household.rosterSectionTargets
  TargetPersonCard (52dp avatar, 18dp name, outline border)
    Name · StatusLine (sage, check_circle)          household.rosterReady
  TargetPersonCard
    Name · LimitationLine (translate, subtle)       home.oneLanguageStanding
         · FixAction (accent)                       home.addLanguage
  Overline                             household.rosterSectionOthers
  Subhead (faint)                      household.rosterOthersSubhead
  NonTargetRow (38dp avatar, 16dp regular, surfaceMuted flat)
    Name + common.you · StatusLine                 household.rosterDone
  NonTargetRow (icon person_add)
    household.rosterAddPerson · household.rosterAddHint
  Spacer (flex)
  PrimaryButton                        household.rosterAddPerson
  QuietButton                          household.rosterStartAnyway
```

**Two visual tiers, and why.** Target children get 52dp avatars, 18dp medium names, and bordered cards. Everyone else gets 38dp avatars, 16dp regular names, and flat tinted rows. The lighter row *is* the message that it is lighter work — and the section subhead says so in words as well, because visual weight alone is not a promise.

**The one-language standing line.** A child enrolled in one language only carries a quiet permanent line here — *"English only — we may miss her in Hebrew"* — with a one-tap fix. Not a red badge, not an error: a fact about the profile, stated once.

---

## 12 · Re-enrollment prompt

**Purpose.** Surface staleness *before* accuracy visibly decays, without alarm and without blame.

**States.** `stale-young` (frequent, framed as fast development) · `stale-older` (rare) · `deferred`.

**Trigger cadence.** Under 6 years: every ~5 months. 6–10: ~9 months. Over 10: ~18 months. The cadence itself is what makes "her voice is changing fastest right now" true rather than rhetorical.

**Tree.**
```
Screen
  Close (trailing)
  PhotoCard (104dp)
  Headline (serif, centred)            household.reenrollTitle
  BodyLarge                            household.reenrollBody   {age}{months}
  AdditivePanel (surface, outline, icon add_circle, sage)
                                       household.reenrollAdditive
  Caption (faint)                      household.reenrollOlderNote
  Spacer (flex)
  PrimaryButton                        household.reenrollCta
  QuietButton                          household.reenrollDefer
```

**Two things the copy must carry.** A new session is **added to** what is already there, never replacing it — a parent who fears wiping two years of learning will not press the button, which is why the additive promise gets its own card rather than a clause. And younger children come round more often, which must read as *"her voice is changing fastest right now"* rather than as the app pestering.

---

## 21 · Manage household — *the ongoing view*

**Purpose.** Answer "**change something that already exists**". No progress, no finish line, organised by person rather than by what is outstanding.

**States.** `default` · `person-menu-open` · `confirm-delete-profile` · `confirm-disable`.

**How this differs from 11 — the line the design must hold visibly.**

| | 11 · Roster | 21 · Manage |
|---|---|---|
| Question answered | What's left to set up | Change something that exists |
| Progress indicator | Yes — "2 of 3" | **None** |
| Grouping | By what's outstanding | By person |
| Per-row affordance | Chevron / add prompt | `more_vert` menu on every row |
| Lifetime | Disappears when complete | Permanent |
| Others section label | "So we can tell them apart" | "Voices we subtract" |
| Primary action | Add the next person | Add someone (outline, not primary) |

**Tree.**
```
Screen
  AppBar (back + household.manageTitle)
  TargetPersonCard
    Avatar 52 · Name 18 · Summary                  household.manageProfileSummary
    Menu (more_vert)
    InsetNote (surfaceSunken, graphic_eq)
      household.staleNote · household.topUpAdditive
  TargetPersonCard
    …
    InsetNote (translate)
      household.oneLanguageNote · home.addLanguage
  Overline                             household.manageSectionOthers
  NonTargetRow (38dp, surfaceMuted, more_vert)
  NonTargetRow
  Spacer (flex)
  OutlineButton (person_add)           household.addSomeone
```

**Menu contents per person.** `household.editPerson` · `household.topUpAdditive` · retention toggle (mirrors 22) · `household.disablePerson` · `household.deletePerson` (destructive).

**Note.** "Nothing is replaced" is written on the re-enroll control itself, not in a help text — the fear of erasing what already works is what stops people re-enrolling. Per-person control over retained clips lives here too, mirroring 22.

---

# Everyday use

## 13 · Home / status

**Purpose.** Answer one question: is it listening right now, and why or why not. The most-seen screen, so calm and mostly empty.

**States.**

| State | Ring | Headline | Notes |
|---|---|---|---|
| `listening` | Sage, filled `graphic_eq` | `home.listeningFor` | Toggle on |
| `listening-limited` | Sage | `home.listeningFor` | + one-language standing panel |
| `paused` | Neutral, hollow, `pause` | `home.paused` | → 17 |
| `outside-window` | Neutral | `home.paused` variant | Names the next window |
| `mic-revoked` | Amber, `mic_off` | `degraded.micRevokedTitle` | One-tap re-grant |
| `service-killed` | Amber, `battery_alert` | `degraded.killedTitle` | → 26 |
| `model-failed` | Amber, `hearing_disabled` | `degraded.modelFailedTitle` | → 27 |
| `mic-busy` | Neutral, `phone_in_talk` | `degraded.blockedTitle` | → 29, self-healing |
| `no-child` | — | `degraded.noChildTitle` | → 28; **toggle absent, not disabled** |

**Tree — `listening`.**
```
Screen (bg: background, E0)
  StatusBar (+ OS mic dot, illustrative)
  TopBar: AppMark · IconButton history · IconButton tune
  StatusRing (132dp outer, 96dp mid, 56dp core)
  Headline (serif, 30dp, centred)      home.listeningFor
  Body (centred)                       home.listeningUntil
  ListeningToggle (72×40)
  MomentsRow (surface, outline, avatar 40)   home.momentsToday
  [if limited] LimitationPanel (surfaceMuted, translate)
      home.oneLanguageStanding · home.addLanguage
  Spacer (flex)
  HomeIndicator
```

**On counting.** **Detected** moments may be shown — *"3 moments today"* is the app working, and seeing it is quietly reassuring. **Missed** moments must never be counted, estimated, hinted at, or given reserved space; that is a guilt metric and the tone constraint forbids it. A zero-count day renders as `home.momentsTodayNone` — an unremarkable state, not an empty state with a sad illustration.

**Redundant state signalling.** Listening status is carried by three signals simultaneously: dot fill (solid vs hollow), colour, and words. It survives colour blindness and a half-second glance.

---

## 14 · The persistent notification

**Purpose.** Android requires this the whole time the app is listening, so it is designed as a deliberate surface, not tolerated as chrome. It sits in the shade next to the parent's messages all day.

**States.**

| State | Channel importance | Sound | Content |
|---|---|---|---|
| `listening` | LOW | Silent | Who is being listened for, until when, pause + turn off |
| `paused` | LOW | Silent | Paused, resume action |
| `outside-window` | LOW | Silent | Next window named |
| `degraded-*` | DEFAULT | **Once** | What stopped, when, will not self-restart, two recovery actions |

**Tree — `listening`.**
```
NotificationShade
  Notification (surface: dark surfaceRaised, radius 22)
    HeaderRow: AppMark 14dp · notification.appName · StatusPill (sage dot + statusListening)
    Title                              notification.ongoingTitle
    Body                               notification.ongoingBody
    Divider
    ActionRow: notification.pauseHour · notification.turnOff
```

**Tree — `degraded`.** Same skeleton on `status.warningSurface` with `warningOutline`, an `error` glyph in the header, `notification.degradedTitle` + `degradedBatteryBody`, and actions `startAgain` (filled amber) · `stopThisHappening`.

**Why degraded shouts here.** For a parent who rarely opens the app, this shade row may be the **only** place they would learn the microphone permission is gone or the service was killed. So the degraded state raises the channel to DEFAULT importance, re-sorts to the top of the shade, and makes a sound exactly once. It is the only time this notification ever makes a noise.

**[iOS-T1]** There is no equivalent. iOS needs no always-on notification, so this surface's reassurance job moves to the Live Activity and the home screen.

---

## 15 · The intervention overlay — *the most important screen in the app*

**Purpose.** Redirect the parent's attention to their child. If it does not feel like a loving nudge, nothing else in the app matters.

**Response intents.**

| Intent | Behaviour |
|---|---|
| `Notify` | Least intrusive. A notification carrying the child's photo (screen 16) |
| `Soften` | **The default.** Dim via translucent overlay, duck or mute audio, show their face |
| `Block` | A hard stop until the parent acknowledges — but never a trap |

**States.**

| State | Notes |
|---|---|
| `soften` | Scrim 0.82 dark / 0.72 light, photo 216dp, ducked-audio chip |
| `block` | Opaque radial ground, photo 248dp, name 48dp, three actions |
| `soften-multi` | Second child folds into the existing card — 200dp + 78dp avatar |
| `dismissing` | 240ms fade, no scale |
| `corrected` | Same exit; correction recorded silently, no confirmation, no penalty |
| `reduced-motion` | 160ms cross-fade only |

**Tree — `soften`.**
```
HostAppContent (untouched, behind)
Scrim (color.{theme}.scrim)
  Column (centred)
    PhotoCard (216dp, radius sheet, e3)
    Name (serif, 44dp)                 — the child's name, largest type in the product
    HedgedLine (bodyLarge 20dp)        overlay.hedgedCall
    DuckedChip (icon volume_down)      overlay.duckedNotice
  ButtonColumn
    OverlayPrimary (56dp, light fill)  overlay.dismiss
    OverlayCorrection (56dp, alpha)    overlay.notThem
    Caption (centred)                  overlay.tapOutside
```

**Tree — `block`.** As above with a radial-gradient ground instead of a scrim, and:
```
  ButtonColumn
    OverlayPrimary                     overlay.dismissBlock
    Row
      OverlayCorrection (flex 1)       overlay.notThem
      OverlayTertiary (flex 1)         overlay.returnToPhone
    Caption                            overlay.blockNeverTraps
```

**Motion — specified precisely, because duration and easing carry most of the emotional weight.**

| Step | Duration | Delay | Easing | Properties |
|---|---|---|---|---|
| Scrim in | 520ms | 0 | `gentleIn` | opacity 0 → 0.82 |
| Photo card in | 560ms | 140ms | `arrive` | scale 0.94 → 1, translateY 12 → 0, opacity 0 → 1 |
| Name & copy in | 420ms | 280ms | `gentleIn` | opacity only |
| Audio duck | 400ms | 0 | linear | → 20% |
| **Total entrance** | **~700ms** | | | |
| Exit (any) | 240ms | 0 | `exit` | opacity, no scale |
| Audio restore | 600ms | 0 | linear | → 100% |
| Second child folds in | 380ms | 0 | `standard` | card grows; avatar fades + slides 16dp inward |
| Reduced motion | 160ms | 0 | linear | cross-fade only |

It breathes in. The scrim starts first so the room dims and the face arrives *into* it. No bounce and no overshoot — overshoot reads as excitement. Leaving is brisk, because the parent has already turned to their child and the app should get out of the way. Under reduced motion the overlay still **fades** rather than cutting in: an instant cut is the one thing that would feel like an alarm.

**The correction control.** Not a "small button" — small is how you design something you hope nobody presses. It is 56dp, full width, warm-tinted, one tap, no dialog, no "are you sure", no penalty. A parent tapping "not them" is doing the app a favour, and it should read as an offered hand rather than a bug report form. It sits directly beneath the dismiss action at identical size, differing only in fill.

**Multi-child.** Show the one the evidence favours, named. Never a group photo, never a carousel, never *"someone is calling you"* — the entire emotional mechanism is one specific face. If a second child crosses the bar while the overlay is already up, fold them into the overlay that exists rather than stacking a second one on top of it.

**Arriving late.** The recognizer accumulates evidence over several seconds; the budget is to catch four of five real moments within ten seconds of the child starting. So the overlay routinely appears after the child has been talking a while, and sometimes after they have wandered off. Two consequences that cannot be retrofitted: the copy has to survive the delay (a present-tense *"is calling you right now"* reads as broken at eight seconds; the hedged form still holds), and the entrance must feel like something arriving gently rather than an alarm catching up. After firing, the app goes quiet for a **45-second cooldown** — one episode is one overlay, not a stutter — and a dismissal does not let it immediately reappear.

**Accessibility.** `overlay.a11yOverlay` is announced as an assertive live region. Both actions are ≥56dp. Focus lands on the dismiss action. The scrim itself is a dismiss target. Never relies on colour.

**[iOS-T1] Needs real rework, not a port.** No overlay exists. Tier 1 is a Live Activity in the Dynamic Island plus a Time-Sensitive notification carrying the photo — a fundamentally different surface with its own size and interaction budget. `Dim` and `Block` are unavailable entirely.

---

## 16 · Notification variant (`Notify`)

**Purpose.** The whole reminder for parents without overlay permission. Carries the child's photo.

**States.** `default` · `expanded` · `corrected`.

**Tree.**
```
NotificationShade (over host app)
  Notification (radius sheet, e3)
    HeaderRow: AppMark · appName · "now"
    Row
      Photo (76dp, radius 16)
      Column: Name (serif 26dp) · overlay.hedgedCall
    Divider
    ActionRow: overlay.dismiss · overlay.notThem
```

**Note.** Both actions are notification actions, so the correction works without unlocking or opening the app. Sound is a single soft two-note chime, never the default alert tone.

---

## 17 · Paused

**Purpose.** Make clear it is not listening, make resuming easy, and nag about nothing.

**States.** `paused-indefinite` · `paused-timed` (an auto-resume chip is selected) · `resuming`.

**Tree.**
```
Screen
  TopBar (mark neutral, history, tune)
  StatusRing (neutral, hollow, pause glyph)
  Headline (serif)                     home.paused
  Body (centred)                       home.pausedBody
  ListeningToggle (off)
  Label (faint)                        "Or start again automatically"
  ChipRow: in an hour · this evening · tomorrow
  Spacer (flex)
  HomeIndicator
```

**Note.** Paused is a deliberate choice by the parent, so it is styled completely neutrally — never as a problem, never with amber, and with no reminder to turn it back on.

---

# Settings

## 18 · Response builder

**Purpose.** Compose the reaction per child, from three preset intents with per-channel adjustment underneath.

**States.**

| State | Notes |
|---|---|
| `android-full` | All channels available |
| `android-overlay-missing` | Dim shows an **openable** lock + action |
| `android-degraded` | Overlay revoked after setup → downgrade banner |
| `ios-tier1` | Dim and Block **permanently unavailable**, non-openable |
| `ios-tier2` | Block real; Dim still unavailable |

**Tree.**
```
Screen
  AppBar (back + response.title)
  [if degraded] DowngradeBanner (warningSurface, error)
      response.downgradeBanner · response.downgradeFix
  PresetRow
    PresetTile notify / soften (selected: accentSoft + accent border) / block
  Overline                             response.channelsHeading
  ChannelList
    ChannelRow (person, toggle)        response.channelPhoto
    ChannelRow (volume_down, toggle)   response.channelDuck
    LockedRowOpenable (lock_open_right + OutlineChip)
        response.channelDim · channelDimNeedsPermission · common.turnOn   → 03c
    ChannelRow (music_note, toggle)    response.channelChime
```

**The two kinds of unavailable, and why they must not look alike.**

| | Openable (permission) | Not openable (platform) |
|---|---|---|
| Glyph | `lock_open_right` | `block` — *a different glyph entirely* |
| Text | Full contrast | `status.unavailable` |
| Background | `surface` | `surfaceMuted` (sunken) |
| Trailing control | Yes — one tap to unlock | **None** |
| Copy | Names the permission | Names the platform limit, absolves the parent |

Reuse one lock affordance for both and you get a parent tapping forever at something that will never open.

**The silent-downgrade banner.** If overlay permission is revoked after setup, the response degrades to `Notify`. That downgrade **must** surface here, naming the substitution and its cause in one sentence, with the re-grant path attached. Quietly doing less than the parent configured is exactly the failure this app cannot afford.

---

## 19 · Sensitivity

**Purpose.** Hand the parent the steering wheel for the false-alarm / missed-moment trade-off. A first-class control, not an advanced setting.

**States.** `low` · `mid` · `high` — each with its own live sentence. No `recommended` marker exists.

**Tree.**
```
Screen
  AppBar (back + sensitivity.title)
  BodyLarge                            sensitivity.explainer
  SliderCard (surface, outline)
    Slider (track surfaceSunken, fill accent, thumb 32dp)
    EndLabels                          sensitivity.lowEnd / highEnd
  LivePanel (surfaceMuted)             sensitivity.liveLow | liveMid | liveHigh
  Caption (faint)                      sensitivity.noRightAnswer
```

**Note.** The honest framing is *"more reminders, some wrong"* versus *"fewer reminders, some missed"*. No numbers, no percentages, no confidence scores, and no implication that a correct setting exists — because the recognizer genuinely cannot be perfect on children's speech and the UI must not pretend otherwise.

---

## 20 · Family time schedule

**Purpose.** Narrow when the app listens. **Optional on Android; required on iOS Tier 1.**

**States.**

| State | Platform | Notes |
|---|---|---|
| `android-empty` | Android | Listens whenever the screen is on. Perfectly valid |
| `android-windows` | Android | Windows listed, each toggleable |
| `ios-gate-unmet` | iOS-T1 | **The app cannot be enabled.** CTA disabled with its condition stated |
| `ios-gate-met` | iOS-T1 | Behaves as Android |
| `ios-tier2` | iOS-T2 | Gate relaxes to a setting — **must demote gracefully** |

**Tree — `android-windows`.**
```
Screen
  AppBar (back + schedule.title)
  Body                                 schedule.androidOptional
  GuaranteePanel (listeningSurface, phonelink_off)
                                       schedule.androidScreenOff
  WindowList
    WindowCard (title, toggle, time range, DayPicker 30dp circles)
    WindowCard (collapsed, toggle off)
  Spacer (flex)
  OutlineButton (add)                  schedule.addWindow
```

**Tree — `ios-gate-unmet`.**
```
Screen (iOS chrome)
  BackAffordance (HIG chevron + label)
  Headline (serif)                     schedule.iosGateTitle
  BodyLarge                            schedule.iosGateBody
  AskCard (surface, outline)
    schedule.iosGateAsk · schedule.iosGateReassure
  SuggestionList
    SuggestionRow (schedule, add)      schedule.suggestAfterSchool
    SuggestionRow                      schedule.suggestWeekendAM
  Spacer (flex)
  PrimaryButton (DISABLED)             home.listeningFor-equivalent CTA
  Caption (centred)                    schedule.iosGateBlocked
```

**Platform-conditional copy — never a single universal claim.** On Android the microphone **never** runs while the screen is off, regardless of schedule; that guarantee is stated. On iOS Tier 1 the guarantee does not hold, which is precisely why the window is mandatory; that is stated instead. The disabled CTA carries its own unlock condition in a line beneath it, so the gate reads as a reasonable condition rather than a mystery or a punishment — the parent is being asked to draw a boundary the platform will not draw for them.

---

## 22 · Recordings and data

**Purpose.** Answer *"what is actually stored about my family?"* in one screen.

**States.** `default` · `confirm-delete-clips` · `confirm-delete-profile` · `confirm-delete-all`.

**Tree.**
```
Screen
  AppBar (back + data.title)
  Body                                 data.summary
  RetentionList
    ToggleRow (per person)             data.keepClipsFor · keepOn | keepOff
    …
  Overline                             data.deleteHeading
  DeleteList
    DeleteRow (delete)                 data.deleteClips
    DeleteRow (person_off)             data.deleteProfile · deleteProfileCost
  Spacer (flex)
  DestructiveButton                    data.deleteEverything
```

**Note.** Every retention toggle's subtitle states the consequence *in its current position*, so the trade is legible without tapping. Default is off, per person. Deleting a profile says plainly what it costs, because it does not come back — no dark patterns, no confirmation mazes, and no false comfort either. `status.destructive` appears here and nowhere else in the product.

---

## 23 · History

**Purpose.** A simple, non-judgmental log of moments. Deliberately not a dashboard, not a score, not a trend line.

**States.** `populated` (day-grouped) · `empty` (screen 31).

**Tree.**
```
Screen
  AppBar (back + history.title)
  DaySection
    Overline                           history.today
    MomentRow (avatar 42, name+time, outcome)
        history.entryTime · entryWentToThem | entryDismissed | entryCorrected
    …
  DaySection
    Overline                           history.yesterday
    MomentRow
```

**Note.** Day headings, not counts. No totals, no week-on-week, no streaks, no charts. A quiet day is simply a shorter list — never *"only 1 moment"* and never a comparison. Corrections are logged in exactly the same neutral voice as successes: a correction is a contribution, not a defect. **If it starts to feel like a report card, redesign it.**

---

## 24 · About and licences

**Purpose.** The speech models are licensed on condition of in-app credit, so this screen is a legal requirement with a design problem attached.

**States.** `summary` · `full-text` (per dependency, scrollable).

**Tree.**
```
Screen
  AppBar (back + "About")
  IdentityRow: AppMark 48dp · about.appName · about.version
  Body                                 about.creditsIntro
  CreditList
    CreditCard: "Silero VAD" · about.creditVadWhat · "MIT licence"
    CreditCard: "SpeechBrain ECAPA-TDNN" · about.creditSpeakerWhat · "Apache 2.0 · attribution required"
    CreditCard: "ONNX Runtime" · about.creditRuntimeWhat · "MIT licence"
  Spacer (flex)
  TextAction (accent)                  about.fullLicences
```

**Note.** Each dependency gets a card, a plain-English line about what it does, and its licence — so the legally required screen is actually legible rather than a grey wall of scrolling legalese. Full texts are one tap away rather than dumped inline. Framed as gratitude, which is what makes an obligation readable.

---

# Degraded states

All five follow one shape: **what stopped · when · what it means for the promise · one action.** Every one appears on home *and* in the persistent notification at the moment it happens.

## 25 · Microphone permission revoked while running

Rendered as home state `mic-revoked` (see 13). Amber ring with `mic_off`, `degraded.micRevokedTitle`, `micRevokedBody`, a reassurance line naming the time and confirming profiles are untouched, and a one-tap re-grant.

## 26 · Background service killed by the battery manager

**Tree.**
```
Screen (centred)
  FeatureIcon (112dp, warning-tinted, battery_alert)
  Headline (serif)                     degraded.killedTitle
  BodyLarge (centred)                  degraded.killedBody   {time}
  WarningPanel (error)                 degraded.killedRepeat   {count}
  Spacer (flex)
  PrimaryButton                        degraded.killedCta
  OutlineButton (open_in_new)          notification.stopThisHappening   → 03d
```
Resuming comes first; the permanent fix second. Naming the time makes the size of the gap clear. Blames the phone, never the parent, and says outright that it will not restart on its own.

## 27 · Voice model failed to load

**Tree.**
```
Screen (centred)
  FeatureIcon (112dp, warning-tinted, hearing_disabled)
  Headline (serif)                     degraded.modelFailedTitle
  BodyLarge (centred)                  degraded.modelFailedBody
  ReassurePanel (surface, outline, shield, sage)   degraded.modelFailedSafe
  Spacer (flex)
  PrimaryButton                        onboarding.returnTryAgain
  QuietButton                          degraded.modelFailedRestart
```
Owns the fault in the first line. The app is fully disabled — **it must never pretend.** The reassurance about profiles matters because a broken app is exactly when a parent fears losing their setup work.

## 28 · No child enrolled yet

**Tree.**
```
Screen
  TopBar (mark only)
  PhotoSlotEmpty (120dp, dashed)
  Headline (serif, centred)            degraded.noChildTitle
  BodyLarge (centred)                  degraded.noChildBody
  Spacer (flex)
  PrimaryButton                        degraded.noChildCta   → 04
  Caption (centred)                    home.nothingToListenFor
```
The listening toggle is **absent**, not present-and-disabled: a dead toggle invites tapping and explains nothing. The reason sits under the CTA instead. Routes warmly into enrollment rather than reporting a blocked state.

## 29 · Listening blocked

**Tree.**
```
Screen (centred)
  FeatureIcon (112dp, neutral, phone_in_talk)
  Headline (serif)                     degraded.blockedTitle
  BodyLarge (centred)                  degraded.blockedBody | blockedOtherApp
  InfoPanel (surfaceMuted, autorenew)  degraded.blockedSelfHealing
```
The only degraded state with no action and no amber, because it genuinely resolves itself. Still stated plainly — the app never claims to be listening when it isn't.

## 31 · Empty history, first run

**Tree.**
```
Screen
  AppBar (back + history.title)
  Column (centred, offset up)
    SmallIcon (56dp circle, surfaceSunken, history)
    HeadlineSmall (serif, centred)     history.emptyTitle
    Body (centred)                     history.emptyBody
```
Two lines and a small glyph. An empty log on day one is the expected state, so it gets no illustration, no encouragement, and no call to action.

---

# Bidirectional layout rules

RTL is a layout constraint on every screen, not a translation task at the end. Retrofitting it is far more expensive than designing for it, and the intervention overlay breaks worst.

**Mirror the layout, not the content.**
- **Flip:** rows, gutters, list chevrons, back arrows, progress fill direction, slider direction, sheet corner asymmetry.
- **Never flip:** photos, faces, the app mark, waveforms, media controls. A mirrored waveform reads as a different sound.

**Every name is a bidi isolate.** Names, user-typed text and numbers are wrapped in `bdi` with the font stack resolved from the *string's own script*, not the UI language. This is what keeps `מאיה might be calling you` and `Maya אולי קוראת לך` from scrambling. Mixed-direction strings are the normal case here, not an edge case.

**Never concatenate.** One localised sentence per key with named placeholders, so the Hebrew writer controls word order. Never `{name}` + `" might be calling you"`.

**Neither language may set the width.** Hebrew runs shorter here. Buttons and cards size to content with a minimum; no layout depends on a line count; headlines use `text-wrap: pretty` and are checked at both lengths. Neither language may go slack when the other is shorter.

**Type.** Latin/Hebrew pairs are `Source Serif 4`/`Frank Ruhl Libre` and `IBM Plex Sans`/`IBM Plex Sans Hebrew`. Sizes are identical across scripts; the only per-script change is body line-height ×1.10 (1.55 → 1.70). Hebrew is never bumped up to compensate for having no case distinction.

**The overlay survives mirroring almost unchanged** — it is centre-aligned by deliberate choice, not luck. Its two buttons **stack** rather than sitting side by side, so their order does not flip and muscle memory holds across languages.

**Hebrew copy is written, not translated.** Every emotionally weighted string in `strings.json` carries an `intent` note so a native writer can reach for the feeling independently. The Hebrew currently in `strings.json` and in the mockups is a placeholder for that pass, not shipping copy.

**Screens delivered in both languages and directions:** 15 (overlay), 13 (home), 06 (live enrollment). All other screens are English-only in the mockups; the rules above govern their RTL builds.

---

# iOS divergence

Android first; Tier 1 designed as a complete product rather than a degraded one. But four screens need genuine redesign, not a mechanical port.

| Screen | Why it is real rework |
|---|---|
| **02 · Privacy** | Different guarantee, so different copy and structure. The Android screen-off promise cannot appear at all. Shipped as `02b` |
| **15 · Intervention** | No overlay exists. Tier 1 is a Live Activity in the Dynamic Island + a Time-Sensitive notification carrying the photo — a different surface with its own size and interaction budget, not a resized overlay |
| **18 · Response builder** | `Dim` and `Block` permanently unavailable, non-openable lock. `Soften` means ducking only, and the copy says "quieter, not silent". Shipped as `18b` |
| **20 · Family time** | Optional on Android, a hard **gate** on iOS. Different information architecture: the schedule becomes part of setup rather than a setting. Shipped as `20b` |

**No iOS equivalent at all:** 14 · the persistent notification. iOS needs no always-on notification, so its reassurance job moves to the Live Activity and the home screen.

**Mechanical ports** (everything else): HIG back affordance, home indicator, 44pt minimum targets, native switches, iOS haptics.

**If Tier 2 lands:** `Block` becomes real via app shielding, and the "only while I'm actually using my phone" rule that Android gets for free is restored — which means 20 relaxes from a gate to a setting. **Screen 20 must therefore be designed to demote gracefully**, not rebuilt.

---

# Name dependence

"Presence" is a working name and may be replaced before launch. Name-dependent surfaces, exhaustively:

1. The launcher label.
2. `about.appName` on screen 24.
3. The persistent notification's app-name row (14, 16).

Nothing else. No copy anywhere uses "presence" as a pun or a concept, no screen's meaning depends on it, and the mark is drawn rather than lettered — a soft ring with a halo, readable as an ear, a ripple, an aperture, or a face, and committed to none. A rename requires no token changes and no redrawing.
