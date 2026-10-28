#!/usr/bin/env bash
# Downloads CHILDES session media listed in a needed-media file, using your own TalkBank login.
#
# TalkBank media sits behind a free account and is offered per file, not as an archive. This
# script never sees your password: you sign in at talkbank.org in your browser, copy that
# session's Cookie header, and pass it in an environment variable that only your shell holds.
#
#   1. Sign in at https://talkbank.org in your browser.
#   2. Open https://media.talkbank.org/childes/Eng-NA/MacWhinney/ in the same browser, open
#      DevTools > Network, reload, click the page request, and copy the value of its "cookie"
#      request header.
#   3. Run, in your own terminal:
#        export TALKBANK_COOKIE='<the copied value>'
#        ./tools/fetch-childes-media.sh data/childes/needed-media.txt data/childes/media
#
# Each line of the list is a path like "Gleason/Dinner/andy" (corpus/folders/media-name, no
# extension), as written beside the transcripts. Files already downloaded are skipped, so the
# script can be re-run after an interruption. Media is CC BY-NC-SA: never commit or share it.
set -euo pipefail

list="${1:?usage: $0 NEEDED_MEDIA_LIST OUT_DIR}"
out="${2:?usage: $0 NEEDED_MEDIA_LIST OUT_DIR}"
# No session recording is smaller than this; anything smaller is a truncated or error response.
MIN_KB=64

: "${TALKBANK_COOKIE:?set TALKBANK_COOKIE to your signed-in talkbank.org Cookie header (see the comment at the top)}"

# Where each corpus lives on the media server.
corpus_base() {
  case "$1" in
    MacWhinney) echo "childes/Eng-NA/MacWhinney" ;;
    Gleason) echo "childes/Eng-NA/Gleason" ;;
    BermanLong) echo "childes/Other/Hebrew/BermanLong" ;;
    *) return 1 ;;
  esac
}

fetched=0 skipped=0 failed=0
while IFS= read -r entry || [[ -n "$entry" ]]; do
  entry="${entry%$'\r'}"
  [[ -z "$entry" ]] && continue
  corpus="${entry%%/*}"
  rest="${entry#*/}"
  if ! base="$(corpus_base "$corpus")"; then
    echo "unknown corpus '$corpus' in: $entry" >&2
    failed=$((failed + 1))
    continue
  fi

  # Only a file of plausible size counts as already downloaded; a truncated fragment is retried.
  if [[ -n "$(find "$out/$(dirname "$entry")" -maxdepth 1 -name "$(basename "$entry").*" ! -name '*.part' -size +"${MIN_KB}k" 2>/dev/null)" ]]; then
    skipped=$((skipped + 1))
    continue
  fi
  mkdir -p "$out/$(dirname "$entry")"

  got=""
  for ext in mp3 wav mp4 m4a; do
    tmp="$out/$entry.$ext.part"
    # The media server answers a request without a Range header with only the first 11 bytes (a
    # 206 for "bytes 0-10"), so ask for the whole file as a range, the way a browser's player does.
    type="$(curl -sS -L --fail --max-time 1800 -H "Cookie: $TALKBANK_COOKIE" -H "Range: bytes=0-" \
      -o "$tmp" -w '%{content_type}' "https://media.talkbank.org/$base/$rest.$ext" 2>/dev/null)" || {
      rm -f "$tmp"
      continue
    }
    # A login page comes back as HTML with status 200; treat it as "not signed in", not as media.
    if [[ "$type" == text/html* ]]; then
      rm -f "$tmp"
      echo "got a login page for $entry — is TALKBANK_COOKIE current?" >&2
      exit 1
    fi
    # A few bytes of header and nothing else means the server cut the transfer short. Stop rather
    # than leave hundreds of fragments that look like finished downloads.
    size="$(wc -c < "$tmp")"
    if (( size < MIN_KB * 1024 )); then
      rm -f "$tmp"
      echo "only $size bytes arrived for $entry.$ext — the server cut the download short; stopping" >&2
      exit 1
    fi
    mv "$tmp" "$out/$entry.$ext"
    got="$ext"
    break
  done

  if [[ -n "$got" ]]; then
    fetched=$((fetched + 1))
    echo "[$((fetched + skipped + failed))] $entry.$got"
  else
    failed=$((failed + 1))
    echo "no media found for $entry" >&2
  fi
done < "$list"

echo "fetched $fetched, already present $skipped, missing $failed"
