#!/usr/bin/env bash
# Lists the speaker-recognition models sherpa-onnx publishes, plus the Silero VAD asset.
# Note the upstream release tag really is spelled "recongition".
#
# You do not need this to run Phase 0 — fetch-models.sh already defaults to the chosen model.
# It exists for swapping models later, e.g. after L3 fine-tuning.
set -euo pipefail

echo "== sherpa-onnx speaker embedding models =="
curl -fsSL \
  "https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/tags/speaker-recongition-models" \
  | grep '"browser_download_url"' \
  | sed -E 's/.*"([^"]+)".*/\1/' \
  | sort

echo
echo "== Silero VAD =="
echo "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx"
echo
echo "NOTE: every one of these consumes filterbank features, not raw audio."
echo "Check any replacement's signature before trusting the existing adapters,"
echo "and check its licence before shipping (spec §10, open question 2)."
