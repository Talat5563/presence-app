#!/usr/bin/env bash
# Downloads the models Phase 0 needs into models/. Binaries are gitignored (*.onnx).
#
# Usage: ./tools/fetch-models.sh [embedding-model-url]
#   Defaults to the model Phase 0 was designed around. Run ./tools/list-models.sh for others.
set -euo pipefail

DEFAULT_EMBEDDING="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx"
EMBEDDING_URL="${1:-$DEFAULT_EMBEDDING}"
VAD_URL="https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx"
DEST="models"

mkdir -p "$DEST"

download() {
  local url="$1" dest="$2" label="$3"
  local tmp="${dest}.partial"

  echo "Fetching ${label}..."
  curl -fsSL "$url" -o "$tmp"

  # An ONNX file is protobuf; its opening bytes are never an HTML document. This catches a 200
  # response that is a landing page rather than the model — the one failure mode curl -f misses.
  if head -c 512 "$tmp" | grep -qiE '<!doctype html|<html'; then
    rm -f "$tmp"
    echo "error: ${url} returned an HTML page, not a model" >&2
    exit 1
  fi

  local size
  size=$(wc -c < "$tmp")
  if [ "$size" -lt 1000000 ]; then
    rm -f "$tmp"
    echo "error: ${url} returned only ${size} bytes; expected a multi-megabyte model" >&2
    exit 1
  fi

  # Only replace the existing file once the download is known good, so a failure mid-transfer
  # cannot destroy a working model.
  mv "$tmp" "$dest"
}

download "$VAD_URL" "$DEST/silero_vad.onnx" "Silero VAD"
download "$EMBEDDING_URL" "$DEST/speaker_embedding.onnx" "speaker embedding model"

# Which model produced which numbers is part of every Phase 0 result.
cat > "$DEST/PROVENANCE.txt" <<PROVENANCE
vad: $VAD_URL
embedding: $EMBEDDING_URL
PROVENANCE

ls -lh "$DEST"
echo
echo "Recorded provenance in $DEST/PROVENANCE.txt — quote it in any results you report."
