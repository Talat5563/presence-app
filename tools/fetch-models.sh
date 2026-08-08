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
echo "Fetching Silero VAD..."
curl -fsSL "$VAD_URL" -o "$DEST/silero_vad.onnx"
echo "Fetching speaker embedding model..."
curl -fsSL "$EMBEDDING_URL" -o "$DEST/speaker_embedding.onnx"

# Which model produced which numbers is part of every Phase 0 result.
cat > "$DEST/PROVENANCE.txt" <<PROVENANCE
vad: $VAD_URL
embedding: $EMBEDDING_URL
PROVENANCE

ls -lh "$DEST"
echo
echo "Recorded provenance in $DEST/PROVENANCE.txt — quote it in any results you report."
