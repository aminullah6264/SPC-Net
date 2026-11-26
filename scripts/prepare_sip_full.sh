#!/bin/bash
# Build the full SIP evaluation set (all 929 images) from the Train/Test split folders.
# Our models never train on SIP, so all images are unseen; this matches the RGB-D papers.
#
# Usage: bash scripts/prepare_sip_full.sh [data_root]   (default: $SOD_DATA_ROOT or ./data)
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="${1:-${SOD_DATA_ROOT:-$(pwd)/data}}"
SRC="$ROOT/SIP_Dataset"; DST="$ROOT/SIP-full"
[ -d "$SRC/Train/Images" ] && [ -d "$SRC/Test/Images" ] || { echo "Expected $SRC/{Train,Test}/{Images,Masks}"; exit 1; }

mkdir -p "$DST/Images" "$DST/Masks"
for split in Train Test; do
  for f in "$SRC/$split/Images"/*.jpg; do ln -sf "$(realpath "$f")" "$DST/Images/"; done
  for f in "$SRC/$split/Masks"/*.png;  do ln -sf "$(realpath "$f")" "$DST/Masks/";  done
done
find "$DST/Masks" -name '*(1)*' -delete   # duplicate masks shipped in some copies
echo "SIP-full: $(ls "$DST/Images" | wc -l) images, $(ls "$DST/Masks" | wc -l) masks (expected 929)"
