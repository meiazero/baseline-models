#!/bin/bash
# Pack best model checkpoints into a single archive with the exact layout
# expected by entropy-guided-router (egr/paths.py:CHECKPOINT_FILES).
#
# Usage (on the Apollo cluster):
#   bash scripts/compress_checkpoints.sh
#   RESULTS_DIR=/custom/results bash scripts/compress_checkpoints.sh
#
# Output:
#   $ARCHIVE_DIR/baseline_checkpoints_<UTCSTAMP>.tar.gz
#   $ARCHIVE_DIR/baseline_checkpoints_<UTCSTAMP>.sha256
#
# To consume locally:
#   tar -xzvf baseline_checkpoints_<UTCSTAMP>.tar.gz -C "$EGR_CHECKPOINTS_DIR"
#
# Env vars (all optional):
#   REPO_DIR     baseline-models repo root      (default: parent of this script)
#   RESULTS_DIR  where trained checkpoints live (default: $REPO_DIR/results)
#   ARCHIVE_DIR  where archive is written       (default: $REPO_DIR/archives)

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/results}"
ARCHIVE_DIR="${ARCHIVE_DIR:-$REPO_DIR/archives}"

# Relative paths inside RESULTS_DIR. Must match egr/paths.py:CHECKPOINT_FILES
# in the entropy-guided-router repo.
CHECKPOINTS=(
    "uncrtaints/uncrtaints_brazil/best.pth.tar"
    "ctgan/ctgan_brazil/G_best.pth"
    "utilise/utilise_brazil/best.pth.tar"
)

mkdir -p "$ARCHIVE_DIR"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ARCHIVE="$ARCHIVE_DIR/baseline_checkpoints_${STAMP}.tar.gz"
MANIFEST="$ARCHIVE_DIR/baseline_checkpoints_${STAMP}.sha256"

PRESENT=()
MISSING=()
for rel in "${CHECKPOINTS[@]}"; do
    if [[ -f "$RESULTS_DIR/$rel" ]]; then
        PRESENT+=("$rel")
    else
        MISSING+=("$rel")
    fi
done

if (( ${#PRESENT[@]} == 0 )); then
    echo "ERROR: no checkpoints found under $RESULTS_DIR" >&2
    printf '  missing: %s\n' "${MISSING[@]}" >&2
    exit 1
fi

echo "=== Packing ${#PRESENT[@]} checkpoint(s) from $RESULTS_DIR ==="
for rel in "${PRESENT[@]}"; do
    size=$(du -h "$RESULTS_DIR/$rel" | cut -f1)
    printf '  + %-50s %s\n' "$rel" "$size"
done
if (( ${#MISSING[@]} > 0 )); then
    echo "--- Skipped (missing): ---"
    printf '  - %s\n' "${MISSING[@]}"
fi

tar -czf "$ARCHIVE" -C "$RESULTS_DIR" "${PRESENT[@]}"

# Per-file sha256 inside the archive + sha256 of the archive itself.
{
    echo "# baseline-models checkpoint archive"
    echo "# generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# host:      $(hostname)"
    echo "# repo:      $REPO_DIR"
    echo "# results:   $RESULTS_DIR"
    echo "# archive:   $(basename "$ARCHIVE")"
    echo
    echo "# sha256 of source files:"
    (cd "$RESULTS_DIR" && sha256sum "${PRESENT[@]}")
    echo
    echo "# sha256 of archive:"
    (cd "$ARCHIVE_DIR" && sha256sum "$(basename "$ARCHIVE")")
} > "$MANIFEST"

echo
echo "=== Done ==="
echo "  archive : $ARCHIVE   ($(du -h "$ARCHIVE" | cut -f1))"
echo "  manifest: $MANIFEST"
echo
echo "To consume on local:"
echo "  scp <user>@<cluster>:$ARCHIVE ."
echo "  scp <user>@<cluster>:$MANIFEST ."
echo "  sha256sum -c $(basename "$MANIFEST")  # verify"
echo "  tar -xzvf $(basename "$ARCHIVE") -C \"\$EGR_CHECKPOINTS_DIR\""

if (( ${#MISSING[@]} > 0 )); then
    exit 2  # partial success
fi
