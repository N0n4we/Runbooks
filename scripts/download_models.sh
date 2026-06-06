#!/usr/bin/env bash
# LoraCLI base model downloader.
#
# Downloads Anima base models to MODEL_DIR if they are not already present.
# Fails loudly when a download URL has not been configured, so missing models
# never silently fall through to a confusing failure inside Python.
#
# Usage: bash scripts/download_models.sh [MODEL_DIR]
#   MODEL_DIR defaults to ./models/base
#
# URL configuration (in priority order):
#   1. environment variables ANIMA_BASE_URL / QWEN3_URL / VAE_URL
#   2. the literal string assigned below (replace REPLACE_ME with real URLs
#      once you have stable hosting)

set -euo pipefail

MODEL_DIR="${1:-./models/base}"
mkdir -p "$MODEL_DIR"

ANIMA_BASE_URL="${ANIMA_BASE_URL:-REPLACE_ME}"
QWEN3_URL="${QWEN3_URL:-REPLACE_ME}"
VAE_URL="${VAE_URL:-REPLACE_ME}"

download() {
    local url="$1"
    local dest="$2"
    local name
    name="$(basename "$dest")"

    if [ -f "$dest" ]; then
        echo "[skip] $name already exists"
        return 0
    fi

    if [ "$url" = "REPLACE_ME" ] || [ -z "$url" ]; then
        echo "ERROR: download URL not configured for $name." >&2
        echo "       Edit scripts/download_models.sh (replace REPLACE_ME) or export the matching env var." >&2
        return 1
    fi

    echo "[download] $name <- $url"
    # -L: follow redirects
    # --fail: non-zero exit on HTTP errors
    # -C -: resume partial downloads
    # download to .partial then rename, so a half-finished file is never
    # confused with a complete one on retry
    curl -L --fail -C - -o "$dest.partial" "$url"
    mv "$dest.partial" "$dest"
}

download "$ANIMA_BASE_URL" "$MODEL_DIR/anima_baseV10.safetensors"
download "$QWEN3_URL"      "$MODEL_DIR/qwen_3_06b_base.safetensors"
download "$VAE_URL"        "$MODEL_DIR/qwen_image_vae.safetensors"

echo "All models ready in $MODEL_DIR"
