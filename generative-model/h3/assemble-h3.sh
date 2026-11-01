#!/bin/bash
# Concatenate chained H3 segments and put the original soundtrack back.
# usage: bash assemble-h3.sh <chain-out-dir> <original.mp4> <out.mp4> [WxH] [seconds]
# [seconds] retimes to a shorter length when only the first part of the source was generated.
# The segments are 243 frames each (10.125 s) for every 10 s of source, so the
# concatenation is retimed to the original duration before the audio is muxed.
set -euo pipefail
dir=$(cd "$1" && pwd); origin=$2; out=$3; size=${4:-}; limit=${5:-}
mapfile -t segs < <(find "$dir" -maxdepth 1 -name 'seg[0-9][0-9].mp4' | sort)
[ ${#segs[@]} -gt 0 ] || { echo "no segments in $dir" >&2; exit 1; }
printf "file '%s'\n" "${segs[@]}" >"$dir/concat.txt"
gen=$dir/_generated.mp4
ffmpeg -nostdin -y -v error -f concat -safe 0 -i "$dir/concat.txt" -c copy "$gen"
gen_dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$gen")
org_dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$origin")
if [ -n "$limit" ]; then org_dur=$limit; fi
ratio=$(/root/h3-venv/bin/python -c "print(f'{$org_dur/$gen_dur:.6f}')")
scale=""
[ -n "$size" ] && scale=",scale=${size}:flags=lanczos,setsar=1"
ffmpeg -nostdin -y -v error -i "$gen" -i "$origin" \
  -vf "setpts=PTS*${ratio}${scale},fps=24" \
  -map 0:v:0 -map 1:a:0 -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p \
  -c:a aac -b:a 128k -t "$org_dur" -shortest -movflags +faststart "$out"
rm -f "$gen"
echo "segments=${#segs[@]} duration=$org_dur -> $out"
