#!/usr/bin/env bash
# Build contact sheets (grids of sampled frames) for manual review.
#
# Deliberately dumb and independent of the clip detector: fixed time step, no
# scene detection, no scoring. Used to label reference intervals by eye before
# the detector runs on the same material.
#
# Usage:
#   tools/contact_sheet.sh SRC START DURATION OUTDIR [STEP=4] [CELLS=4x4]
#
# Prints one line per sheet: "<absolute start s> <sheet path>".
set -euo pipefail

src=${1:?source video}
start=${2:?start seconds}
duration=${3:?duration seconds}
outdir=${4:?output directory}
step=${5:-4}
cells=${6:-4x4}

cols=${cells%x*}
rows=${cells#*x}
frame_w=${FRAME_W:-480}

die() { echo "contact_sheet: $*" >&2; exit 2; }

command -v ffmpeg >/dev/null || die "ffmpeg missing"
command -v ffprobe >/dev/null || die "ffprobe missing"
[ -r "$src" ] || die "cannot read $src"
case "$cols$rows" in *[!0-9]*) die "cells must look like 4x4" ;; esac
[ "$cols" -gt 0 ] && [ "$rows" -gt 0 ] || die "cells must be positive"
awk -v a="$start" 'BEGIN{exit !(a>=0)}' || die "start must be >= 0"
awk -v a="$duration" 'BEGIN{exit !(a>0)}' || die "duration must be > 0"
awk -v a="$step" 'BEGIN{exit !(a>0)}' || die "step must be > 0"

total_duration=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$src")
awk -v a="$start" -v d="$duration" -v t="$total_duration" \
  'BEGIN{exit !(a < t)}' || die "start $start is past the end of a ${total_duration}s source"

frame_h=$(( frame_w * 9 / 16 ))
per_sheet=$(( cols * rows ))
span=$(awk -v s="$step" -v n="$per_sheet" 'BEGIN{print s*n}')

mkdir -p "$outdir"

# y offset for the timestamp so it never sits on top of a subtitled lower third
label_h=$(( frame_h / 9 ))

n=0
offset=0
remaining=$duration
while awk -v r="$remaining" 'BEGIN{exit !(r>0)}'; do
  chunk=$(awk -v r="$remaining" -v s="$span" 'BEGIN{print (r<s? r : s)}')
  abs=$(awk -v a="$start" -v o="$offset" 'BEGIN{printf "%.3f", a+o}')
  n=$(( n + 1 ))
  sheet=$(printf "%s/sheet_%02d_%06.1fs.png" "$outdir" "$n" "$(awk -v a="$abs" 'BEGIN{print a}')")
  ffmpeg -nostdin -y -v error -ss "$abs" -t "$chunk" -i "$src" \
    -vf "fps=1/${step},scale=${frame_w}:${frame_h}:flags=lanczos,\
drawtext=text='%{pts\\:hms\\:${abs}}':x=6:y=h-${label_h}:fontsize=$(( label_h * 3 / 4 )):\
fontcolor=white:box=1:boxcolor=black@0.6,\
tile=${cols}x${rows}:padding=4:margin=4:color=black" \
    -frames:v 1 "$sheet" || die "ffmpeg failed for sheet $n ($abs)"
  echo "$abs $sheet"
  offset=$(awk -v o="$offset" -v c="$chunk" 'BEGIN{print o+c}')
  remaining=$(awk -v r="$remaining" -v c="$chunk" 'BEGIN{print r-c}')
done
