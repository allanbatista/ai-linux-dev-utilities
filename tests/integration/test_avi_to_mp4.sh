#!/usr/bin/env bash

set -euo pipefail

script="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/avi-to-mp4"
test_dir="$(mktemp -d)"
trap 'rm -rf "$test_dir"' EXIT

mkdir -p "$test_dir/single output" "$test_dir/directory" "$test_dir/glob source"
ffmpeg -nostdin -hide_banner -loglevel error \
    -f lavfi -i "testsrc2=size=32x32:rate=2:duration=1" \
    -f lavfi -i "sine=frequency=1000:duration=1" \
    -c:v mpeg4 -c:a libmp3lame -shortest "$test_dir/source.avi"
cp "$test_dir/source.avi" "$test_dir/directory/one.avi"
cp "$test_dir/source.avi" "$test_dir/directory/two.AVI"
cp "$test_dir/source.avi" "$test_dir/glob source/three.avi"

"$script" "$test_dir/source.avi" -o "$test_dir/single output"
"$script" "$test_dir/directory"
"$script" "$test_dir/glob source/*.avi" -o "$test_dir/glob output"

test -f "$test_dir/single output/source.mp4"
test -f "$test_dir/directory/one.mp4"
test -f "$test_dir/directory/two.mp4"
test -f "$test_dir/glob output/three.mp4"

source_hash="$(ffmpeg -hide_banner -loglevel error -i "$test_dir/source.avi" -map 0:v:0 -c copy -f hash -hash sha256 -)"
output_hash="$(ffmpeg -hide_banner -loglevel error -i "$test_dir/single output/source.mp4" -map 0:v:0 -c copy -f hash -hash sha256 -)"
test "$source_hash" = "$output_hash"
source_hash="$(ffmpeg -hide_banner -loglevel error -i "$test_dir/source.avi" -map 0:a:0 -c copy -f hash -hash sha256 -)"
output_hash="$(ffmpeg -hide_banner -loglevel error -i "$test_dir/single output/source.mp4" -map 0:a:0 -c copy -f hash -hash sha256 -)"
test "$source_hash" = "$output_hash"

if "$script" "$test_dir/source.avi" -o "$test_dir/single output"; then
    echo "Falha: saída existente deveria ser recusada." >&2
    exit 1
fi
