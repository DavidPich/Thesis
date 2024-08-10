#!/bin/bash

audio_count=$(find data -name "data/*.m4a" | wc -l)

mkdir -p data/wav

echo ${audio_count}


for audio in *.m4a; do
    ffmpeg -hide_banner -loglevel warning -n -i "$audio" "data/wav/${audio%.*}.wav"
    echo -n "."
done | pv -pte -i0.1 -s${audio_count} > /dev/null
