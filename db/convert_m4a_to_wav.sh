#!/bin/bash

audio_count=`find . -name \*m4a | wc -l`
echo ${audio_count}
for audio in *.m4a; do
    ffmpeg -hide_banner -loglevel warning -n -i "$audio" "wav/${audio%.*}.wav"
    echo -n "."
done | pv -pte -i0.1 -s${audio_count} > /dev/null
