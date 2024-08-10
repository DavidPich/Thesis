#!/bin/bash

audio_count=`find . -name "data/*.m4a" | wc -l`
echo ${audio_count}
for audio in data/*.m4a; do
    ffmpeg -n -i "$audio" "wav/${audio%.*}.wav"
    sleep 0.2
    echo -n "."
done | pv -pte -i0.1 -s${audio_count} > /dev/null
