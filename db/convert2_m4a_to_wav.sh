#!/bin/bash

audio_count=`find . -name \*m4a | wc -l`
echo ${audio_count}
for audio in *.m4a; do
#    ffmpeg -n -i "$audio" "wav/${audio%.*}.wav"
    sleep 0.2
    echo -n "."
done 
#| pv -pte -i0.1 -s${audio_count} > /dev/null
