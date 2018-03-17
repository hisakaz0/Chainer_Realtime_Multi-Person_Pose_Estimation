#!/usr/bin/env bash
mp4_file="$1"
png_dir="$2"
fps='30'
ffmpeg -i $mp4_file -f image2 -vcodec png -r $fps $png_dir/%05d.png
