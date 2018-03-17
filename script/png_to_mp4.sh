#!/usr/bin/env bash
png_dir="$1"
mp4_file="$2"
fps='30'
ffmpeg -r $fps -i $png_dir/%05d.png -vcodec libx264 -pix_fmt yuv420p $mp4_file
