#!/usr/bin/bash
image_dir="$1"
image_list_file="$2"
ls -1 $image_dir/*.png > $image_list_file
