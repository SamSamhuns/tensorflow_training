#!/bin/bash
find "$1" -maxdepth 1 -mindepth 1 -type d | while read dir; do
  printf "%-55.55s : " "$dir"
  find "$dir" -type f | wc -l
done
