#!/bin/bash
# this script downloads VIIRS files from the NASA LAADS DAAC to any system.
# Author: Javier Villegas Bravo
# Modified: 02/20/2025


echo "building environment"
# token from LAADS DAAC (60 day experation)
token="eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6InZsbGdzYnIyIiwiZXhwIjoxNzQxNzI4MDMxLCJpYXQiOjE3MzY1NDQwMzEsImlzcyI6Imh0dHBzOi8vdXJzLmVhcnRoZGF0YS5uYXNhLmdvdiIsImlkZW50aXR5X3Byb3ZpZGVyIjoiZWRsX29wcyIsImFjciI6ImVkbCIsImFzc3VyYW5jZV9sZXZlbCI6M30.1nL7yQ-BPKLQthSFhqru2VD6Tvoq4dSUyErwMkoXGJ3FCNb49kTtK-hfj5ThG_SV5MTqKuvdtW-kkmyjbpFRmcDKlB2sQApOPOLmSingdd51gXTltZbzXmmV3Ta04tffFHLzDZgQFQ8Jelg41ktRpYJcAlMr5pso9kNEKAS5C00_oIC-diJAPl-Xh9A9N7pMuguD5kNGbfI8BuQJevXrnxYHt7jwWXNUtCu7szeNhEc_tnySb5EdPAIXbZkj33HXCUk1p3NVnXWH8bTNG0uoOxQPLoODVg1A41eS9Ng4iNTeTfx6L_NH-G4zwX8z2R4IVLkhIzea_eXG56XwT4RS5g"
# directory to download data to (on ZARATAN)
target_directory_VJ109="/scratch/zt1/project/vllgsbr2-prj/raw_data_burnscar/data/noaa_20_viirs/operational_data_feed/VJ109"
target_directory_VJ103="/scratch/zt1/project/vllgsbr2-prj/raw_data_burnscar/data/noaa_20_viirs/operational_data_feed/VJ103"
# home="/Users/vllgsbr2/Downloads"
# target_directory_VJ109=$home
# target_directory_VJ103=$home

# grab current year and day to download latest files
current_date=$(date +%Y%j)
# string slicing in bash > ${string:position:length}
year=${current_date:0:4}
# bash time on ZARATAN is ET but VIIRS files are in UTC
# only need to make sure download time is 8:30pm ET in
# crontab -e and day light savings 7:30pm ET
day_of_year=${current_date:4:3}
URL_source="https://nrt3.modaps.eosdis.nasa.gov/archive/allData"

echo "current datetime is: $current_date, $year, $day_of_year"

#list of valid hours to download
declare -a valid_hours_list=("19" "20" "21" "22")

#list of valid mins to download
valid_min_list=()
max_digits=2 # Desired number of digits, including leading zeros
for i in {0..54..6}; do
  padded_num=$(printf "%0${max_digits}d" $i)
  valid_min_list+=("$padded_num")
done

# download each valid hour min combo
for hour in "${valid_hours_list[@]}"
do
  for min in "${valid_min_list[@]}"
  do
    VJ109_file_str="VJ109_NRT.A$year$day_of_year.$hour$min.002"
    echo $VJ109_file_str
  done
done
