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
day_of_year="057" #${current_date:4:3}
URL_source="https://nrt3.modaps.eosdis.nasa.gov/archive/allData"

echo "current datetime is: $current_date, $year, $day_of_year"
:'
path_to_viirs_VJ109="$URL_source/5200/VJ109_NRT/$year/$day_of_year/"
wget -e robots=off -m -np -R .html,.tmp -nH --cut-dirs=3 $path_to_viirs_VJ109 \
--header "Authorization: Bearer $token" -P $target_directory_VJ109
'
echo "done downloading VJ109"

# now delete all file not within the aquisition hours for day time western US
# These are hours outside of 19z-22z for both NOAA-20 and 21
declare -a invalid_hours_list=("00" "01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12" "13" "14" "15" "16" "17" "18" "23")
:'
for hour in "${invalid_hours_list[@]}"
do
  for f in $(find "$target_directory_VJ109/VJ109_NRT/$year/$day_of_year/" -name "VJ109_NRT.A$year$day_of_year.$hour*"); do rm $f; done
done

# delete extra directories from download after moving files up 3 dir
mv $target_directory_VJ109/VJ109_NRT/$year/$day_of_year/* $target_directory_VJ109
rmdir $target_directory_VJ109/VJ109_NRT/$year/$day_of_year
rmdir $target_directory_VJ109/VJ109_NRT/$year
rmdir $target_directory_VJ109/VJ109_NRT

path_to_viirs_VJ103="$URL_source/5201/VJ103MOD_NRT/$year/$day_of_year/$file_name_VJ103"
wget -e robots=off -m -np -R .html,.tmp -nH --cut-dirs=3 $path_to_viirs_VJ103 \
--header "Authorization: Bearer $token" -P $target_directory_VJ103
'
echo "done downloading VJ103"

# now delete all file not within the aquisition hours for day time western US
# These are hours outside of 19z-22z for both NOAA-20 and 21
for hour in "${invalid_hours_list[@]}"
do
  for f in $(find "$target_directory_VJ103/VJ103MOD_NRT/$year/$day_of_year/" -name "VJ103MOD_NRT.A$year$day_of_year.$hour*"); do rm $f; done
done

# delete extra directories from download after moving files up 3 dir
mv $target_directory_VJ103/VJ103MOD_NRT/$year/$day_of_year/* $target_directory_VJ103
rmdir $target_directory_VJ103/VJ103MOD_NRT/$year/$day_of_year
rmdir $target_directory_VJ103/VJ103MOD_NRT/$year
rmdir $target_directory_VJ103/VJ103MOD_NRT

# nasty pile
#19-22z should cover the NOAA-20 & 21 orbit over the Western U.S.
#loop through the hours and download all VJ109 and 103 files
# declare -a valid_hours_list=("19" "20" "21" "22")
# for hour in "${valid_hours_list[@]}"
# do
#   echo $hour
#   # wget -r -l1 --no-parent -A ".deb" http://www.shinken-monitoring.org/pub/debian/
#   #
#   # -r recursively
#   # -l1 to a maximum depth of 1
#   # --no-parent ignore links to a higher directory
#   # -A "*.deb" your pattern
#
#   file_name_VJ109="VJ109.A$year$day_of_year.$hour*"
#   file_name_VJ103="VJ103MOD.A$year$day_of_year.$hour*"
#
#   echo "downloading $file_name_VJ109"
#   path_to_viirs_VJ109="https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5200/VJ109/$year/$day_of_year/"
#   wget -r -l1 --no-parent -A $file_name_VJ109 -e robots=off -m -np -R .html,.tmp -nH --cut-dirs=3 $path_to_viirs_VJ109 \
#   --header "Authorization: Bearer $token" -P $target_directory_VJ109
#
#   echo "downloading $file_name_VJ103"
#   path_to_viirs_VJ103="https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5201/VJ103/$year/$day_of_year/$file_name_VJ103"
#   wget -r -l1 --no-parent -A $file_name_VJ103 -e robots=off -m -np -R .html,.tmp -nH --cut-dirs=3 $path_to_viirs_VJ103 \
#   --header "Authorization: Bearer $token" -P $target_directory_VJ103
#
#   echo "both files downloaded successfully for hour $hour"
# done
#

