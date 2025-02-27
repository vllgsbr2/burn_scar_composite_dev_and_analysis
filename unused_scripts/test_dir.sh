year="2025"
day_of_year="057"

target_directory_VJ109="/scratch/zt1/project/vllgsbr2-prj/raw_data_burnscar/data/noaa_20_viirs/operational_data_feed/VJ109"
declare -a invalid_hours_list=("00" "01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12" "13" "14" "15" "16" "17" "18" "23")
for hour in "${invalid_hours_list[@]}"
do
  echo $hour
  for f in $(find "$target_directory_VJ109/VJ109_NRT/$year/$day_of_year/" -name "VJ109_NRT.A$year$day_of_year.$hour*"); do rm $f; done
done


# delete extra directories from download after moving files up 3 dir
mv $target_directory_VJ109/VJ109_NRT/$year/$day_of_year/* $target_directory_VJ109
rmdir $target_directory_VJ109/VJ109_NRT/$year/$day_of_year
rmdir $target_directory_VJ109/VJ109_NRT/$year
rmdir $target_directory_VJ109/VJ109_NRT
