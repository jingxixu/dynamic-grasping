#!/usr/bin/env bash

for object_name in bleach_cleanser mustard_bottle potted_meat_can sugar_box tomato_soup_can cube power_drill
do
  echo ${object_name};
  screen -dmS ${object_name} bash -c "source ../../devel/setup.bash;
    python collect_motion_aware_dataset.py \
      --object_name $object_name \
      --robot_config_name ur5_robotiq \
      --grasp_database_path assets/grasps/filtered_grasps_noise_robotiq_100_1.00 \
      --save_folder_path motion_aware_dataset \
      --num_trials_per_grasp 1000 \
      --use_simple \
      --disable_gui;
    $SHELL"
  sleep 5;
done