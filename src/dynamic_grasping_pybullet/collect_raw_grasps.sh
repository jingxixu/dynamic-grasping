#!/usr/bin/env bash
# Usage: ./collect_raw_grasps.sh 3  would run in batch of 3s

object_names=(bleach_cleanser mustard_bottle potted_meat_can sugar_box tomato_soup_can cube power_drill)

ncpu=${1:-1}    # optional argument ncpu defaults to 1
num_objects=${#object_names[@]}
num_batches=$(((num_objects + ncpu - 1) / ncpu))

node_id=10000

for ((i = 0; i < num_batches; i++)); do
  echo "batch $i"
  for ((j = 0; j < ncpu; j++)); do

    idx=$((i * ncpu + j))
    object_name=${object_names[$idx]}
    echo $idx $object_name

    gnome-terminal -e "bash -ci '\
            source ../../devel/setup.bash && \
            export ROS_MASTER_URI=http://localhost:$(($node_id + $j)) && \
            roslaunch grid_sample_plugin grid_sample_plugin.launch;'"
    sleep 5

    source ../../devel/setup.bash &&
      export ROS_MASTER_URI=http://localhost:$(($node_id + $j)) &&
      python collect_raw_grasps.py --object_name $object_name --robot_name robotiq_85_gripper --num_grasps 5000 --max_steps 40000 --grasp_folder_path raw_grasps &&
      sleep 5 &

  done
  echo "waiting"
  wait
done

# for object_name in bleach_cleanser mustard_bottle potted_meat_can sugar_box tomato_soup_can cube power_drill
# do
#     python collect_raw_grasps.py --object_name $object_name --robot_name MicoGripper --num_grasps 5000 --max_steps 40000 --grasp_folder_path raw_grasps
#     sleep 5
# done