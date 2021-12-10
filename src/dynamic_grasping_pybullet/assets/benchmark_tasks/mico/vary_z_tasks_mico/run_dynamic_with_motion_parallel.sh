#!/usr/bin/env bash
exp_name="vary_z_mico"
timestr=${exp_name}_mico_$(hostname)_$(date '+%Y-%m-%d_%H-%M-%S')
echo $timestr
node_id=11000

mkdir $timestr
cp run_dynamic_with_motion_parallel.sh $timestr

for object_name in bleach_cleanser mustard_bottle potted_meat_can sugar_box tomato_soup_can cube power_drill
do

    gnome-terminal -e "bash -ci '\
        source ../../devel/setup.bash && \
        export ROS_MASTER_URI=http://localhost:$node_id && \
        roslaunch launch/mico_moveit_ros.launch planner:=chomp; $SHELL'"
    sleep 3

    gnome-terminal -e "bash -ci '\
        source ../../devel/setup.bash && \
        export ROS_MASTER_URI=http://localhost:$node_id && \
        python run_dynamic_with_motion.py \
            --object_name $object_name \
            --robot_config_name mico \
            --motion_mode dynamic_linear \
            --rendering \
            --num_trials 100 \
            --result_dir $timestr \
            --grasp_database_path assets/grasps/filtered_grasps_noise_100 \
            --grasp_threshold 0.03 \
            --lazy_threshold  0.3 \
            --conveyor_speed 0.03 \
            --conveyor_z_low 0.01 \
            --conveyor_z_high 0.3 \
            --close_delay 0.5 \
            --back_off 0.05 \
            --distance_low 0.15 \
            --distance_high 0.4 \
            --pose_freq 5 \
            --max_check 3 \
            --use_box \
            --use_kf \
            --fix_motion_planning_time 0.14 \
            --fix_grasp_ranking_time 0.135 \
            --always_try_switching \
            --use_joint_space_dist \
            --approach_prediction; $SHELL'"
    sleep 3
    ((node_id++))
done
