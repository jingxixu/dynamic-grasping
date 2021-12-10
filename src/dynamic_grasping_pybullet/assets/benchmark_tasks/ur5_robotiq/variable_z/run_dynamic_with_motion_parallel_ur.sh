#!/usr/bin/env bash

exp_name="fast_ours_PRM_No_Lazy_No_seed_run_1"
timestr=${exp_name}_ur5_$(hostname)_$(date '+%Y-%m-%d_%H-%M-%S')
echo $timestr
node_id=30005

mkdir $timestr
cp run_dynamic_with_motion_parallel_ur.sh $timestr

for object_name in bleach_cleanser mustard_bottle potted_meat_can sugar_box tomato_soup_can cube power_drill
do

    gnome-terminal -e "bash -ci '\
        source ../../devel/setup.bash && \
        export ROS_MASTER_URI=http://localhost:$node_id && \
        roslaunch launch/ur5_robotic_moveit_ros.launch planner:=ompl;'"
    sleep 3

    gnome-terminal -e "bash -ci '\
        source ../../devel/setup.bash && \
        export ROS_MASTER_URI=http://localhost:$node_id && \
        python run_dynamic_with_motion.py \
            --object_name $object_name \
            --robot_config_name ur5_robotiq \
            --motion_mode dynamic_linear \
            --rendering \
            --num_trials 100 \
            --conveyor_z_low 0.01 \
            --conveyor_z_high 0.3 \
            --result_dir $timestr \
            --grasp_database_path assets/grasps/filtered_grasps_noise_robotiq_100_1.00 \
            --baseline_experiment_path fast_obstacles \
            --grasp_threshold 0.1 \
            --lazy_threshold  0.3 \
            --conveyor_speed 0.05 \
            --close_delay 0.5 \
            --back_off -0.075 \
            --distance_low 0.3 \
            --distance_high 0.7 \
            --pose_freq 5 \
            --record_videos \
            --max_check 10 \
            --always_try_switching \
            --use_joint_space_dist \
            --use_box \
            --use_kf \
            | tee ${timestr}/${object_name}_output.file; $SHELL'"
    sleep 3
    ((node_id++))
done
            # --load_obstacles \