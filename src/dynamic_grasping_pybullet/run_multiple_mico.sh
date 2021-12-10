#!/usr/bin/env bash

# initial node id
node_id=10000

### Motion-aware
# linear 3 cm/s
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_ma_linear \
  --motion_mode dynamic_linear \
  --use_reachability false \
  --use_motion_aware true \
  --baseline_experiment_path assets/benchmark_tasks/mico/linear_tasks_mico \
  --conveyor_speed 0.03;
let "node_id=node_id+1";
sleep 5;

# linear 3 cm/s, with obstacles
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_ma_linear_ob \
  --motion_mode dynamic_linear \
  --use_reachability false \
  --use_motion_aware true \
  --baseline_experiment_path assets/benchmark_tasks/mico/linear_tasks_mico \
  --conveyor_speed 0.03 \
  --load_obstacles true;
let "node_id=node_id+1";
sleep 5;

# linear 2 cm/s, with slab
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_ma_linear_slab \
  --motion_mode dynamic_linear \
  --use_reachability false \
  --use_motion_aware true \
  --baseline_experiment_path assets/benchmark_tasks/mico/linear_tasks_mico \
  --conveyor_speed 0.02 \
  --add_top_shelf true;
let "node_id=node_id+1";
sleep 5;

# linear 3 cm/s, with z motion
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_ma_linear_z \
  --motion_mode dynamic_linear \
  --use_reachability false \
  --use_motion_aware true \
  --baseline_experiment_path assets/benchmark_tasks/mico/vary_z_tasks_mico \
  --conveyor_speed 0.03;
let "node_id=node_id+1";
sleep 5;

# linear 3 cm/s, vary speed
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_ma_linear_vary \
  --motion_mode dynamic_linear_vary_speed \
  --use_reachability false \
  --use_motion_aware true \
  --baseline_experiment_path assets/benchmark_tasks/mico/linear_tasks_mico \
  --conveyor_speed 0.03;
let "node_id=node_id+1";
sleep 5;

# circular 2 cm/s
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_ma_circular \
  --motion_mode dynamic_circular \
  --use_reachability false \
  --use_motion_aware true \
  --baseline_experiment_path assets/benchmark_tasks/mico/circular_tasks_mico \
  --conveyor_speed 0.02;
let "node_id=node_id+1";
sleep 5;

# sinusoidal 1 cm/s
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_ma_sinusoid \
  --motion_mode dynamic_sinusoid \
  --use_reachability false \
  --use_motion_aware true \
  --use_gt true \
  --use_kf false \
  --baseline_experiment_path assets/benchmark_tasks/mico/linear_tasks_mico \
  --conveyor_speed 0.01;
let "node_id=node_id+1";
sleep 5;

### Reachability-aware
# linear 3 cm/s
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_ra_linear \
  --motion_mode dynamic_linear \
  --use_reachability true \
  --use_motion_aware false \
  --baseline_experiment_path assets/benchmark_tasks/mico/linear_tasks_mico \
  --conveyor_speed 0.03;
let "node_id=node_id+1";
sleep 5;

# linear 3 cm/s, with obstacles
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_ra_linear_ob \
  --motion_mode dynamic_linear \
  --use_reachability true \
  --use_motion_aware false \
  --baseline_experiment_path assets/benchmark_tasks/mico/linear_tasks_mico \
  --conveyor_speed 0.03 \
  --load_obstacles true;
let "node_id=node_id+1";
sleep 5;

# linear 2 cm/s, with slab
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_ra_linear_slab \
  --motion_mode dynamic_linear \
  --use_reachability true \
  --use_motion_aware false \
  --baseline_experiment_path assets/benchmark_tasks/mico/linear_tasks_mico \
  --conveyor_speed 0.02 \
  --add_top_shelf true;
let "node_id=node_id+1";
sleep 5;

# linear 3 cm/s, with z motion
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_ra_linear_z \
  --motion_mode dynamic_linear \
  --use_reachability true \
  --use_motion_aware false \
  --baseline_experiment_path assets/benchmark_tasks/mico/vary_z_tasks_mico \
  --conveyor_speed 0.03;
let "node_id=node_id+1";
sleep 5;

# linear 3 cm/s, vary speed
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_ra_linear_vary \
  --motion_mode dynamic_linear_vary_speed \
  --use_reachability true \
  --use_motion_aware false \
  --baseline_experiment_path assets/benchmark_tasks/mico/linear_tasks_mico \
  --conveyor_speed 0.03;
let "node_id=node_id+1";
sleep 5;

# circular 2 cm/s
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_ra_circular \
  --motion_mode dynamic_circular \
  --use_reachability true \
  --use_motion_aware false \
  --baseline_experiment_path assets/benchmark_tasks/mico/circular_tasks_mico \
  --conveyor_speed 0.02;
let "node_id=node_id+1";
sleep 5;

# sinusoidal 1 cm/s
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_ra_sinusoid \
  --motion_mode dynamic_sinusoid \
  --use_reachability true \
  --use_motion_aware false \
  --use_gt true \
  --use_kf false \
  --baseline_experiment_path assets/benchmark_tasks/mico/linear_tasks_mico \
  --conveyor_speed 0.01;
let "node_id=node_id+1";
sleep 5;

### Reachability-aware + Motion-aware
# linear 3 cm/s
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_ma_ra_linear \
  --motion_mode dynamic_linear \
  --use_reachability true \
  --use_motion_aware true \
  --baseline_experiment_path assets/benchmark_tasks/mico/linear_tasks_mico \
  --conveyor_speed 0.03;
let "node_id=node_id+1";
sleep 5;

# linear 3 cm/s, with obstacles
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_ma_ra_linear_ob \
  --motion_mode dynamic_linear \
  --use_reachability true \
  --use_motion_aware true \
  --baseline_experiment_path assets/benchmark_tasks/mico/linear_tasks_mico \
  --conveyor_speed 0.03 \
  --load_obstacles true;
let "node_id=node_id+1";
sleep 5;

# linear 2 cm/s, with slab
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_ma_ra_linear_slab \
  --motion_mode dynamic_linear \
  --use_reachability true \
  --use_motion_aware true \
  --baseline_experiment_path assets/benchmark_tasks/mico/linear_tasks_mico \
  --conveyor_speed 0.02 \
  --add_top_shelf true;
let "node_id=node_id+1";
sleep 5;

# linear 3 cm/s, with z motion
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_ma_ra_linear_z \
  --motion_mode dynamic_linear \
  --use_reachability true \
  --use_motion_aware true \
  --baseline_experiment_path assets/benchmark_tasks/mico/vary_z_tasks_mico \
  --conveyor_speed 0.03;
let "node_id=node_id+1";
sleep 5;

# linear 3 cm/s, vary speed
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_ma_ra_linear_vary \
  --motion_mode dynamic_linear_vary_speed \
  --use_reachability true \
  --use_motion_aware true \
  --baseline_experiment_path assets/benchmark_tasks/mico/linear_tasks_mico \
  --conveyor_speed 0.03;
let "node_id=node_id+1";
sleep 5;

# circular 2 cm/s
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_ma_ra_circular \
  --motion_mode dynamic_circular \
  --use_reachability true \
  --use_motion_aware true \
  --baseline_experiment_path assets/benchmark_tasks/mico/circular_tasks_mico \
  --conveyor_speed 0.02;
let "node_id=node_id+1";
sleep 5;

# sinusoidal 1 cm/s
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_ma_ra_sinusoid \
  --motion_mode dynamic_sinusoid \
  --use_reachability true \
  --use_motion_aware true \
  --use_gt true \
  --use_kf false \
  --baseline_experiment_path assets/benchmark_tasks/mico/linear_tasks_mico \
  --conveyor_speed 0.01;
let "node_id=node_id+1";
sleep 5;

### No-seeding
# linear 3 cm/s
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_no_seed_linear \
  --motion_mode dynamic_linear \
  --use_reachability true \
  --use_motion_aware false \
  --baseline_experiment_path assets/benchmark_tasks/mico/linear_tasks_mico \
  --conveyor_speed 0.03 \
  --use_seed_trajectory false;
let "node_id=node_id+1";
sleep 5;

# linear 3 cm/s, with obstacles
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_no_seed_linear_ob \
  --motion_mode dynamic_linear \
  --use_reachability true \
  --use_motion_aware false \
  --baseline_experiment_path assets/benchmark_tasks/mico/linear_tasks_mico \
  --conveyor_speed 0.03 \
  --load_obstacles true \
  --use_seed_trajectory false;
let "node_id=node_id+1";
sleep 5;

# linear 2 cm/s, with slab
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_no_seed_linear_slab \
  --motion_mode dynamic_linear \
  --use_reachability true \
  --use_motion_aware false \
  --baseline_experiment_path assets/benchmark_tasks/mico/linear_tasks_mico \
  --conveyor_speed 0.02 \
  --add_top_shelf true \
  --use_seed_trajectory false;
let "node_id=node_id+1";
sleep 5;

# linear 3 cm/s, with z motion
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_no_seed_linear_z \
  --motion_mode dynamic_linear \
  --use_reachability true \
  --use_motion_aware false \
  --baseline_experiment_path assets/benchmark_tasks/mico/vary_z_tasks_mico \
  --conveyor_speed 0.03 \
  --use_seed_trajectory false;
let "node_id=node_id+1";
sleep 5;

# linear 3 cm/s, vary speed
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_no_seed_linear_vary \
  --motion_mode dynamic_linear_vary_speed \
  --use_reachability true \
  --use_motion_aware false \
  --baseline_experiment_path assets/benchmark_tasks/mico/linear_tasks_mico \
  --conveyor_speed 0.03 \
  --use_seed_trajectory false;
let "node_id=node_id+1";
sleep 5;

# circular 2 cm/s
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_no_seed_circular \
  --motion_mode dynamic_circular \
  --use_reachability true \
  --use_motion_aware false \
  --baseline_experiment_path assets/benchmark_tasks/mico/circular_tasks_mico \
  --conveyor_speed 0.02 \
  --use_seed_trajectory false;
let "node_id=node_id+1";
sleep 5;

# sinusoidal 1 cm/s
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_no_seed_sinusoid \
  --motion_mode dynamic_sinusoid \
  --use_reachability true \
  --use_motion_aware false \
  --use_gt true \
  --use_kf false \
  --baseline_experiment_path assets/benchmark_tasks/mico/linear_tasks_mico \
  --conveyor_speed 0.01 \
  --use_seed_trajectory false;
let "node_id=node_id+1";
sleep 5;


### Baseline: NO-Reachability-aware + NO Motion-aware
# linear 3 cm/s
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_b_linear \
  --motion_mode dynamic_linear \
  --use_reachability false \
  --use_motion_aware false \
  --baseline_experiment_path assets/benchmark_tasks/mico/linear_tasks_mico \
  --conveyor_speed 0.03;
let "node_id=node_id+1";
sleep 5;

# linear 3 cm/s, with obstacles
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_b_linear_ob \
  --motion_mode dynamic_linear \
  --use_reachability false \
  --use_motion_aware false \
  --baseline_experiment_path assets/benchmark_tasks/mico/linear_tasks_mico \
  --conveyor_speed 0.03 \
  --load_obstacles true;
let "node_id=node_id+1";
sleep 5;

# linear 2 cm/s, with slab
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_b_linear_slab \
  --motion_mode dynamic_linear \
  --use_reachability false \
  --use_motion_aware false \
  --baseline_experiment_path assets/benchmark_tasks/mico/linear_tasks_mico \
  --conveyor_speed 0.02 \
  --add_top_shelf true;
let "node_id=node_id+1";
sleep 5;

# linear 3 cm/s, with z motion
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_b_linear_z \
  --motion_mode dynamic_linear \
  --use_reachability false \
  --use_motion_aware false \
  --baseline_experiment_path assets/benchmark_tasks/mico/vary_z_tasks_mico \
  --conveyor_speed 0.03;
let "node_id=node_id+1";
sleep 5;

# linear 3 cm/s, vary speed
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_b_linear_vary \
  --motion_mode dynamic_linear_vary_speed \
  --use_reachability false \
  --use_motion_aware false \
  --baseline_experiment_path assets/benchmark_tasks/mico/linear_tasks_mico \
  --conveyor_speed 0.03;
let "node_id=node_id+1";
sleep 5;

# circular 2 cm/s
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_b_circular \
  --motion_mode dynamic_circular \
  --use_reachability false \
  --use_motion_aware false \
  --baseline_experiment_path assets/benchmark_tasks/mico/circular_tasks_mico \
  --conveyor_speed 0.02;
let "node_id=node_id+1";
sleep 5;

# sinusoidal 1 cm/s
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_b_sinusoid \
  --motion_mode dynamic_sinusoid \
  --use_reachability false \
  --use_motion_aware false \
  --use_gt true \
  --use_kf false \
  --baseline_experiment_path assets/benchmark_tasks/mico/linear_tasks_mico \
  --conveyor_speed 0.01;
let "node_id=node_id+1";
sleep 5;

### NO-KF
# linear 3 cm/s
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_no_kf_linear \
  --motion_mode dynamic_linear \
  --use_reachability true \
  --use_motion_aware false \
  --use_kf false \
  --baseline_experiment_path assets/benchmark_tasks/mico/linear_tasks_mico \
  --conveyor_speed 0.03;
let "node_id=node_id+1";
sleep 5;

# linear 3 cm/s, with obstacles
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_no_kf_linear_ob \
  --motion_mode dynamic_linear \
  --use_reachability true \
  --use_motion_aware false \
  --use_kf false \
  --baseline_experiment_path assets/benchmark_tasks/mico/linear_tasks_mico \
  --conveyor_speed 0.03 \
  --load_obstacles true;
let "node_id=node_id+1";
sleep 5;

# linear 2 cm/s, with slab
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_no_kf_linear_slab \
  --motion_mode dynamic_linear \
  --use_reachability true \
  --use_motion_aware false \
  --use_kf false \
  --baseline_experiment_path assets/benchmark_tasks/mico/linear_tasks_mico \
  --conveyor_speed 0.02 \
  --add_top_shelf true;
let "node_id=node_id+1";
sleep 5;

# linear 3 cm/s, with z motion
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_no_kf_linear_z \
  --motion_mode dynamic_linear \
  --use_reachability true \
  --use_motion_aware false \
  --use_kf false \
  --baseline_experiment_path assets/benchmark_tasks/mico/vary_z_tasks_mico \
  --conveyor_speed 0.03;
let "node_id=node_id+1";
sleep 5;

# linear 3 cm/s, vary speed
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_no_kf_linear_vary \
  --motion_mode dynamic_linear_vary_speed \
  --use_reachability true \
  --use_motion_aware false \
  --use_kf false \
  --baseline_experiment_path assets/benchmark_tasks/mico/linear_tasks_mico \
  --conveyor_speed 0.03;
let "node_id=node_id+1";
sleep 5;

# circular 2 cm/s
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_no_kf_circular \
  --motion_mode dynamic_circular \
  --use_reachability true \
  --use_motion_aware false \
  --use_kf false \
  --baseline_experiment_path assets/benchmark_tasks/mico/circular_tasks_mico \
  --conveyor_speed 0.02;
let "node_id=node_id+1";
sleep 5;

# sinusoidal 1 cm/s
echo node id, ${node_id}
./run_single_mico.sh --node_id ${node_id} \
  --exp_name mico_no_kf_sinusoid \
  --motion_mode dynamic_sinusoid \
  --use_reachability true \
  --use_motion_aware false \
  --use_kf false \
  --baseline_experiment_path assets/benchmark_tasks/mico/linear_tasks_mico \
  --conveyor_speed 0.01;
let "node_id=node_id+1";
sleep 5;