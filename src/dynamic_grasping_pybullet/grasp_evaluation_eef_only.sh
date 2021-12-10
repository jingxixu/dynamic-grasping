#!/usr/bin/env bash

for object_name in bleach_cleanser mustard_bottle potted_meat_can sugar_box tomato_soup_can cube power_drill
do
    python grasp_evaluation_eef_only.py --object_name $object_name
    sleep 5
done