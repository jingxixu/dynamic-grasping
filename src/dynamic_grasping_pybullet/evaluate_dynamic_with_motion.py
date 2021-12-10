from __future__ import division
import numpy as np
import pandas as pd
import argparse
import os
import ast
import matplotlib.pyplot as plt
import pprint
import json
from collections import OrderedDict, defaultdict

"""
Given a directory of results for grasping dynamic object with planned motion,
evaluate success rates.
"""

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)


def get_args():
    parser = argparse.ArgumentParser(description='Analyse results.')
    parser.add_argument('--result_dir', type=str, required=True)
    args = parser.parse_args()
    return args


def evaluate_results(df):
    stats = {}
    df_success = df.loc[df['success']]

    stats['num_successes'] = len(df_success)
    stats['num_trials'] = len(df)
    stats['success_rate'] = stats['num_successes'] / stats['num_trials']
    stats['dynamic_grasping_time'] = df.mean().dynamic_grasping_time
    stats['dynamic_grasping_time_success_only'] = df_success.mean().dynamic_grasping_time

    if 'grasp_switched_list' in df:
        stats['num_grasp_switched'] = df['grasp_switched_list'].apply(
            lambda x: np.array(sum(ast.literal_eval(x)))).mean()
    if 'num_ik_called_list' in df:
        stats['num_ik_called'] = df['num_ik_called_list'].apply(lambda x: np.array(np.mean(ast.literal_eval(x)))).mean()

    return stats


def get_overall_stats(stat_dict):
    overall_stats = defaultdict(list)

    for object_name in stat_dict:
        for stat in stat_dict[object_name]:
            overall_stats[stat].append(stat_dict[object_name][stat])

    overall_stats['success_rate'] = sum(overall_stats['num_successes']) / sum(overall_stats['num_trials'])
    overall_stats['num_successes'] = sum(overall_stats['num_successes'])
    overall_stats['num_trials'] = sum(overall_stats['num_trials'])
    for key in ['dynamic_grasping_time', 'dynamic_grasping_time_success_only', 'num_grasp_switched', 'num_ik_called']:
        if key in overall_stats:
            overall_stats[key] = np.average(overall_stats[key])
    return overall_stats


if __name__ == '__main__':
    args = get_args()

    object_names = [n for n in os.listdir(args.result_dir) if os.path.isdir(os.path.join(args.result_dir, n))]
    csv_paths = [os.path.join(object_name, object_name+'.csv') for object_name in object_names]
    stat_dict = OrderedDict()
    for name, path in zip(object_names, csv_paths):
        result_file_path = os.path.join(args.result_dir, path)
        if not os.path.exists(result_file_path):
            print('{} not found'.format(result_file_path))
            continue
        df = pd.read_csv(result_file_path, index_col=0)
        stats = evaluate_results(df)
        stat_dict[name] = stats

    overall_stats = get_overall_stats(stat_dict)
    stat_dict['summary'] = overall_stats
    print(json.dumps(stat_dict, indent=4))
    with open(os.path.join(args.result_dir, 'results.json'), "w") as f:
        json.dump(stat_dict, f, indent=4)

