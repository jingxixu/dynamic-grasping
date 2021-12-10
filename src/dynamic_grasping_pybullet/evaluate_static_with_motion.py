from __future__ import division
import numpy as np
import pandas as pd
import argparse
import os
import ast
import matplotlib.pyplot as plt
import pprint

"""
Given a directory of results for grasping static object with planned motion,
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
    df_attempted = df.loc[df['grasp_attempted']]

    stats['num_successes'] = len(df_success)
    stats['num_trials'] = len(df)
    stats['num_attempted'] = len(df_attempted)
    stats['raw_success_rate'] = stats['num_successes'] / stats['num_trials']
    stats['attemped_success_rate'] = stats['num_successes'] / stats['num_attempted']
    stats['grasp_panning_time'] = df.mean().grasp_planning_time
    stats['num_ik_called'] = df.mean().num_ik_called
    return stats


def get_overall_stats(stat_list):
    overall_stats = {}

    num_successes_list = []
    num_trials_list = []
    num_attempted_list = []
    raw_success_rate_list = []
    attemped_success_rate_list = []
    grasp_planning_time_list = []
    num_ik_called_list = []
    for stats in stat_list:
        num_successes_list.append(stats['num_successes'])
        num_trials_list.append(stats['num_trials'])
        num_attempted_list.append(stats['num_attempted'])
        raw_success_rate_list.append(stats['raw_success_rate'])
        attemped_success_rate_list.append(stats['attemped_success_rate'])
        grasp_planning_time_list.append(stats['grasp_panning_time'])
        num_ik_called_list.append(stats['num_ik_called'])

    overall_stats['num_successes'] = sum(num_successes_list)
    overall_stats['num_trials'] = sum(num_trials_list)
    overall_stats['num_attempted'] = sum(num_attempted_list)
    overall_stats['raw_success_rate'] = overall_stats['num_successes'] / overall_stats['num_trials']
    overall_stats['attemped_success_rate'] = overall_stats['num_successes'] / overall_stats['num_attempted']
    overall_stats['grasp_panning_time'] = np.average(grasp_planning_time_list)
    overall_stats['num_ik_called'] = np.average(num_ik_called_list)
    return overall_stats


if __name__ == '__main__':
    args = get_args()

    csv_names = os.listdir(args.result_dir)
    stat_list = []
    for n in csv_names:
        result_file_path = os.path.join(args.result_dir, n)
        df = pd.read_csv(result_file_path, index_col=0)
        stats = evaluate_results(df)
        stat_list.append(stats)

        print('')
        print(n)
        print("Statistics:")
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(stats)

    overall_stats = get_overall_stats(stat_list)
    print('')
    print('Summary')
    print("Statistics:")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(overall_stats)

