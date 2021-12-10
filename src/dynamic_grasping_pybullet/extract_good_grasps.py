import os
import argparse
import pandas as pd
import tqdm
import numpy as np


"""
Extract good grasps from a grasp_folder_path to output_folder_path based on the result csv.

grasp_folder_path
    - cube
        - result.csv
        - grasp_0000.npy
        - grasp_0001.npy
        - ...
    - bleach_cleanser
    - ...
    
output_folder_path
    - cube.npy
    - bleach_cleanser.npy
    - ...
"""


def get_args():
    parser = argparse.ArgumentParser(description='Run Dynamic Grasping Experiment')

    parser.add_argument('--grasp_folder_path', type=str, required=True)
    parser.add_argument('--output_folder_path', type=str, required=True)
    parser.add_argument('--num_grasps', type=int, help='number of good grasps to extract')
    args = parser.parse_args()

    if not os.path.exists(args.output_folder_path):
        os.makedirs(args.output_folder_path)

    return args


if __name__ == "__main__":
    args = get_args()

    object_names = os.listdir(args.grasp_folder_path)
    for obj in object_names:
        grasps = []
        source_folder_path = os.path.join(args.grasp_folder_path, obj)
        dest_file_path = os.path.join(args.output_folder_path, obj+'.npy')
        result_file_path = os.path.join(source_folder_path, 'result.csv')
        df = pd.read_csv(result_file_path, index_col=0)
        df_success = df.loc[df['num_successes'] / df['num_trials'] >= 1]

        num_grasps = len(df_success) if args.num_grasps is None else args.num_grasps
        bar = tqdm.tqdm(total=num_grasps, desc=obj)
        for index in range(num_grasps):
            row = df_success.iloc[index]
            # command = 'cp '+os.path.join(source_folder_path, row['grasp_fnm'])+' '+dest_folder_path
            # os.system(command)
            grasps.append(np.load(os.path.join(source_folder_path, row['grasp_fnm']), allow_pickle=True))
            bar.update(1)
            bar.set_description(obj + ' | ' + row['grasp_fnm'])
        bar.close()
        np.save(dest_file_path, np.array(grasps))
