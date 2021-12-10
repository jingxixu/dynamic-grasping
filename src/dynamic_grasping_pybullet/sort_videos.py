import numpy as np
import pandas as pd
import argparse
import os
from collections import OrderedDict
import shutil


def get_args():
    parser = argparse.ArgumentParser(description='Analyse results.')
    parser.add_argument('--result_dir', type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    object_names = [n for n in os.listdir(args.result_dir) if os.path.isdir(os.path.join(args.result_dir, n))]

    csv_paths = [os.path.join(object_name, object_name + '.csv') for object_name in object_names]
    stat_dict = OrderedDict()
    for name, path in zip(object_names, csv_paths):
        result_file_path = os.path.join(args.result_dir, path)
        df = pd.read_csv(result_file_path, index_col=0)
        video_path = os.path.join(os.path.dirname(result_file_path), 'videos')

        success_video_path = os.path.join(video_path, 'success')
        os.makedirs(success_video_path)
        for idx in df.index[df['success']]:
            v_fname = '{}.mp4'.format(idx)
            shutil.move(os.path.join(video_path, v_fname), os.path.join(success_video_path, v_fname))

        failure_video_path = os.path.join(video_path, 'failure')
        os.makedirs(failure_video_path)
        for idx in df.index[~df['success']]:
            v_fname = '{}.mp4'.format(idx)
            shutil.move(os.path.join(video_path, v_fname), os.path.join(failure_video_path, v_fname))
