import os
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint_path', type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    object_names = os.listdir(args.checkpoint_path)
    for n in object_names:
        best_acc = 0
        best_epoch_name = None
        epoch_folders = [f for f in os.listdir(os.path.join(args.checkpoint_path, n)) if os.path.isdir(os.path.join(args.checkpoint_path, n, f))]
        epoch_folders.sort()
        for e in epoch_folders:
            parts = e.split('_')
            test_acc = float(parts[3])
            if test_acc >= best_acc:
                best_acc = test_acc
                best_epoch_name = e
        print(os.path.join(args.checkpoint_path, n, best_epoch_name))
