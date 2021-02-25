import os
import numpy as np
import argparse

def split_data_list(list_file, save_path, split_ratio=0.8):
    with open(list_file, "r") as f:
        label_list = f.readlines()
    size = len(label_list)
    train_size = int(split_ratio*size)
    indices = np.random.permutation(size)
    training_idx, validating_idx = indices[:train_size], indices[train_size:]
    with open(os.path.join(save_path, "train.txt"), "w", encoding="utf-8") as train_list_file:
        for i in training_idx:
            train_list_file.write(label_list[i])

    with open(os.path.join(save_path, "val.txt"), "w", encoding="utf-8") as val_list_file:
        for i in validating_idx:
            val_list_file.write(label_list[i])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--file-path", type=str, required=True, help="the path of list file")
    parser.add_argument("-r", "--split-ratio", type = float, help="the ratio of train data to whole data")
    parser.add_argument("-s", "--save-path", type=str, help="the path to save generated list files")
    args = parser.parse_args()
    if args.save_path is None:
         args.save_path = os.path.split(args.file_path)[0]
    if args.split_ratio is None:
        split_data_list(args.file_path, args.save_path)
    else:
        split_data_list(args.file_path, args.save_path, args.split_ratio)