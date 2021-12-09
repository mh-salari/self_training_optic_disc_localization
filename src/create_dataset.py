#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 08 2021
@author: MohammadHossein Salari
@email: mohammad.hossein.salari@gmail.com
"""

import os
import cv2
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from utils.image_stuff import image_stuff


def split_and_save_images(images_name, images_path, dataset_outout_path):

    # split images to 70% train, 15% val, 15% test
    splits = np.split(
        images_name, [int(0.7 * len(images_name)), int(0.85 * len(images_name))]
    )

    for idx, sub_dir in enumerate(["train", "val", "test"]):
        bbox_average = []
        total = 0
        output_path = os.path.join(dataset_outout_path, sub_dir)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for image_name in tqdm(splits[idx]):
            image_name = image_name.replace("_bbox", "")
            bbox_name = image_name.replace(".jpg", ".txt")

            bbox = []
            with open(os.path.join(images_path, bbox_name), "r") as f:
                for line in f:
                    bbox.append(int(line.strip()))

            image = cv2.imread(os.path.join(images_path, image_name))

            resized_image, ratio, bbox_delta = image_stuff.resize_and_pad(
                image, 224, PADDING=True
            )

            resized_bbox = [int(x * ratio) for x in bbox]
            resized_bbox = [a + b for a, b in zip(resized_bbox, bbox_delta)]

            bbox_ratio = (resized_bbox[0] - resized_bbox[2]) / (
                resized_bbox[1] - resized_bbox[3]
            )

            # filter out nun perfect bonding boxes
            if 0.8 <= bbox_ratio <= 1.1:
                cv2.imwrite(os.path.join(output_path, image_name), resized_image)
                with open(os.path.join(output_path, bbox_name), "w") as f:
                    for coordinate in resized_bbox:
                        f.write(f"{int(coordinate)}\n")

                bbox_average.append(bbox_ratio)
                total += 1
        print(f"Total saved images: {total}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Make dataset")

    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    input_dataset_path = args.input_path
    output_dataset_path = args.output_path

    if os.path.exists(output_dataset_path):
        try:
            shutil.rmtree(output_dataset_path)

        except OSError as e:
            print(f"Error: {e.filename} - { e.strerror}.")
    else:
        os.mkdir(output_dataset_path)

    images_name_list = []
    for root, dirs, files in os.walk(input_dataset_path):
        for file in files:
            if "bbox" not in file and ".jpg" in file:
                images_name_list.append(file)

    split_and_save_images(images_name_list, input_dataset_path, output_dataset_path)
