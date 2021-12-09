#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 11 2021
@author: MohammadHossein Salari
@email: mohammad.hossein.salari@gmail.com
"""

import os
import shutil
import pickle
import random
import argparse
from tqdm import tqdm
import numpy as np
import cv2

# https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
def resize_and_pad(image, desired_size):

    old_size = image.shape[:2]  # old_size is in (height, width) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format

    new_image = cv2.resize(image, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_image = cv2.copyMakeBorder(
        new_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return new_image, ratio, [left, top, right, bottom]


def draw_bbox(image, bbox, class_name="", box_color=(255, 0, 0), thickness=2):
    """draw a single bounding box on the image"""
    img = image.copy()
    x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    cv2.rectangle(
        img, (x_min, y_min), (x_max, y_max), color=box_color, thickness=thickness
    )

    ((text_width, text_height), _) = cv2.getTextSize(
        class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35 * thickness, 1
    )
    cv2.rectangle(
        img,
        (x_min, y_min - int(1.3 * text_height)),
        (x_min + text_width, y_min),
        box_color,
        -1,
    )
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=(255, 255, 255),
        lineType=cv2.LINE_AA,
    )
    return img


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

            resized_image, ratio, bbox_delta = resize_and_pad(image, 224)
            cv2.imwrite(os.path.join(output_path, image_name), resized_image)

            resized_bbox = [int(x * ratio) for x in bbox]
            resized_bbox = [a + b for a, b in zip(resized_bbox, bbox_delta)]
            with open(os.path.join(output_path, bbox_name), "w") as f:
                for coordinate in resized_bbox:
                    f.write(f"{int(coordinate)}\n")
            bbox_average.append(
                (resized_bbox[0] - resized_bbox[2])
                / (resized_bbox[1] - resized_bbox[3])
            )
            total += 1
        print(sum(bbox_average) / len(bbox_average))
        print(max(bbox_average))
        print(min(bbox_average))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Make dataset")

    parser.add_argument("--ipath", type=str, required=True)
    parser.add_argument("--opath", type=str, required=True)
    parser.add_argument("--lpath", type=str, required=True)
    args = parser.parse_args()
    input_dataset_path = args.ipath
    output_dataset_path = args.opath
    labels_path = args.lpath

    # dataset_outout_path = os.path.join(output_dataset_path, "classification_images")

    # delete old dataset

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

    labels = pickle.load(open(labels_path, "rb"))

    shuffled_labels = {}
    for image_name, label in labels.items():
        if image_name.replace("_bbox", "") in images_name_list:
            shuffled_labels[image_name] = label
    keys = list(shuffled_labels.keys())
    random.shuffle(keys)
    shuffled_labels = dict([(key, shuffled_labels[key]) for key in keys])

    ok_images = [name for name, label in shuffled_labels.items() if label == "wrong"]

    print(f"Number of ok images: {len(ok_images)}")
    # import sys
    # sys.exit()
    # eq_labels = {}
    # num_inside = 0
    # num_outside = 0
    # eq_num = len(message_images)

    # for name, label in shuffled_labels.items():
    #     if label== "text":
    #         num_inside +=1
    #         eq_labels[name]=label
    #     if num_inside == eq_num:
    #         break
    # for name, label in shuffled_labels.items():
    #     if label== "other":
    #         num_outside +=1
    #         eq_labels[name]=label
    #     if num_outside == eq_num:
    #         break
    # print(len(eq_labels))

    split_and_save_images(ok_images, input_dataset_path, output_dataset_path)
