#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 08 2021
@author: MohammadHossein Salari
@email: mohammad.hossein.salari@gmail.com
"""

import os
import cv2
from tqdm import tqdm
from utils.image_stuff import image_stuff


def normalize_and_resize(image_path, desired_size):
    image = cv2.imread(image_path)

    cropped_image = image_stuff.crop_eye_area(image)
    eq_image = image_stuff.fix_illumination(cropped_image)
    resized_image = eq_image
    if desired_size != 0:
        resized_image = image_stuff.resize_and_pad(
            eq_image, desired_size, PADDING=True
        )[0]

    return resized_image


if __name__ == "__main__":

    dataset_dir_path = "/home/hue/Codes/AIROGS/datasets/5/"
    outout_dir_path = "/home/hue/Codes/AIROGS/datasets/5_normalized_224x224"
    if not os.path.exists(outout_dir_path):
        os.mkdir(outout_dir_path)

    images_path = [
        os.path.join(dataset_dir_path, f)
        for f in sorted(os.listdir(f"{dataset_dir_path}"))
    ]
    print(f"Total number of images: {len(images_path)}")

    desired_size = 224
    for image_path in tqdm(images_path[:]):
        resized_image = normalize_and_resize(image_path, desired_size)

        image_name = os.path.basename(image_path)
        cv2.imwrite(
            os.path.join(outout_dir_path, f"{image_name[:-4]}_x{desired_size}.jpg"),
            resized_image,
        )
