#!/bin/env python
"""
Script for redistributing images into training and testing directories.
"""

import os
import random
import shutil

def redistribute_images(download_dir, train_dir, test_dir, train_ratio=0.7):
    """
    Move images from the download directory to train and test directories based on the specified ratio.
    """
    images = [f for f in os.listdir(download_dir) if os.path.isfile(os.path.join(download_dir, f))]
    total_images = len(images)
    train_count = int(total_images * train_ratio)
    
    random.shuffle(images)
    
    train_images = images[:train_count]
    test_images = images[train_count:]
    
    for img in train_images:
        shutil.move(os.path.join(download_dir, img), os.path.join(train_dir, img))
    
    for img in test_images:
        shutil.move(os.path.join(download_dir, img), os.path.join(test_dir, img))

def count_images_in_folder(folder_path):
    """
    Count the number of images in a folder.
    """
    count = 0
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            count += 1
    return count

if __name__ == "__main__":
    download_dir = "output_directory/panstamps"
    train_dir = "output_directory/train"
    test_dir = "output_directory/test"

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    redistribute_images(download_dir, train_dir, test_dir, train_ratio=3398/4855)

    num_train_images = count_images_in_folder(train_dir)
    num_test_images = count_images_in_folder(test_dir)

    print(f"Number of training images: {num_train_images}")
    print(f"Number of test images: {num_test_images}")
