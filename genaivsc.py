
import os
import shutil
import pandas as pd
import logging
import concurrent.futures
import random
import time
import requests
import urllib
import csv
import sys
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm

def load_images_from_folder(folder_path, img_height, img_width, channels=3):
    images = []
    for filename in tqdm(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            try:
                img = load_img(img_path, target_size=(img_height, img_width), color_mode='rgb' if channels == 3 else 'grayscale')
                img_array = img_to_array(img)
                images.append(img_array)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return np.array(images)

# Define paths to the unzipped directories
train_dir = r'D:\GenAI\output_directory\train'
test_dir = r'D:\GenAI\output_directory\test'
def count_files_in_directory(folder_path):
    count = 0
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            count += 1
    return count

# Load images
train_images = load_images_from_folder(train_dir, img_height=128, img_width=128, channels=3)
test_images = load_images_from_folder(test_dir, img_height=128, img_width=128, channels=3)

print(f"Loaded {train_images.shape[0]} training images with shape {train_images.shape[1:]}")
print(f"Loaded {test_images.shape[0]} testing images with shape {test_images.shape[1:]}")
train_files_count = count_files_in_directory(train_dir)
test_files_count = count_files_in_directory(test_dir)

print(f"Number of files in train directory: {train_files_count}")
print(f"Number of files in test directory: {test_files_count}")
#pip install tensorflow numpy matplotlib tqdm
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

# Image dimensions
IMG_HEIGHT = 64
IMG_WIDTH = 64
CHANNELS = 3
def load_images_from_folder(folder_path, img_height, img_width, channels):
    images = []
    for filename in tqdm(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            img = load_img(img_path, target_size=(img_height, img_width))
            img_array = img_to_array(img)
            images.append(img_array)
    return np.array(images)

train_images = load_images_from_folder(train_dir, IMG_HEIGHT, IMG_WIDTH, CHANNELS)
train_images = train_images / 127.5 - 1  # Normalize images to [-1, 1]
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 256)))
    assert model.output_shape == (None, 8, 8, 256)

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(CHANNELS, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 64, 64, CHANNELS)

    return model
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[IMG_HEIGHT, IMG_WIDTH, CHANNELS]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator = build_generator()
discriminator = build_discriminator()

generator_optimizer = tf.keras.optimizers.Adam(1e-5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

import time

EPOCHS = 200
BATCH_SIZE = 32
NOISE_DIM = 100
NUM_EXAMPLES_TO_GENERATE = 16

# Seed to visualize progress
seed = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM])
def train(dataset, epochs):
    with open('training_metrics.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Gen Loss", "Disc Loss", "Time (s)"])

        for epoch in range(epochs):
            start = time.time()

            for image_batch in dataset:
                gen_loss, disc_loss = train_step(image_batch)

            # Produce images for the GIF as we go
            generate_and_save_images(generator, epoch + 1, seed)

            epoch_time = time.time() - start
            print(f'Epoch {epoch + 1}, Gen Loss: {gen_loss}, Disc Loss: {disc_loss}, Time: {epoch_time:.2f} sec')
            writer.writerow([epoch + 1, gen_loss.numpy(), disc_loss.numpy(), epoch_time])


    # Generate after the final epoch
    generate_and_save_images(generator, epochs, seed)

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i, :, :, :] + 1) / 2)
        plt.axis('off')

    plt.savefig(f'image_at_epoch_{epoch:04d}.png')
    plt.close(fig)

BUFFER_SIZE = 60000
BATCH_SIZE = 32

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

train(train_dataset, EPOCHS)
