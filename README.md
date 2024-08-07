# Galaxy Image Generation with GANs

This project aims to generate synthetic galaxy images using Generative Adversarial Networks (GANs). It includes scripts for downloading galaxy images from the SDSS database, training GAN models, and managing the dataset.

## Table of Contents
- [Scripts Overview](#scripts-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Scripts Overview

- **`Downloading_SDSS.py`**: Downloads galaxy image cutouts from SDSS.
- **`Redistributing_images.py`**: Redistributing the images into train and test.
- **`genaivsc.py`**: Trains GANs locally with the provided dataset.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Tjschndrn/Gen-AI.git
    ```

2. Navigate to the repository directory:
    ```bash
    cd Gen-AI
    ```

3. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Downloading Images

To download SDSS galaxy images, run:
bash

python Downloading_SDSS.py

Note: Update the csv_path variable in the script to point to your catalog file.

### 2. Redistributing Images

To redistribute images into train and test, run:
bash

python Redistributing_images.py

### 3. Training GAN Locally
To train GANs locally, run:

bash

python genaivsc.py

Make sure to update the paths for train_dir and test_dir as per your setup.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your proposed changes. Ensure to follow the coding standards and add relevant documentation.
