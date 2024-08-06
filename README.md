# Galaxy Image Generation with GANs

This project aims to generate synthetic galaxy images using Generative Adversarial Networks (GANs). It includes scripts for downloading galaxy images from the SDSS database, training GAN models, and managing the dataset. The project is designed to utilize both local and cloud-based computing resources for enhanced performance.

## Table of Contents
- [Scripts Overview](#scripts-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Scripts Overview

- **`Downloading_sorting.py`**: Downloads galaxy image cutouts from SDSS and sorts them into training and testing sets.
- **`genaivsc.py`**: Trains GANs locally using VS Code with the provided dataset.
- **`genai.py`**: Trains GANs on Google Colab with additional GPU resources.

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

### 1. Downloading and Sorting Images

To download and sort SDSS galaxy images, run:
bash

python Downloading_sorting.py

Note: Update the csv_path variable in the script to point to your catalog file.

### 2. Training GAN Locally
To train GANs locally using VS Code, run:

bash

python genaivsc.py

Make sure to update the paths for train_dir and test_dir as per your setup.

### 3. Training GAN on Google Colab
To train GANs using Google Colab, upload the test.zip and train.zip files and run:

python
!python genai.py

Follow the instructions in the script for uploading files and setting up the environment.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your proposed changes. Ensure to follow the coding standards and add relevant documentation.
