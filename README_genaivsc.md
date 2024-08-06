# Training GANs Locally with VS Code

This script trains GANs using a local machine and VS Code. It handles image loading, model training, and saving generated images.

## Features
- Load and preprocess images from local directories.
- Define and build GAN models (Generator and Discriminator).
- Train models and save generated images.
## Usage

1. Ensure that the paths to `train_dir` and `test_dir` are correctly set.
2. Run the script:
   ```bash
   python genaivsc.py
## Configuration

- `train_dir`: Path to the training images directory.
- `test_dir`: Path to the testing images directory.
- `IMG_HEIGHT`, `IMG_WIDTH`: Image dimensions.
- `EPOCHS`, `BATCH_SIZE`: Training parameters.
## Dependencies

- `tensorflow`
- `numpy`
- `matplotlib`
- `tqdm`

Install them using:
```bash
pip install tensorflow numpy matplotlib tqdm
