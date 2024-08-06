# Training GANs on Google Colab

This script is designed to run on Google Colab to take advantage of additional GPU resources. It handles image loading from uploaded zip files, model training, and image generation.

## Features
- Upload and extract zip files with training and testing images.
- Define and build GAN models (Generator and Discriminator).
- Train models and save generated images.
## Usage

1. Upload `test.zip` and `train.zip` files to Colab.
2. Run the script:
   ```python
   !python genai.py
## Configuration

- `train_dir`, `test_dir`: Directories for extracted images.
- `IMG_HEIGHT`, `IMG_WIDTH`: Image dimensions.
- `EPOCHS`, `BATCH_SIZE`: Training parameters.
## Dependencies

- `tensorflow`
- `numpy`
- `matplotlib`
- `tqdm`

Install them using:
```python
!pip install tensorflow numpy matplotlib tqdm
