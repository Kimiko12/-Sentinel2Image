# Sentinel-2 Image Matching

This repository provides a robust solution for identifying similar Sentinel-2 satellite images captured at different time intervals, under varying seasonal conditions, and diverse weather scenarios. Leveraging state-of-the-art deep learning models for keypoint detection and feature matching, this project is ideal for applications such as environmental monitoring, urban planning, and disaster assessment.

---

## Features

- **Data Preprocessing**: Automates the preparation of Sentinel-2 `.SAFE` format images for analysis.
- **Feature Extraction**: Utilizes SuperPoint for detecting keypoints and extracting robust descriptors.
- **Keypoint Matching**: Employs SuperGlue, a graph neural network, for accurate feature matching between image pairs.

---

## Project Structure

```
Sentinel2ImageMatching/
├── data/                      # Raw and processed Sentinel-2 datasets
├── model_checkpoints/         # Directory to save trained models
├── config.py                  # Configuration file for constants and hyperparameters
├── dataset.py                 # Custom dataset class for image matching
├── data_preprocess.py         # Data preprocessing utility script
├── SuperPoint.py              # Feature extraction script using SuperPoint
├── SuperGlue_architecture.py  # Keypoint matching script using SuperGlue
├── train_superpoint.py        # Training script for SuperPoint model
├── train_superglue.py         # Training script for SuperGlue model
├── inference.py               # Inference script for image matching
├── utils.py                   # Utility functions and helpers
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── LICENSE                    # License information
```

---

## Getting Started

### Accessing the Dataset and Model Weights

Download the dataset and pre-trained model weights from the following Google Drive link:

[Google Drive - Sentinel-2 Image Matching Dataset and Model Weights](https://drive.google.com/drive/folders/1TJ0i8HkmOMWjQIMeQtR0PXWeHdGCSXtw?usp=drive_link)

### Prerequisites

- **Python**: Version 3.8 or higher
- **GPU**: Recommended for accelerated training and processing
- **Dependencies**: Listed in `requirements.txt`

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Kimiko12/Sentinel2Image.git
   cd Sentinel2Image
   ```

2. **Install Dependencies**

   Install the required Python packages using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download and Prepare Data**

   Ensure that the downloaded datasets and model weights are placed in the appropriate directories as specified in the `config.py` file.

---

## Configuration

Configure hyperparameters and file paths by updating the `config.py` file:

```python
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_SUPERPOINT = 50
EPOCHS_SUPERGLUE = 30
SEED = 42
LEARNING_RATE_SUPERPOINT = 1e-4
LEARNING_RATE_SUPERGLUE = 1e-4
MODEL_NAME_SUPERPOINT = 'superpoint_model'
MODEL_NAME_SUPERGLUE = 'superglue_model'
PATH_TO_MODEL_CHECKPOINTS = 'model_checkpoints/'
DATA_PATH = 'data/'
```

---

## Workflow

### 1. Data Preprocessing

**Script**: `data_preprocess.py`

Sentinel-2 images are provided in the `.SAFE` format containing `.jp2` (JPEG 2000) files. The preprocessing steps include:

- **Resampling**: Adjusts the spatial resolution to standardize pixel sizes.
- **Normalization**: Scales pixel intensity values to ensure uniformity across the dataset.
- **Resizing**: Reshapes images to a fixed resolution (224x224) for computational efficiency.
- **Format Conversion**: Converts `.jp2` files to `.jpeg` format for streamlined processing.

```bash
python data_preprocess.py
```

This script processes raw `.SAFE` datasets into standardized `.jpeg` images suitable for feature extraction.

### 2. Feature Extraction with SuperPoint

**Script**: `SuperPoint.py`

- **SuperPoint** is a deep learning-based model designed for detecting keypoints and extracting descriptors from images.
- Ensures robustness to changes in scale, rotation, and illumination, making it ideal for analyzing satellite imagery.
- Each image undergoes keypoint detection, and corresponding descriptors are generated for further processing.

```bash
python SuperPoint.py
```

This script generates keypoints and descriptors for each preprocessed image, storing them for the matching phase.

### 3. Keypoint Matching with SuperGlue

**Script**: `SuperGlue_architecture.py`

- **SuperGlue** is a graph neural network (GNN) tailored for feature matching, leveraging attention mechanisms and contextual information.
- Establishes robust correspondences between keypoints in image pairs, ensuring accurate matching despite significant variations due to seasonal, temporal, or atmospheric changes.

```bash
python SuperGlue_architecture.py
```

This script matches keypoints between image pairs, producing matched keypoint pairs that indicate similar regions across different images.

---

## How to Use

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Kimiko12/Sentinel2Image.git
   cd Sentinel2Image
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Preprocess Images**

   Run the preprocessing script to convert `.jp2` files into standardized `.jpeg` images:

   ```bash
   python data_preprocess.py
   ```

4. **Feature Extraction with SuperPoint**

   Generate keypoints and descriptors from the preprocessed images:

   ```bash
   python SuperPoint.py
   ```

5. **Keypoint Matching with SuperGlue**

   Perform feature matching between image pairs:

   ```bash
   python SuperGlue_architecture.py
   ```

6. **Inference and Visualization**

   Use the `inference.py` script to obtain and visualize matching results between new image pairs:

   ```python
   from inference import ImageMatcher

   matcher = ImageMatcher(model_path='model_checkpoints/', device='cuda')
   image1_path = 'data/preprocessed/image1.jpeg'
   image2_path = 'data/preprocessed/image2.jpeg'
   matches = matcher.match_images(image1_path, image2_path)
   matcher.visualize_matches(image1_path, image2_path, matches)
   ```

   **Expected Output:**

   ![Matched Images Result](https://github.com/user-attachments/assets/35a0ac40-d63d-4026-8a0b-06ec68ff7681/matched_images_result_4.png)

---

## Future Improvements

1. **Multi-Spectral Image Support**
   - Integrate the capability to process and match multi-spectral data from Sentinel-2, leveraging its multiple bands (e.g., Red, Green, Blue, Near-Infrared) for more accurate feature extraction and matching across different conditions and seasons.

2. **Improved Matching Accuracy with Deep Learning**
   - Fine-tune the **SuperPoint** and **SuperGlue** models for better matching accuracy under challenging scenarios, including cloudy or partially obscured images, by using more extensive training datasets or domain-specific fine-tuning.

3. **Geospatial Metadata Integration**
   - Incorporate **geospatial metadata** (such as location coordinates, time, and sensor data) to enhance feature matching, allowing for more context-aware image comparisons and reducing false matches.

4. **Higher Resolution for Enhanced Detail**
   - Increase the resolution of input images to capture finer details, leading to the detection of more keypoints. This is especially useful in areas where smaller features need to be identified for accurate matching.

---

## Results

The system effectively matches Sentinel-2 images captured under varying conditions, demonstrating robust performance in keypoint detection and feature matching.

![Matched Images Result](https://github.com/user-attachments/assets/35a0ac40-d63d-4026-8a0b-06ec68ff7681/matched_images_result_4.png)
