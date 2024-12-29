### Sentinel-2 Image Matching  

#### Overview  

This repository provides a robust solution for identifying similar Sentinel-2 satellite images captured at different time intervals, under varying seasonal conditions, and diverse weather scenarios. The project employs state-of-the-art deep learning models for keypoint detection and feature matching, making it suitable for applications such as environmental monitoring, urban planning, and disaster assessment.

---

### Project Workflow  

#### 1. **Data Preprocessing**  
##### Implemented in `data_preprocess.py`  

Sentinel-2 images are provided in the `.SAFE` format containing `.jp2` (JPEG 2000) files. To ensure compatibility with the feature extraction and matching pipeline, the following preprocessing steps are performed:  
- **Resampling**: Adjusts the spatial resolution to standardize pixel sizes.  
- **Normalization**: Scales pixel intensity values to ensure uniformity across the dataset.  
- **Resizing**: Reshapes images to a fixed resolution (256x256) for computational efficiency.  
- **Format Conversion**: Converts `.jp2` files to `.jpeg` format for streamlined processing.  

---

#### 2. **Feature Extraction with SuperPoint**  
##### Implemented in `SuperPoint.py`  

- **SuperPoint** is a deep learning-based model designed for detecting keypoints and extracting descriptors from images.  
- It ensures robustness to changes in scale, rotation, and illumination, making it ideal for analyzing satellite imagery.  
- Each image undergoes keypoint detection, and corresponding descriptors are generated for further processing.  

---

#### 3. **Keypoint Matching with SuperGlue**  
##### Implemented in `SuperGlue_architecture.py`  

- The descriptors generated by SuperPoint are passed into **SuperGlue**, a graph neural network (GNN) tailored for feature matching.  
- SuperGlue leverages attention mechanisms and contextual information to establish robust correspondences between keypoints in image pairs.  
- This approach ensures accurate matching, even in the presence of significant variations due to seasonal, temporal, or atmospheric changes.  

---

### How to Use  

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/Kimiko12/Sentinel2Image.git
   ```  

2. **Navigate to the Repository Folder**  
   ```bash
   cd Sentinel2Image
   ```  

3. **Install Dependencies**  
   Ensure you have the required Python packages installed. Use the following command to set up the environment:  
   ```bash
   pip install -r requirements.txt
   ```  

4. **Preprocess Images**  
   Run the preprocessing script `data_preprocess.py` to convert `.jp2` files into raster `.jpeg` images

5. **Feature Extraction with SuperPoint**  
   Run the `SuperPoint.py` script to generate keypoints and descriptors from the preprocessed images

6. **Keypoint Matching with SuperGlue**  
   Finally, input the keypoints, descriptors, and paths to the images into the `inference.py` script to obtain the final matching results

---
**Future Improvements**
Multi-Spectral Image Support

    Integrate the capability to process and match multi-spectral data from Sentinel-2, leveraging its multiple bands (e.g., Red, Green, Blue, Near-Infrared) for more accurate feature extraction and matching across different conditions and seasons.
Improved Matching Accuracy with Deep Learning

    Fine-tune the SuperPoint and SuperGlue models for better matching accuracy under challenging scenarios, including cloudy or partially obscured images, by using more extensive training datasets or domain-specific fine-tuning.
Geospatial Metadata Integration

    Incorporate geospatial metadata (such as location coordinates, time, and sensor data) to enhance feature matching, allowing for more context-aware image comparisons and reducing false matches.
Higher Resolution for Enhanced Detail

    Increasing the resolution of input images allows for capturing finer details, leading to the detection of more keypoints. This is especially useful in areas where smaller features need to be identified for accurate matching.

Results:


![matched_images_result_4](https://github.com/user-attachments/assets/35a0ac40-d63d-4026-8a0b-06ec68ff7681)

