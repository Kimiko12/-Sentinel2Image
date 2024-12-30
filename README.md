### Sentinel-2 Image Matching  

#### Overview  

This repository provides a comprehensive solution for identifying similar Sentinel-2 satellite images captured at different time intervals, under varying seasonal and weather conditions. It leverages state-of-the-art deep learning models for keypoint detection and feature matching, making it ideal for applications such as environmental monitoring, urban planning, and disaster assessment.

---

### Project Workflow  

#### Link to Model Weights and Dataset  
Access the model weights and dataset here: [Google Drive Link](https://drive.google.com/drive/folders/1TJ0i8HkmOMWjQIMeQtR0PXWeHdGCSXtw?usp=drive_link)

#### 1. **Data Preprocessing**  
- **File**: `data_preprocess.py`  

Sentinel-2 images in .SAFE format containing .jp2 (JPEG 2000) files undergo preprocessing for compatibility with the feature extraction and matching pipeline:  
- **Resampling**: Standardizes pixel sizes by adjusting spatial resolution.  
- **Normalization**: Scales pixel intensity values for uniformity across datasets.  
- **Resizing**: Reshapes images to a fixed resolution (224x224) for efficiency.  
- **Format Conversion**: Converts .jp2 files to .jpeg for streamlined processing.  

---

#### 2. **Feature Extraction with SuperPoint**  
- **File**: `SuperPoint.py`  

SuperPoint, a deep learning-based model, detects keypoints and extracts descriptors from images, ensuring robustness to variations in:  
- Scale  
- Rotation  
- Illumination  

These descriptors are crucial for accurate feature matching.  

---

#### 3. **Keypoint Matching with SuperGlue**  
- **File**: `SuperGlue_architecture.py`  

SuperGlue, a graph neural network (GNN), matches keypoints using:  
- Attention mechanisms  
- Contextual information  

This ensures accurate correspondences, even under significant seasonal, temporal, or atmospheric variations.  

---

### How to Use  

#### Step 1: Clone the Repository  
```bash
git clone https://github.com/Kimiko12/Sentinel2Image.git
```  

#### Step 2: Navigate to the Repository Folder  
```bash
cd Sentinel2Image
```  

#### Step 3: Install Dependencies  
Install the required Python packages:  
```bash
pip install -r requirements.txt
```  

#### Step 4: Preprocess Images  
Run the preprocessing script to convert .jp2 files to raster .jpeg images:  
```bash
python data_preprocess.py
```  

#### Step 5: Feature Extraction with SuperPoint  
Generate keypoints and descriptors from preprocessed images:  
```bash
python SuperPoint.py
```  

#### Step 6: Keypoint Matching with SuperGlue  
Match keypoints by running:  
```bash
python inference.py
```  

---

### Future Improvements  

1. **Multi-Spectral Image Support**  
   - Extend support for Sentinel-2's multi-spectral data, including Red, Green, Blue, and Near-Infrared bands, for enhanced feature extraction and matching.  

2. **Enhanced Matching Accuracy**  
   - Fine-tune SuperPoint and SuperGlue models using domain-specific datasets to improve accuracy in challenging conditions like cloud cover or partial obstructions.  

3. **Geospatial Metadata Integration**  
   - Use geospatial metadata (coordinates, time, and sensor details) to reduce false matches and enable context-aware comparisons.  

4. **Higher Image Resolution**  
   - Support higher-resolution images for detailed analysis and detection of smaller features, enabling more precise keypoint matching.  

---

### Results  

Below is an example of matched images:  
![Matched Images Result](https://github.com/user-attachments/assets/35a0ac40-d63d-4026-8a0b-06ec68ff7681)  
