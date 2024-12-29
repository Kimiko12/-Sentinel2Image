
import os 
import glob 
import torch 
import logging 
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Tuple, Dict
from torchvision.transforms import Compose, ToTensor, Normalize
from transformers import AutoImageProcessor, SuperPointForKeypointDetection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SuperPoint:
    '''
    Goal of this class to extract keypoints and descriptions from the images
    '''
    def __init__(self, model_name_or_path: str = 'magic-leep-community/superpoint', device: str = 'cuda'):
        self.model_name_or_path = model_name_or_path
        self.device = device 
        self.model, self.processor = self._load_model()

    def _load_model(self):
        model = SuperPointForKeypointDetection.from_pretrained(
            self.model_name_or_path
        ).to(self.device)

        # Processor object take model name as argument and make corresponding image processor and return correct processed image
        processor = AutoImageProcessor.from_pretrained(
            self.model_name_or_path
        )
        model.eval()

        return model, processor
    
    def _process_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not os.path.exists(image_path):
            logging.error(f'Image path {image_path} does not exist')
            return np.array([]), np.array([]), np.array([])
        
        image = Image.open(image_path)
        # Ensure that the image has 3 channels
        img_array = np.array(image)

        if len(img_array.shape) == 2:
            # shape: (H, W)
            # Expand to (H, W, 1), then repeat
            img_array = np.expand_dims(img_array, axis=-1)
            img_array = np.repeat(img_array, 3, axis=-1)
            # Optionally convert to uint8 if needed
            if img_array.dtype != np.uint8:
                # e.g., scale if needed
                img_array = img_array.astype(np.uint8)
            image = Image.fromarray(img_array)

        # Now pass to processor
        inputs = self.processor(
            image,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        keypoints = outputs.keypoints[0].cpu().numpy()
        descriptors = outputs.descriptors[0].cpu().numpy()
        scores = outputs.scores[0].cpu().numpy()

        return keypoints, descriptors, scores

    
    def process_dataset(self, dataset_path: str, output_path: str) -> None:

        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        if not os.path.exists(dataset_path):
            logging.error(f'Dataset path {dataset_path} does not exist')
            return
        
        for folder_with_images in os.listdir(dataset_path):
            folder_path = os.path.join(dataset_path, folder_with_images)
            if not os.path.isdir(folder_path):
                continue

            image_paths = glob.glob(os.path.join(folder_path, '*.jpg'))
            if not image_paths:
                logging.warning(f'No images found in {folder_path}')
                continue

            for image_path in image_paths:
                try:
                    keypoints, descriptors, scores = self._process_image(image_path)

                    output_file = os.path.join(output_path, os.path.basename(image_path).replace('.jpg', '.npz'))
                    np.savez(output_file, keypoints=keypoints, descriptors=descriptors, scores=scores)
                
                except Exception as e:
                    logging.error(f'Error processing image {image_path}: {e}')

    def visualize_keypoints(self, image_path: str, output_path: str) -> None:
        if not os.path.exists(image_path):
            logging.error(f'Image path {image_path} does not exist')
            return

        image = Image.open(image_path).convert("RGB") 
        key_points, _, _ = self._process_image(image_path)

        if key_points.shape[0] == 0:
            logging.warning(f'No keypoints found for image {image_path}')
            return

        logging.info(f"Detected {len(key_points)} keypoints for image {image_path}")

        width, height = image.size
        key_points[:, 0] *= width  
        key_points[:, 1] *= height

        draw = ImageDraw.Draw(image)
        for key_point in key_points:
            x, y = key_point

            draw.ellipse((x - 1, y - 1, x + 1, y + 1), outline="green", fill="green", width=2)

        if not os.path.splitext(output_path)[1]:
            output_path += ".jpg"  

        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        try:
            image.save(output_path)
            logging.info(f"Image with keypoints saved to {output_path}")
        except Exception as e:
            logging.error(f"Failed to save image: {e}")


if __name__ == '__main__':
    dataset_path = '/home/nikolay/test_task_quantum/Sentinel_2_image/data'
    output_path = '/home/nikolay/test_task_quantum/Sentinel_2_image/data_superpoints'
    model_name_or_path = 'magic-leap-community/superpoint'

    image_path="/home/nikolay/test_task_quantum/Sentinel_2_image/data/L1C_T36UXA_A007240_20180726T084437/T36UXA_20180726T084009_B01.jpg",
    output_path="/home/nikolay/test_task_quantum/Sentinel_2_image/output/T36UXA_20180726T084009_B01"
    
    extractor = SuperPoint(
        model_name_or_path=model_name_or_path,
        device=device
    )

    extractor.process_dataset(
        dataset_path=dataset_path, 
        output_path=output_path
    )

    extractor.visualize_keypoints(
    image_path="/home/nikolay/test_task_quantum/Sentinel_2_image/data/L1C_T36UXA_A007240_20180726T084437/T36UXA_20180726T084009_B01.jpg",
    output_path="/home/nikolay/test_task_quantum/Sentinel_2_image/output/T36UXA_20180726T084009_B01"
    )

