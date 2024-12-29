import os
import glob
import logging
import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from test_task_quantum.Sentinel_2_image.SuperGlue_architecture import SuperGlue

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ImageMatching:
    def __init__(self, model_weights_path: str, device: str = 'cuda'):
        self.device = device
        self.model = self._load_model(model_weights_path)

    def _load_model(self, model_weights_path: str):
        model = SuperGlue.from_local_weights(
            path_to_weights=model_weights_path,
            config={
                'descriptor_dim': 256,
                'weights': 'indoor',
                'sinkborn_iterations': 50,
                'match_threshold': 0.05 
            },
            device=self.device
        ).to(self.device)
        model.eval()
        return model
    
    def _match_keypoints(self, kpts0: np.ndarray, desc0: np.ndarray, scores0: np.ndarray, kpts1: np.ndarray, desc1: np.ndarray, scores1: np.ndarray) -> np.ndarray:

        kpts0_tensor = torch.from_numpy(kpts0)[None].float().to(self.device)
        desc0_tensor = torch.from_numpy(desc0).transpose(0, 1)[None].float().to(self.device)
        scores0_tensor = torch.from_numpy(scores0)[None].float().to(self.device)

        kpts1_tensor = torch.from_numpy(kpts1)[None].float().to(self.device)
        desc1_tensor = torch.from_numpy(desc1).transpose(0, 1)[None].float().to(self.device)
        scores1_tensor = torch.from_numpy(scores1)[None].float().to(self.device)

        image0_tensor = torch.zeros((1, 1, 224, 224), device=self.device)
        image1_tensor = torch.zeros((1, 1, 224, 224), device=self.device)

        inputs = {
            'descriptors0': desc0_tensor,
            'descriptors1': desc1_tensor,
            'keypoints0': kpts0_tensor,
            'keypoints1': kpts1_tensor,
            'scores0': scores0_tensor,
            'scores1': scores1_tensor,
            'image0': image0_tensor,
            'image1': image1_tensor
        }

        with torch.no_grad():
            outputs = self.model(inputs)

        matches0 = outputs['matches0'][0].cpu().numpy()
        return matches0    
    def match_with_the_most_compatible_image(self, target_descriptor_path: str, folder_with_descriptors: str):
        data_target = np.load(target_descriptor_path)

        kpts0 = data_target['keypoints']
        desc0 = data_target['descriptors']
        scores0 = data_target['scores']

        best_match_path = None
        best_match_score = float('-inf')
        best_keypoints = None
        best_matches = None

        for file_path in glob.glob(os.path.join(folder_with_descriptors, '*.npz')):
            if file_path == target_descriptor_path:
                continue

            data = np.load(file_path)
            if not all(key in data for key in ['keypoints', 'descriptors', 'scores']):
                continue 

            kpts1 = data['keypoints']
            desc1 = data['descriptors']
            scores1 = data['scores']

            matches0 = self._match_keypoints(kpts0, desc0, scores0, kpts1, desc1, scores1)
            valid_match = np.argwhere(matches0 != -1).flatten()
            match_score = len(valid_match)

            logging.info(f'Processing file: {file_path}')
            logging.info(f'Number of matches: {match_score}')

            if best_match_score < match_score:
                best_match_score = match_score
                best_match_path = file_path
                best_keypoints = kpts1
                best_matches = matches0

        if best_match_path is None:
            raise ValueError('No compatible images found')
        
        target_image_path, similar_image_path = self.find_image_paths(target_descriptor_path, best_match_path)
        self.paint_keypoints(target_image_path, kpts0, similar_image_path, best_keypoints, best_matches)

        logging.info(f'The most compatible image: {best_match_path} with score {best_match_score}')
        return best_match_path

    def find_image_paths(self, target_descriptor_path: str, best_match_path: str):
        main_folder = os.path.dirname(target_descriptor_path)
        main_folder = main_folder.replace('data_superpoints', 'data')
        target_image_name = os.path.basename(target_descriptor_path).replace('.npz', '.jpg')
        similar_image_name = os.path.basename(best_match_path).replace('.npz', '.jpg')

        target_image_path, similar_image_path = None, None

        for folder in os.listdir(main_folder):
            folder_path = os.path.join(main_folder, folder)
            for image_name in os.listdir(folder_path):
                if target_image_name == image_name:
                    target_image_path = os.path.join(folder_path, image_name)
                if similar_image_name == image_name:
                    similar_image_path = os.path.join(folder_path, image_name)

        if target_image_path is None or similar_image_path is None:
            raise FileNotFoundError("One or both image paths could not be located.")

        return target_image_path, similar_image_path

    def paint_keypoints(self, target_image_path: str, target_keypoints: np.ndarray, similar_image_path: str, similar_keypoints: np.ndarray, matches0: np.ndarray):

        target_image = Image.open(target_image_path).convert("RGB")
        similar_image = Image.open(similar_image_path).convert("RGB")
    
        target_width, target_height = target_image.size
        similar_width, similar_height = similar_image.size
        
        target_keypoints_scaled = target_keypoints * [target_width, target_height]
        similar_keypoints_scaled = similar_keypoints * [similar_width, similar_height]
        
        canvas_width = target_width + similar_width
        canvas_height = max(target_height, similar_height)
        canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
        
        # Размещаем изображения рядом
        canvas.paste(target_image, (0, 0))
        canvas.paste(similar_image, (target_width, 0))
        
        draw = ImageDraw.Draw(canvas)
        
        for x, y in target_keypoints_scaled:
            draw.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill="blue", outline="blue")
        
        for x, y in similar_keypoints_scaled:
            draw.ellipse([x + target_width - 1.5, y - 1.5, x + target_width + 1.5, y + 1.5], fill="red", outline="red")
        
        for idx, match in enumerate(matches0):
            if match != -1:
                x0, y0 = target_keypoints_scaled[idx]
                x1, y1 = similar_keypoints_scaled[match]
                draw.line([(x0, y0), (x1 + target_width, y1)], fill="green", width=1)
        
        # Сохранение результата
        result_path = "matched_images_result_scaled.jpg"
        canvas.save(result_path)
        logging.info(f"Image with keypoints and matches saved to {result_path}")


# Take as input image and find in folder the most compatible image with the highest number of matches

if __name__ == '__main__':
    model_weights_path = '/home/nikolay/test_task_quantum/Sentinel_2_image/SuperGlue_Weights/superglue_indoor.pth'

    matcher = ImageMatching(model_weights_path=model_weights_path, device=device)

    target_descriptor_path = '/home/nikolay/test_task_quantum/Sentinel_2_image/data_superpoints/T36UXA_20180726T084009_B02.npz'
    folder_with_descriptors = '/home/nikolay/test_task_quantum/Sentinel_2_image/data_superpoints'

    best_match = matcher.match_with_the_most_compatible_image(
        target_descriptor_path=target_descriptor_path,
        folder_with_descriptors=folder_with_descriptors
    )

    print(f'The most compatible image is {best_match}')
