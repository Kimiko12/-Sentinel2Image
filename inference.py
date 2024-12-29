import os
import logging
import torch
import numpy as np
from PIL import Image, ImageDraw

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

    def match_and_visualize(self, descriptor_path_1: str, descriptor_path_2: str, image_path_1: str, image_path_2: str):
        # Load descriptors and keypoints for both images
        data_1 = np.load(descriptor_path_1)
        kpts0, desc0, scores0 = data_1['keypoints'], data_1['descriptors'], data_1['scores']

        data_2 = np.load(descriptor_path_2)
        kpts1, desc1, scores1 = data_2['keypoints'], data_2['descriptors'], data_2['scores']

        # Match keypoints
        matches0 = self._match_keypoints(kpts0, desc0, scores0, kpts1, desc1, scores1)

        # Visualize results
        self.paint_keypoints(image_path_1, kpts0, image_path_2, kpts1, matches0)

    def paint_keypoints(self, image_path_1: str, keypoints1: np.ndarray, image_path_2: str, keypoints2: np.ndarray, matches0: np.ndarray):
        image1 = Image.open(image_path_1).convert("RGB")
        image2 = Image.open(image_path_2).convert("RGB")

        width1, height1 = image1.size
        width2, height2 = image2.size

        keypoints1_scaled = keypoints1 * [width1, height1]
        keypoints2_scaled = keypoints2 * [width2, height2]

        # Create a canvas to place images side by side
        canvas_width = width1 + width2
        canvas_height = max(height1, height2)
        canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
        canvas.paste(image1, (0, 0))
        canvas.paste(image2, (width1, 0))

        draw = ImageDraw.Draw(canvas)

        # Draw all keypoints
        for x, y in keypoints1_scaled:
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill="blue", outline="blue")
        for x, y in keypoints2_scaled:
            draw.ellipse([x + width1 - 3, y - 3, x + width1 + 3, y + 3], fill="red", outline="red")

        # Draw matching lines
        for idx, match in enumerate(matches0):
            if match != -1:
                x1, y1 = keypoints1_scaled[idx]
                x2, y2 = keypoints2_scaled[match]
                draw.line([(x1, y1), (x2 + width1, y2)], fill="green", width=1)

        result_path = "matched_images_result.jpg"
        canvas.save(result_path)
        logging.info(f"Image with keypoints and matches saved to {result_path}")

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

if __name__ == '__main__':
    # Path to localy saved model weights
    model_weights_path = '/home/nikolay/test_task_quantum/Sentinel_2_image/SuperGlue_Weights/superglue_indoor.pth'

    # Create instance of class
    matcher = ImageMatching(model_weights_path=model_weights_path, device=device)

    # Provide 2 paths with descriptors 
    descriptor_path_1 = '/home/nikolay/test_task_quantum/Sentinel_2_image/data_superpoints/T36UXA_20180726T084009_B02.npz'
    descriptor_path_2 = '/home/nikolay/test_task_quantum/Sentinel_2_image/data_superpoints/T36UXA_20180726T084009_B08.npz'

    # You can replace this line with actual paths to images 
    target_image_path, similar_image_path = matcher.find_image_paths(descriptor_path_1, descriptor_path_2)

    # Run match_and_visualize function for visualize results of matching 
    matcher.match_and_visualize(descriptor_path_1, descriptor_path_2, target_image_path, similar_image_path)
