import os
import cv2
import glob
import logging
import numpy as np
import rasterio
from rasterio.enums import Resampling
from skimage.transform import resize

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s'
)

class SentinelDataPreprocess:
    def __init__(self, input_data_path: str, path_to_processed_data: str, target_size: tuple = (224, 224)):

        self.logger = logging.getLogger(self.__class__.__name__) 
        self.path_to_raw_data = input_data_path
        self.path_to_processed_data = path_to_processed_data
        self.taget_size = target_size

        if not os.path.exists(self.path_to_processed_data):
            self.logger.warning(f'Output directory {self.path_to_processed_data} does not exist. Creating it...')
            os.makedirs(self.path_to_processed_data, exist_ok=True)
        
    def preprocess(self) -> None:
        folders = [os.path.join(self.path_to_raw_data, f) for f in os.listdir(self.path_to_raw_data) if os.path.isdir(os.path.join(self.path_to_raw_data, f))]

        if not folders:
            self.logger.error(f'No folders found in {self.path_to_raw_data}')
            return 
        
        self.logger.info(f'Found {len(folders)} folders in {self.path_to_raw_data}')

        for folder in folders:
            safe_folders = glob.glob(os.path.join(folder, "*.SAFE"))
            if not safe_folders:
                self.logger.warning(f'No .SAFE folder found in {folder}. Skipping...')
                continue

            for safe_folder in safe_folders:
                granule_path = os.path.join(safe_folder, 'GRANULE')
                if not granule_path:
                    self.logger.warning(f'No GRANULE folder found in {safe_folder}. Skipping...')
                    continue

                granule_folders = os.path.join(granule_path, '*')
                if not granule_folders:
                    self.logger.warning(f'No granules found in {granule_folders}. Skipping...')
                    continue

                for granule_folder in granule_folders:
                    path_to_image_data = os.path.join(granule_folder, 'IMG_DATA')
                    if not path_to_image_data:
                        self.logger.warning(f'No IMG_DATA folder found in {granule_folder}. Skipping...')
                        continue

                    output_folder = os.path.join(self.path_to_processed_data, os.path.basename(granule_folder))
                    os.makedirs(output_folder, exist_ok=True)

                    self._process_jp2_files(path_to_image_data, output_folder)
    
    def _process_jp2_files(self, path_to_image_data: str, output_folder: str) -> None:
        jp2_files = glob.glob(os.path.join(path_to_image_data, '*.jp2'))
        if not jp2_files:
            self.logger.warning(f'No .jp2 files found in {path_to_image_data}. Skipping...')
            return 
        
        for jp2_file in jp2_files:
            try:
                process_image = self._process_jp2_image(jp2_file)
                image_path = os.path.join(output_folder, os.path.basename(jp2_file).replace('.jp2', '.jpeg'))
                cv2.imwrite(image_path, process_image)
            except Exception as e:
                self.logger.exception(f'Error processing {jp2_file}: {str(e)}')

    def _process_jp2_image(self, jp2_file: str) -> np.ndarray:
        with rasterio.open(jp2_file) as src:
            image = src.read(1, resampling=Resampling.bilinear)

            if image is None or image.size == 0:
                self.logger.exception(f'Image cannot be read: {jp2_file}')
                return
            
            image = image.astype(np.float32)
            resize_image = resize(image, self.target_size, anti_aliasing=True)

            max_value = np.max(resize_image)
            if max_value == 0:
                normalized_image = resize_image
            else:
                normalized_image = resize_image / max_value

            image_uint8 = (normalized_image * 255).astype(np.uint8)
            return image_uint8
        
if __name__ == '__main__':
    input_data_path = '/mnt/c/Users/kolak/Downloads/archive'
    output_data_path = '/home/nikolay/test_task_quantum/Sentinel_2_image/data'

    data_preparation = SentinelDataPreprocess(
        input_data_path=input_data_path,
        path_to_processed_data=output_data_path,
        target_size=(224, 224),
    )

    data_preparation.preprocess()
