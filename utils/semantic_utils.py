import time
import numpy as np
import cv2
import torch
from PIL import Image
from mmseg.datasets import ADE20KDataset
from mmseg.apis import MMSegInferencer


class SemanticSegmentor:
    """Semantic segmentation and visualization class"""

    def __init__(self,
                 model_name='mask2former_swin-t_8xb2-160k_ade20k-512x512'):
        """Initialize segmentor"""
        self.inferencer = MMSegInferencer(
            model=model_name,
            classes=ADE20KDataset.METAINFO['classes'],
            palette=ADE20KDataset.METAINFO['palette'])

        # Class configurations
        self.seg_classes = {
            'table', 'chair', 'sofa', 'desk', 'lamp', 'computer', 'door',
            'stairs', 'box', 'bottle', 'book', 'monitor', 'ball'
        }
        self.dyna_classes = {'person'}

        # Class mappings
        self.classes = ADE20KDataset.METAINFO['classes']
        self.palette = np.array(ADE20KDataset.METAINFO['palette'])
        self.class_to_idx = {
            name: idx
            for idx, name in enumerate(self.classes)
        }
        self.idx_to_class = {
            idx: name
            for idx, name in enumerate(self.classes)
        }

        # Cache for last segmentation results
        self.last_predictions = None
        self.last_original_image = None
        self.last_reference_input = None

    def get_class_name(self, idx):
        """Get class name by index"""
        return self.idx_to_class.get(idx, 'unknown')

    def get_class_idx(self, class_name):
        """Get class index by name"""
        return self.class_to_idx.get(class_name, -1)

    def _convert_to_numpy_for_inference(self, image):
        """Convert input image to numpy format for inference"""
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] == 3:
                image = image.permute(1, 2, 0)
            image = image.cpu().numpy()
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        elif isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        return image

    def _convert_output_format(self, mask, reference_input):
        """Convert output format based on reference input"""
        if isinstance(reference_input, torch.Tensor):
            return torch.from_numpy(mask).to(reference_input.device)
        else:
            return mask

    def segment_image(self, image):
        """
        Main segmentation interface

        Args:
            image: Input image (numpy.ndarray or torch.Tensor)

        Returns:
            predictions: Segmentation results in same format as input
        """
        # Convert to numpy for inference
        original_image = self._convert_to_numpy_for_inference(image)

        print(
            f"Running semantic segmentation on image shape: {original_image.shape}"
        )

        # Run inference
        start_time = time.time()
        result = self.inferencer(original_image, show=False)
        inference_time = time.time() - start_time

        predictions = result['predictions']

        # Cache results
        self.last_predictions = predictions
        self.last_original_image = original_image
        self.last_reference_input = image

        # Log basic statistics
        unique_classes = np.unique(predictions)
        detected_classes = []

        for class_idx in unique_classes:
            class_name = self.get_class_name(class_idx)
            pixel_count = np.sum(predictions == class_idx)
            if class_name in self.seg_classes or class_name in self.dyna_classes:
                detected_classes.append(f"{class_name}({pixel_count})")

        print(f"Segmentation completed in {inference_time:.3f}s")
        print(
            f"Detected classes: {', '.join(detected_classes) if detected_classes else 'None'}"
        )

        # Convert to same format as input
        predictions = self._convert_output_format(predictions, image)

        return predictions

    def get_dynamic_mask(self):
        """Get dynamic object binary mask from last segmentation"""
        if self.last_predictions is None:
            return None

        mask = np.zeros(self.last_predictions.shape, dtype=bool)

        for class_name in self.dyna_classes:
            if class_name in self.class_to_idx:
                class_idx = self.class_to_idx[class_name]
                mask[self.last_predictions == class_idx] = True

        return self._convert_output_format(mask, self.last_reference_input)

    def get_static_mask(self):
        """Get static object binary mask from last segmentation"""
        if self.last_predictions is None:
            return None

        mask = np.zeros(self.last_predictions.shape, dtype=bool)

        for class_name in self.seg_classes:
            if class_name in self.class_to_idx:
                class_idx = self.class_to_idx[class_name]
                mask[self.last_predictions == class_idx] = True

        return self._convert_output_format(mask, self.last_reference_input)

    def get_class_mask(self, class_names):
        """
        Get binary mask for specified classes from last segmentation

        Args:
            class_names: Class name(s) - string or list of strings

        Returns:
            mask: Binary mask (0/1) for specified classes
        """
        if self.last_predictions is None:
            return None

        if isinstance(class_names, str):
            class_names = [class_names]

        mask = np.zeros(self.last_predictions.shape, dtype=bool)

        for class_name in class_names:
            if class_name in self.class_to_idx:
                class_idx = self.class_to_idx[class_name]
                mask[self.last_predictions == class_idx] = True

        return self._convert_output_format(mask, self.last_reference_input)

    def get_semantic_image(self):
        if self.last_predictions is None:
            return None

        h, w = self.last_predictions.shape
        color_image = np.zeros((h, w, 3), dtype=np.uint8)

        for class_name in self.seg_classes:
            if class_name in self.class_to_idx:
                class_idx = self.class_to_idx[class_name]
                mask = self.last_predictions == class_idx
                if np.any(mask):
                    color = self.palette[class_idx]
                    color_image[mask] = color

        if isinstance(self.last_reference_input, torch.Tensor):
            result = torch.from_numpy(color_image).to(
                self.last_reference_input.device)
            if self.last_reference_input.dim(
            ) == 3 and self.last_reference_input.shape[0] == 3:
                result = result.permute(2, 0, 1)
            if self.last_reference_input.max() <= 1.0:
                result = result.float() / 255.0
            return result

        return color_image

    def get_overlay_image(self, alpha=0.8):
        if self.last_predictions is None or self.last_original_image is None:
            return None

        h, w = self.last_predictions.shape
        visualization_mask = np.zeros((h, w, 3), dtype=np.uint8)
        alpha_mask = np.zeros((h, w), dtype=np.float32)

        for class_name in self.seg_classes:
            if class_name in self.class_to_idx:
                class_idx = self.class_to_idx[class_name]
                mask = self.last_predictions == class_idx
                if np.any(mask):
                    color = self.palette[class_idx]
                    visualization_mask[mask] = color
                    alpha_mask[mask] = alpha

        for class_name in self.dyna_classes:
            if class_name in self.class_to_idx:
                class_idx = self.class_to_idx[class_name]
                mask = self.last_predictions == class_idx
                if np.any(mask):
                    visualization_mask[mask] = [0, 0, 0]
                    alpha_mask[mask] = alpha

        result_image = self.last_original_image.copy().astype(np.float32)
        for i in range(3):
            result_image[:, :, i] = (self.last_original_image[:, :, i] *
                                     (1 - alpha_mask) +
                                     visualization_mask[:, :, i] * alpha_mask)

        result_image = result_image.astype(np.uint8)

        if isinstance(self.last_reference_input, torch.Tensor):
            result = torch.from_numpy(result_image).to(
                self.last_reference_input.device)
            if self.last_reference_input.dim(
            ) == 3 and self.last_reference_input.shape[0] == 3:
                result = result.permute(2, 0, 1)
            if self.last_reference_input.max() <= 1.0:
                result = result.float() / 255.0
            return result

        return result_image
