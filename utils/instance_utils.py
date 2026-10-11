import time
import numpy as np
import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.plotting import colors


class InstanceSegmentor:
    """Instance segmentation and visualization class"""

    def __init__(self,
                 model_name='model/yolo11m-seg.pt',
                 verbose=False,
                 conf_threshold=0.4,
                 use_custom_colors=True):
        """Initialize segmentor"""
        self.model = YOLO(model_name)
        self.verbose = verbose
        self.conf_threshold = conf_threshold
        self.use_custom_colors = use_custom_colors

        # Custom color palette
        self.custom_palette = [
            [255, 99, 71],
            [60, 179, 113],
            [255, 165, 0],
            [30, 144, 255],
            [255, 20, 147],
            [50, 205, 50],
            [255, 215, 0],
            [138, 43, 226],
            [220, 20, 60],
            [0, 191, 255],
            [255, 105, 180],
            [32, 178, 170],
            [255, 140, 0],
            [106, 90, 205],
            [255, 69, 0],
            [0, 206, 209],
            [255, 182, 193],
            [127, 255, 0],
            [255, 160, 122],
            [64, 224, 208],
            [255, 218, 185],
            [173, 216, 230],
            [240, 230, 140],
            [221, 160, 221],
            [176, 196, 222],
            [255, 228, 181],
            [152, 251, 152],
            [255, 192, 203],
            [135, 206, 235],
            [255, 239, 213],
        ]

        self.coco_names = self.model.names

        # Target classes for SLAM
        self.target_classes = {
            'person', 'chair', 'sofa', 'dining table', 'tv', 'laptop', 'cell phone', 'book',
            'keyboard', 'mouse'
        }

        self.dynamic_classes = {'person'}
        self.static_classes = self.target_classes - self.dynamic_classes

        # Cache last segmentation results
        self.last_masks = None
        self.last_boxes = None
        self.last_classes = None
        self.last_scores = None
        self.last_original_image = None
        self.last_reference_input = None

    def get_class_name(self, class_id):
        """Get class name by ID"""
        return self.coco_names.get(class_id, 'unknown')

    def get_class_id(self, class_name):
        """Get class ID by name"""
        for id, name in self.coco_names.items():
            if name == class_name:
                return id
        return -1

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
        """Main segmentation interface"""
        original_image = self._convert_to_numpy_for_inference(image)

        if self.verbose:
            print(f"Running instance segmentation, image size: {original_image.shape}")

        start_time = time.time()
        results = self.model(original_image, verbose=False)
        inference_time = time.time() - start_time

        result = results[0]
        h, w = original_image.shape[:2]
        instance_mask = np.zeros((h, w), dtype=np.int32)

        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            scores = result.boxes.conf.cpu().numpy()

            # Log all detected objects
            all_detected = []
            for i, cls in enumerate(classes):
                if scores[i] > self.conf_threshold:
                    class_name = self.get_class_name(cls)
                    pixel_count = np.sum(cv2.resize(masks[i], (w, h)) > 0.5)
                    all_detected.append(f"{class_name}({scores[i]:.3f},{pixel_count})")

            # Filter target classes
            valid_indices = []
            target_detected = []
            for i, cls in enumerate(classes):
                class_name = self.get_class_name(cls)
                if class_name in self.target_classes and scores[i] > self.conf_threshold:
                    valid_indices.append(i)
                    pixel_count = np.sum(cv2.resize(masks[i], (w, h)) > 0.5)
                    target_detected.append(f"{class_name}({scores[i]:.3f},{pixel_count})")

            # Generate instance mask
            instance_id = 1
            for i in valid_indices:
                mask = masks[i]
                mask = cv2.resize(mask, (w, h)) > 0.5
                instance_mask[mask] = instance_id
                instance_id += 1

            self.last_masks = masks[valid_indices] if valid_indices else None
            self.last_boxes = boxes[valid_indices] if valid_indices else None
            self.last_classes = classes[valid_indices] if valid_indices else None
            self.last_scores = scores[valid_indices] if valid_indices else None
        else:
            all_detected = []
            target_detected = []
            self.last_masks = None
            self.last_boxes = None
            self.last_classes = None
            self.last_scores = None

        self.last_original_image = original_image
        self.last_reference_input = image

        if self.verbose:
            print(f"Segmentation completed in {inference_time:.3f}s")
            print(f"All detected: {', '.join(all_detected) if all_detected else 'None'}")
            print(f"Target objects: {', '.join(target_detected) if target_detected else 'None'}")

        return self._convert_output_format(instance_mask, image)

    def get_dynamic_mask(self):
        """Get binary mask of dynamic objects"""
        if self.last_masks is None or self.last_classes is None:
            return None

        h, w = self.last_original_image.shape[:2]
        mask = np.zeros((h, w), dtype=bool)

        for i, cls in enumerate(self.last_classes):
            class_name = self.get_class_name(cls)
            if class_name in self.dynamic_classes:
                instance_mask = cv2.resize(self.last_masks[i], (w, h)) > 0.5
                mask[instance_mask] = True

        return self._convert_output_format(mask, self.last_reference_input)

    def get_static_mask(self):
        """Get binary mask of static objects"""
        if self.last_masks is None or self.last_classes is None:
            return None

        h, w = self.last_original_image.shape[:2]
        mask = np.zeros((h, w), dtype=bool)

        for i, cls in enumerate(self.last_classes):
            class_name = self.get_class_name(cls)
            if class_name in self.static_classes:
                instance_mask = cv2.resize(self.last_masks[i], (w, h)) > 0.5
                mask[instance_mask] = True

        return self._convert_output_format(mask, self.last_reference_input)

    def get_class_mask(self, class_names):
        """Get binary mask of specified classes"""
        if self.last_masks is None or self.last_classes is None:
            return None

        if isinstance(class_names, str):
            class_names = [class_names]

        h, w = self.last_original_image.shape[:2]
        mask = np.zeros((h, w), dtype=bool)

        for i, cls in enumerate(self.last_classes):
            class_name = self.get_class_name(cls)
            if class_name in class_names:
                instance_mask = cv2.resize(self.last_masks[i], (w, h)) > 0.5
                mask[instance_mask] = True

        return self._convert_output_format(mask, self.last_reference_input)

    def get_semantic_image(self):
        """Get colored visualization of instance segmentation"""
        if self.last_masks is None or self.last_classes is None:
            return None

        h, w = self.last_original_image.shape[:2]
        color_image = np.zeros((h, w, 3), dtype=np.uint8)

        for i, cls in enumerate(self.last_classes):
            class_name = self.get_class_name(cls)
            if class_name in self.target_classes:
                mask = cv2.resize(self.last_masks[i], (w, h)) > 0.5
                color = self._get_color(i, cls)
                color_image[mask] = color

        if isinstance(self.last_reference_input, torch.Tensor):
            result = torch.from_numpy(color_image).to(self.last_reference_input.device)
            if self.last_reference_input.dim() == 3 and self.last_reference_input.shape[0] == 3:
                result = result.permute(2, 0, 1)
            if self.last_reference_input.max() <= 1.0:
                result = result.float() / 255.0
            return result

        return color_image

    def get_overlay_image(self, alpha=0.6):
        """Get overlay of original image and instance segmentation"""
        if self.last_masks is None or self.last_original_image is None:
            return None

        h, w = self.last_original_image.shape[:2]
        overlay = self.last_original_image.copy().astype(np.float32)

        for i, cls in enumerate(self.last_classes):
            class_name = self.get_class_name(cls)
            if class_name in self.target_classes:
                mask = cv2.resize(self.last_masks[i], (w, h)) > 0.5
                color = self._get_color(i, cls)

                for c in range(3):
                    overlay[mask, c] = overlay[mask, c] * (1 - alpha) + color[c] * alpha

        overlay = overlay.astype(np.uint8)

        if isinstance(self.last_reference_input, torch.Tensor):
            result = torch.from_numpy(overlay).to(self.last_reference_input.device)
            if self.last_reference_input.dim() == 3 and self.last_reference_input.shape[0] == 3:
                result = result.permute(2, 0, 1)
            if self.last_reference_input.max() <= 1.0:
                result = result.float() / 255.0
            return result

        return overlay

    def get_instance_info(self):
        """Get information of detected instances"""
        if self.last_classes is None:
            return []

        instances = []
        for i, cls in enumerate(self.last_classes):
            class_name = self.get_class_name(cls)
            instances.append({
                'class_id': cls,
                'class_name': class_name,
                'score': self.last_scores[i],
                'bbox': self.last_boxes[i],
                'is_dynamic': class_name in self.dynamic_classes
            })

        return instances

    def print_available_classes(self):
        """Print all available COCO classes"""
        print("Available COCO classes:")
        for id, name in self.coco_names.items():
            print(f"  {id}: {name}")

    def add_target_class(self, class_name, is_dynamic=False):
        """Add a class to target classes"""
        if class_name in self.coco_names.values():
            self.target_classes.add(class_name)
            if is_dynamic:
                self.dynamic_classes.add(class_name)
            else:
                self.static_classes.add(class_name)
            print(f"Added {class_name} as {'dynamic' if is_dynamic else 'static'} target class")
        else:
            print(f"Class {class_name} not found in COCO dataset")

    def remove_target_class(self, class_name):
        """Remove a class from target classes"""
        self.target_classes.discard(class_name)
        self.dynamic_classes.discard(class_name)
        self.static_classes.discard(class_name)
        print(f"Removed {class_name} from target classes")

    def _get_color(self, index, class_id=None):
        """Get color for visualization"""
        if self.use_custom_colors:
            color_index = class_id if class_id is not None else index
            return self.custom_palette[color_index % len(self.custom_palette)]
        else:
            color = colors(class_id if class_id is not None else index, True)
            return [int(c) for c in color]


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    segmentor = InstanceSegmentor('model/yolo11m-seg.pt', verbose=True, conf_threshold=0.4)

    image_path = "test.png"
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        exit(1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Image shape: {image.shape}")

    instance_mask = segmentor.segment_image(image)
    print(f"Instance IDs: {np.unique(instance_mask)}")

    dynamic_mask = segmentor.get_dynamic_mask()
    static_mask = segmentor.get_static_mask()
    chair_mask = segmentor.get_class_mask('chair')

    instance_image = segmentor.get_semantic_image()
    overlay_image = segmentor.get_overlay_image()

    instances = segmentor.get_instance_info()
    for i, inst in enumerate(instances):
        print(f"{i+1}. {inst['class_name']} (score: {inst['score']:.3f})")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    if instance_image is not None:
        axes[0, 1].imshow(instance_image)
        axes[0, 1].set_title('Instances')
        axes[0, 1].axis('off')

    if overlay_image is not None:
        axes[1, 0].imshow(overlay_image)
        axes[1, 0].set_title('Overlay')
        axes[1, 0].axis('off')

    if dynamic_mask is not None:
        axes[1, 1].imshow(dynamic_mask, cmap='gray')
        axes[1, 1].set_title('Dynamic')
        axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('results.png', dpi=150)
    print("Results saved to results.png")

    torch_image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    torch_mask = segmentor.segment_image(torch_image)
    print(f"Torch test - mask shape: {torch_mask.shape}")
