# Copyright (c) Opendatalab. All rights reserved.

from docvortex.image import gray_image, resize_image
from PIL import Image, ImageOps
from transformers.image_processing_utils import BaseImageProcessor
import numpy as np
import torch
from torchvision.transforms.functional import resize


class UnimerSwinImageProcessor(BaseImageProcessor):
    def __init__(
            self,
            image_size = (192, 672),
        ):
        self.input_size = [int(_) for _ in image_size]
        assert len(self.input_size) == 2

    def __call__(self, item):
        image = self.prepare_input(item)
        return self.to_normalized_gray_tensor(image)

    @staticmethod
    def to_normalized_gray_tensor(image: np.ndarray) -> torch.Tensor:
        """将图像确定性转灰度、按 UniMERNet 参数归一化，并转为单通道 tensor。"""
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3 and image.shape[2] == 1:
            gray = image[:, :, 0]
        elif image.ndim == 3 and image.shape[2] == 3:

            gray = gray_image(image, color_order='rgb')
        else:
            raise ValueError(f"Unsupported image shape for UnimerSwinImageProcessor: {image.shape}")

        normalized = (gray.astype(np.float32) - 0.7931 * 255.0) / (0.1738 * 255.0)
        return torch.from_numpy(normalized[None, :, :])

    @staticmethod
    def crop_margin(img: Image.Image) -> Image.Image:
        data = np.array(img.convert("L"))
        data = data.astype(np.uint8)
        max_val = data.max()
        min_val = data.min()
        if max_val == min_val:
            return img
        data = (data - min_val) / (max_val - min_val) * 255
        gray = 255 * (data < 200).astype(np.uint8)

        ys, xs = np.nonzero(gray)
        if xs.size == 0:
            return img.crop((0, 0, 0, 0))
        return img.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))

    @staticmethod
    def crop_margin_numpy(img: np.ndarray) -> np.ndarray:
        """Crop margins of image using NumPy operations"""
        # Convert to grayscale if it's a color image
        if len(img.shape) == 3 and img.shape[2] == 3:

            gray = gray_image(img, color_order='rgb')
        else:
            gray = img.copy()

        # Normalize and threshold
        if gray.max() == gray.min():
            return img

        normalized = (((gray - gray.min()) / (gray.max() - gray.min())) * 255).astype(np.uint8)
        binary = 255 * (normalized < 200).astype(np.uint8)

        # Find bounding box
        mask = binary[:, :, 0] if binary.ndim == 3 and binary.shape[2] == 1 else binary
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return img[0:0, 0:0]
        return img[int(ys.min()):int(ys.max()) + 1, int(xs.min()):int(xs.max()) + 1]

    def prepare_input(self, img, random_padding: bool = False):
        """
        Convert PIL Image or numpy array to properly sized and padded image after:
            - crop margins
            - resize while maintaining aspect ratio
            - pad to target size
        """
        if img is None:
            return None

        # Handle numpy array
        elif isinstance(img, np.ndarray):
            try:
                img = self.crop_margin_numpy(img)
            except Exception:
                # might throw an error for broken files
                return None

            if img.shape[0] == 0 or img.shape[1] == 0:
                return None

            # Get current dimensions
            h, w = img.shape[:2]
            target_h, target_w = self.input_size

            # Calculate scale to preserve aspect ratio (equivalent to resize + thumbnail)
            scale = min(target_h / h, target_w / w)

            # Calculate new dimensions
            new_h, new_w = int(h * scale), int(w * scale)

            # Resize the image while preserving aspect ratio

            resized_img = resize_image(img, (new_w, new_h))

            # Calculate padding values using the existing method
            delta_width = target_w - new_w
            delta_height = target_h - new_h

            pad_width, pad_height = self._get_padding_values(new_w, new_h, random_padding)

            # 常量零填充无需 OpenCV，保持缩放结果的维度和位深。
            padding = [(pad_height, delta_height - pad_height), (pad_width, delta_width - pad_width)]
            if resized_img.ndim == 3:
                padding.append((0, 0))
            padded_img = np.pad(resized_img, padding, mode="constant")

            return padded_img

        # Handle PIL Image
        elif isinstance(img, Image.Image):
            try:
                img = self.crop_margin(img.convert("RGB"))
            except OSError:
                # might throw an error for broken files
                return None

            if img.height == 0 or img.width == 0:
                return None

            # Resize while preserving aspect ratio
            img = resize(img, min(self.input_size))
            img.thumbnail((self.input_size[1], self.input_size[0]))
            new_w, new_h = img.width, img.height

            # Calculate and apply padding
            padding = self._calculate_padding(new_w, new_h, random_padding)
            return np.array(ImageOps.expand(img, padding))

        else:
            return None

    def _calculate_padding(self, new_w, new_h, random_padding):
        """Calculate padding values for PIL images"""
        delta_width = self.input_size[1] - new_w
        delta_height = self.input_size[0] - new_h

        pad_width, pad_height = self._get_padding_values(new_w, new_h, random_padding)

        return (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )

    def _get_padding_values(self, new_w, new_h, random_padding):
        """Get padding values based on image dimensions and padding strategy"""
        delta_width = self.input_size[1] - new_w
        delta_height = self.input_size[0] - new_h

        if random_padding:
            pad_width = np.random.randint(low=0, high=delta_width + 1)
            pad_height = np.random.randint(low=0, high=delta_height + 1)
        else:
            pad_width = delta_width // 2
            pad_height = delta_height // 2

        return pad_width, pad_height
