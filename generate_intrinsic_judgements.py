import os
import json
import random
import glob
import math
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from tqdm import tqdm
import argparse
import textwrap


INTRINSIC_MODALITIES = {
    "rgb": {"index": 0, "channels": 3},
    "albedo": {"index": 1, "channels": 3},
    "normal": {"index": 2, "channels": 3},
    "depth": {"index": 3, "channels": 1},
    "irradiance": {"index": 4, "channels": 3}
}

QUESTION_TEMPLATES = {
    "depth": [
        "Looking at this image, which colored point appears to be closer to the camera - red or green?"
    ],
    "normal": [
        "Which point has a surface more facing towards the camera, red or green?",
    ],
    "irradiance": [
        "Which point is more illuminated?"
    ],
    "albedo": [
        "Do the red and green points have the same base color?"
    ]
}


class RGBXIntrinsicJudgmentGenerator:
    """Generate training data for intrinsic image judgment using RGB images with painted points.

    The model takes a point pair annotated RGB image as input and learns to make relative judgments with
    respect to the intrinsic properties of the image (normals, depth, irradiance, albedo) via RGB visual cues.
    Ground truth: Generated from corresponding intrinsic image values

    Expected data directory layout (all standard 8-bit PNGs):
        data_dir/
          scene_001_rgb.png
          scene_001_depth.png
          scene_001_normal.png
          scene_001_albedo.png
          scene_001_irradiance.png   # optional
          scene_002_rgb.png
          ...
    """

    def __init__(self, image_size=(512, 512)):
        self.image_size = image_size

    # ------------------------------------------------------------------
    # Image processing helpers (from rgbx_data.py pipeline)
    # ------------------------------------------------------------------

    def depth_to_disparity(self, input, fov=85, baseline=1.0):
        """Convert depth to disparity."""
        focal = 0.5 * max(input.shape[0], input.shape[1]) * math.tan(math.radians(90.0) - (0.5 * math.radians(fov)))
        return (focal * baseline) / (input + 1e-4)

    def is_image_invalid(self, img, threshold=0.01):
        """Check if image is invalid (completely black, white, or has no variation)."""
        try:
            if np.max(img) < 0.05:
                return True
            if np.min(img) > 0.95:
                return True
            if np.std(img) < threshold:
                return True
            if np.any(np.isnan(img)) or np.any(np.isinf(img)):
                return True
            if np.min(img) < -0.1 or np.max(img) > 1.1:
                return True
            return False
        except Exception as e:
            print(f"Error validating image: {e}")
            return True

    def process_image(self, img, type="image", channels=3, srgb=False, normalize=True, max_value=1.0, min_value=0.0):
        """Process image using rgbx_data.py pipeline to avoid artifacts."""
        try:
            valid = True
            if type in ['depth', 'roughness', 'metallic'] and img.ndim == 3:
                img = img[:, :, 1] if type == 'metallic' else img[:, :, 0]

            infinite = ~np.isfinite(img)
            if np.all(infinite):
                print(f"Warning: '{type}' image has no finite values.")
                img = 0.5 * np.ones_like(img)
                infinite = np.zeros_like(infinite, dtype=bool)
                valid = False

            img[infinite] = img[~infinite].min()

            if srgb is not None:
                if srgb == 'approx':
                    img = np.clip(img, 0.0, 1.0) ** (1 / 2.2)
                elif srgb == 'inv':
                    if np.max(img) > 300 or np.min(img) < 0:
                        print(f"Warning: sRGB 'inv' input has unusual range [{np.min(img):.2f}, {np.max(img):.2f}]")
                    img = img.astype(np.float32) / 255.
                    img = img ** 2.2
                    img = np.clip(img, 0.0, 1.0) ** (1 / 2.2)
                    if np.max(img) < 0.01 or np.min(img) > 0.99:
                        print(f"Warning: sRGB 'inv' processing resulted in extreme image [{np.min(img):.6f}, {np.max(img):.6f}]")
                        valid = False
            else:
                if type == 'depth' and valid:
                    img = self.depth_to_disparity(img)
                    img = img / (img.max() + 1e-4)
                if normalize:
                    if valid and img.max() == img.min() and type in ['roughness', 'metallic']:
                        pass
                    elif valid and img.max() > img.min():
                        img = np.clip(img, min_value, max_value)
                        img = (img - min_value) / (max_value - min_value)
                    else:
                        img = 0.5 * np.ones_like(img)
                        valid = False

            img = np.clip(img, 0.0, 1.0)

            if channels > 1:
                img[infinite] = 1.0
            else:
                img[infinite] = 0.0

            if channels == 1:
                img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

        except Exception as e:
            print(f"Error processing image: {e}")
            img = 0.5 * np.ones_like(img)
            valid = False
        return img, valid

    def process_raw_depth(self, img):
        """Process raw depth image without converting to disparity."""
        try:
            valid = True

            if img.ndim == 3:
                img = img[:, :, 0]

            infinite = ~np.isfinite(img)
            if np.all(infinite):
                print(f"Warning: depth image has no finite values.")
                img = np.ones_like(img) * 10.0
                infinite = np.zeros_like(infinite, dtype=bool)
                valid = False

            max_finite = img[~infinite].max() if np.any(~infinite) else 10.0
            img[infinite] = max_finite * 2.0

            if valid and img.max() > img.min():
                img = (img - img.min()) / (img.max() - img.min())
            else:
                img = 0.5 * np.ones_like(img)
                valid = False

            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

            return img, valid

        except Exception as e:
            print(f"Error processing raw depth: {e}")
            if len(img.shape) == 2:
                h, w = img.shape
                img = 0.5 * np.ones((h, w, 3), dtype=np.float32)
            else:
                img = 0.5 * np.ones_like(img, dtype=np.float32)
            return img, False

    def load_local_paths(self, data_dir, modalities, max_samples=None):
        """Discover scene files from a local directory.

        Expected naming: ``{scene_id}_rgb.{ext}``, ``{scene_id}_depth.{ext}``, etc.
        """
        rgb_paths = sorted(
            glob.glob(os.path.join(data_dir, '*_rgb.png')) +
            glob.glob(os.path.join(data_dir, '*_rgb.jpg')) +
            glob.glob(os.path.join(data_dir, '*_rgb.jpeg'))
        )

        if not rgb_paths:
            print(f"No RGB images found in {data_dir} (expected *_rgb.png or *_rgb.jpg)")
            return {}

        print(f"Found {len(rgb_paths)} RGB images in {data_dir}")

        if max_samples and len(rgb_paths) > max_samples:
            rgb_paths = random.sample(rgb_paths, max_samples)

        path_dict = {"rgb": rgb_paths}

        for modality in modalities:
            mod_paths = []
            for rgb_path in rgb_paths:
                mod_path = rgb_path.replace('_rgb.', f'_{modality}.')
                mod_paths.append(mod_path if os.path.exists(mod_path) else None)
            path_dict[modality] = mod_paths

        available = {m: sum(1 for p in path_dict[m] if p is not None) for m in modalities}
        print(f"Available modality maps: {available}")

        return path_dict

    def load_local_image(self, path, modality="image"):
        """Load and process a local image file (PNG / JPG)."""
        try:
            if modality == 'depth':
                # load without RGB conversion to preserve 16-bit precision
                img = np.array(Image.open(path)).astype(np.float32)
                return self.process_raw_depth(img)

            img = np.array(Image.open(path).convert('RGB')).astype(np.float32)

            if modality == 'normal':
                # PNG normal maps store [-1,1] as [0,255] via (n + 1) / 2 * 255
                img = img / 255.0 * 2.0 - 1.0
                return self.process_image(img, type='normal', channels=3, srgb=None,
                                          min_value=-1.0, max_value=1.0)
            else:
                img = img / 255.0
                return self.process_image(
                    img, type=modality if modality != 'image' else 'image',
                    channels=3, srgb=None, min_value=0.0, max_value=1.0)

        except Exception as e:
            print(f"Error loading {path}: {e}")
            return np.zeros((256, 256, 3), dtype=np.float32), False

    # ------------------------------------------------------------------
    # Point sampling & ground truth
    # ------------------------------------------------------------------

    def _sample_balanced_albedo_points(self, rgb_img, albedo_img, prefer_same_color=False, max_attempts=50):
        """Sample point pairs for albedo biased towards 'same' or 'different' base colors."""
        h, w = self.image_size
        margin = 30

        for attempt in range(max_attempts):
            distance_type = random.choice(['very_close', 'close', 'medium', 'far', 'very_far'])

            min_distance = 60
            max_distance = 150

            if distance_type == 'very_close':
                min_distance = 20
                max_distance = 40
            elif distance_type == 'close':
                min_distance = random.randint(30, 50)
                max_distance = 80
            elif distance_type == 'medium':
                min_distance = random.randint(60, 100)
                max_distance = 150
            elif distance_type == 'far':
                min_distance = random.randint(120, 200)
                max_distance = 280
            elif distance_type == 'very_far':
                min_distance = random.randint(220, 300)
                max_distance = 350

            point1_x = random.randint(margin, w - margin)
            point1_y = random.randint(margin, h - margin)

            attempts = 0
            max_point_attempts = 100
            point2_x = random.randint(margin, w - margin)
            point2_y = random.randint(margin, h - margin)

            while attempts < max_point_attempts:
                distance = np.sqrt((point1_x - point2_x)**2 + (point1_y - point2_y)**2)
                if min_distance <= distance <= max_distance:
                    break
                point2_x = random.randint(margin, w - margin)
                point2_y = random.randint(margin, h - margin)
                attempts += 1

            if albedo_img is not None:
                val1 = albedo_img[point1_y, point1_x]
                val2 = albedo_img[point2_y, point2_x]

                if len(val1.shape) > 0 and len(val1) == 3 and len(val2.shape) > 0 and len(val2) == 3:
                    color_diff = np.sqrt(np.sum((val1 - val2)**2))
                    color_threshold = 0.02
                    is_same_color = color_diff < color_threshold

                    if (prefer_same_color and is_same_color) or (not prefer_same_color and not is_same_color):
                        return point1_x, point1_y, point2_x, point2_y, distance_type

                    if attempt > max_attempts // 2:
                        return point1_x, point1_y, point2_x, point2_y, distance_type
            else:
                return point1_x, point1_y, point2_x, point2_y, distance_type

        return None

    def create_rgb_image_with_points_and_judgment(self, rgb_img_processed, intrinsic_img_processed, modality, target_question=None, albedo_img_processed=None, prefer_same_answer=None):
        """Create RGB image with colored points and generate ground truth judgment."""
        rgb_img, rgb_valid = rgb_img_processed
        intrinsic_img, intrinsic_valid = intrinsic_img_processed

        if not rgb_valid:
            print("Warning: RGB image is invalid")
        if not intrinsic_valid:
            print("Warning: Intrinsic image is invalid")

        if self.is_image_invalid(rgb_img):
            print(f"Warning: RGB image is completely black/white, skipping...")
            return None, "", "", [], {}

        rgb_uint8 = (rgb_img * 255).astype(np.uint8)
        rgb_pil = Image.fromarray(rgb_uint8).convert('RGB')
        rgb_pil = rgb_pil.resize(self.image_size, Image.LANCZOS)

        intrinsic_resized = cv2.resize(intrinsic_img, self.image_size, interpolation=cv2.INTER_NEAREST)

        albedo_resized = None
        if albedo_img_processed is not None:
            albedo_img, albedo_valid = albedo_img_processed
            if albedo_valid:
                albedo_resized = cv2.resize(albedo_img, self.image_size, interpolation=cv2.INTER_NEAREST)

        h, w = self.image_size

        # --- Point sampling ---
        if modality == "albedo" and albedo_resized is not None and prefer_same_answer is not None:
            points_result = self._sample_balanced_albedo_points(
                rgb_img, albedo_resized, prefer_same_color=prefer_same_answer, max_attempts=100
            )
            if points_result is not None:
                point1_x, point1_y, point2_x, point2_y, distance_type = points_result
                min_distance = 20 if distance_type in ['very_close', 'close'] else 60
                max_distance = 80 if distance_type in ['very_close', 'close'] else 150
            else:
                distance_type = 'medium'
                point1_x = random.randint(30, w - 30)
                point1_y = random.randint(30, h - 30)
                point2_x = random.randint(30, w - 30)
                point2_y = random.randint(30, h - 30)
                min_distance = 60
                max_distance = 150
        else:
            margin = 30
            distance_type = random.choice(['very_close', 'close', 'medium', 'far', 'very_far'])
            min_distance = 60
            max_distance = 150

            if distance_type == 'very_close':
                min_distance = 20
                max_distance = 40
            elif distance_type == 'close':
                min_distance = random.randint(30, 50)
                max_distance = 80
            elif distance_type == 'medium':
                min_distance = random.randint(60, 100)
                max_distance = 150
            elif distance_type == 'far':
                min_distance = random.randint(120, 200)
                max_distance = 280
            elif distance_type == 'very_far':
                min_distance = random.randint(220, 300)
                max_distance = 350

            point1_x = random.randint(margin, w - margin)
            point1_y = random.randint(margin, h - margin)

            attempts = 0
            max_attempts = 100
            point2_x = random.randint(margin, w - margin)
            point2_y = random.randint(margin, h - margin)

            while attempts < max_attempts:
                distance = np.sqrt((point1_x - point2_x)**2 + (point1_y - point2_y)**2)
                if min_distance <= distance <= max_distance:
                    break
                point2_x = random.randint(margin, w - margin)
                point2_y = random.randint(margin, h - margin)
                attempts += 1

            if attempts >= max_attempts:
                attempts = 0
                if distance_type in ['very_close', 'close']:
                    fallback_min_distance = 20
                    fallback_max_distance = 80
                else:
                    fallback_min_distance = 40
                    fallback_max_distance = min(max_distance, 350)

                while attempts < 50:
                    point2_x = random.randint(margin, w - margin)
                    point2_y = random.randint(margin, h - margin)
                    actual_distance = np.sqrt((point1_x - point2_x)**2 + (point1_y - point2_y)**2)
                    if fallback_min_distance <= actual_distance <= fallback_max_distance:
                        break
                    attempts += 1

                if attempts >= 50:
                    attempts = 0
                    while (np.sqrt((point1_x - point2_x)**2 + (point1_y - point2_y)**2) < 20 and
                           attempts < 30):
                        point2_x = random.randint(margin, w - margin)
                        point2_y = random.randint(margin, h - margin)
                        attempts += 1

        # --- Ground truth ---
        rgb_resized = cv2.resize(rgb_img, self.image_size, interpolation=cv2.INTER_NEAREST)

        question, answer, comparison_result = self._compute_ground_truth(
            intrinsic_resized, modality, point1_x, point1_y, point2_x, point2_y, target_question,
            rgb_img=rgb_resized, albedo_img=albedo_resized
        )

        if question == "":
            return None, "", "", [], {}

        # --- Color assignment ---
        red_color = (255, 0, 0)
        green_color = (0, 255, 0)

        if comparison_result in ["unknown", "same"]:
            if random.random() < 0.5:
                color1, color2 = red_color, green_color
            else:
                color1, color2 = green_color, red_color
        else:
            red_is_first = random.random() < 0.5

            if red_is_first:
                if comparison_result == "point1":
                    color1, color2 = red_color, green_color
                else:
                    color1, color2 = green_color, red_color
            else:
                if comparison_result == "point1":
                    color1, color2 = green_color, red_color
                else:
                    color1, color2 = red_color, green_color

            if ((modality == "normal" and "same" in question.lower()) or
                (modality == "albedo" and "same" in question.lower())):
                pass
            else:
                answer = self._update_answer_for_colors(answer, comparison_result, red_is_first, modality)

        point_coords = [(point1_x, point1_y), (point2_x, point2_y)]

        rgb_with_points = self._draw_points_pytorch(rgb_pil, point_coords, color1, color2)

        final_distance = np.sqrt((point1_x - point2_x)**2 + (point1_y - point2_y)**2)
        metadata = {
            'modality': modality,
            'comparison_result': comparison_result,
            'distance_type': distance_type,
            'actual_distance': float(final_distance),
            'target_min': min_distance,
            'target_max': max_distance,
            'point_coordinates': point_coords,
            'colors_used': {'color1': color1, 'color2': color2}
        }

        return rgb_with_points, question, answer, point_coords, metadata

    def _compute_ground_truth(self, intrinsic_img, modality, point1_x, point1_y, point2_x, point2_y, target_question=None, rgb_img=None, albedo_img=None):
        """Compute ground truth answer based on intrinsic image values."""
        val1 = intrinsic_img[point1_y, point1_x]
        val2 = intrinsic_img[point2_y, point2_x]

        if modality == "depth":
            question = QUESTION_TEMPLATES["depth"][0]

            depth1 = float(val1[0]) if hasattr(val1, '__len__') and len(val1) > 0 else float(val1)
            depth2 = float(val2[0]) if hasattr(val2, '__len__') and len(val2) > 0 else float(val2)

            if not (np.isfinite(depth1) and np.isfinite(depth2)):
                return "", "", "unknown"

            depth_range = intrinsic_img.max() - intrinsic_img.min()
            if depth_range < 1e-6:
                return "", "", "unknown"

            depth_diff = abs(depth1 - depth2)

            if depth_diff < 0.02:
                return "", "", "unknown"

            if depth1 < depth2:
                answer = "Based on visual depth cues in the image, the red point appears closer to the camera."
                comparison_result = "point1"
            else:
                answer = "Based on visual depth cues in the image, the green point appears closer to the camera."
                comparison_result = "point2"

        elif modality == "normal":
            if target_question is None:
                question = random.choice(QUESTION_TEMPLATES["normal"])
            else:
                question = target_question

            if "same" in question.lower():
                norm1 = np.linalg.norm(val1)
                norm2 = np.linalg.norm(val2)
                if norm1 < 1e-6 or norm2 < 1e-6:
                    return "", "", "unknown"
                n1 = val1 / norm1
                n2 = val2 / norm2
                similarity = np.dot(n1, n2)
                threshold = 0.97
                if similarity > threshold:
                    answer = "Yes, the red and green points have similar surface orientations."
                    comparison_result = "same"
                else:
                    answer = "No, the red and green points have different surface orientations."
                    comparison_result = "different"
            else:
                z1 = val1[2] if len(val1) >= 3 else 0
                z2 = val2[2] if len(val2) >= 3 else 0
                if abs(z1 - z2) < 0.02:
                    return "", "", "unknown"
                if z1 > z2:
                    answer = "The red point has a surface more facing towards the camera."
                    comparison_result = "point1"
                else:
                    answer = "The green point has a surface more facing towards the camera."
                    comparison_result = "point2"

        elif modality == "irradiance":
            question = QUESTION_TEMPLATES["irradiance"][0]

            if rgb_img is not None and albedo_img is not None:
                rgb_val1 = rgb_img[point1_y, point1_x]
                rgb_val2 = rgb_img[point2_y, point2_x]
                albedo_val1 = albedo_img[point1_y, point1_x]
                albedo_val2 = albedo_img[point2_y, point2_x]

                rgb_lum1 = 0.299*rgb_val1[0] + 0.587*rgb_val1[1] + 0.114*rgb_val1[2] if len(rgb_val1) == 3 else np.mean(rgb_val1)
                rgb_lum2 = 0.299*rgb_val2[0] + 0.587*rgb_val2[1] + 0.114*rgb_val2[2] if len(rgb_val2) == 3 else np.mean(rgb_val2)
                albedo_lum1 = 0.299*albedo_val1[0] + 0.587*albedo_val1[1] + 0.114*albedo_val1[2] if len(albedo_val1) == 3 else np.mean(albedo_val1)
                albedo_lum2 = 0.299*albedo_val2[0] + 0.587*albedo_val2[1] + 0.114*albedo_val2[2] if len(albedo_val2) == 3 else np.mean(albedo_val2)

                eps = 1e-6
                effective_irrad1 = rgb_lum1 / (albedo_lum1 + eps)
                effective_irrad2 = rgb_lum2 / (albedo_lum2 + eps)

                if not (np.isfinite(effective_irrad1) and np.isfinite(effective_irrad2)):
                    return "", "", "unknown"

                irrad1, irrad2 = effective_irrad1, effective_irrad2
            else:
                if len(val1.shape) > 0 and len(val1) == 3:
                    irrad1 = 0.299*val1[0] + 0.587*val1[1] + 0.114*val1[2]
                else:
                    irrad1 = np.mean(val1) if len(val1.shape) > 0 else val1
                if len(val2.shape) > 0 and len(val2) == 3:
                    irrad2 = 0.299*val2[0] + 0.587*val2[1] + 0.114*val2[2]
                else:
                    irrad2 = np.mean(val2) if len(val2.shape) > 0 else val2

            if not (np.isfinite(irrad1) and np.isfinite(irrad2)):
                return "", "", "unknown"

            irrad_diff = abs(irrad1 - irrad2)
            irrad_range = max(irrad1, irrad2) - min(irrad1, irrad2)
            adaptive_threshold = max(0.02, irrad_range * 0.01) if irrad_range > 0 else 0.02

            if irrad_diff < adaptive_threshold:
                return "", "", "unknown"

            if irrad1 > irrad2:
                answer = "The red point is more illuminated."
                comparison_result = "point1"
            else:
                answer = "The green point is more illuminated."
                comparison_result = "point2"

        elif modality == "albedo":
            if target_question is None:
                question = random.choice(QUESTION_TEMPLATES["albedo"])
            else:
                question = target_question

            if len(val1.shape) > 0 and len(val1) == 3:
                bright1 = 0.299*val1[0] + 0.587*val1[1] + 0.114*val1[2]
            else:
                bright1 = np.mean(val1) if len(val1.shape) > 0 else val1
            if len(val2.shape) > 0 and len(val2) == 3:
                bright2 = 0.299*val2[0] + 0.587*val2[1] + 0.114*val2[2]
            else:
                bright2 = np.mean(val2) if len(val2.shape) > 0 else val2

            if not (np.isfinite(bright1) and np.isfinite(bright2)):
                return "", "", "unknown"

            if len(val1.shape) > 0 and len(val1) == 3 and len(val2.shape) > 0 and len(val2) == 3:
                color_diff = np.sqrt(np.sum((val1 - val2)**2))
                color_threshold = 0.05
                if color_diff < color_threshold:
                    answer = "Yes, the red and green points have similar base colors."
                    comparison_result = "same"
                else:
                    answer = "No, the red and green points have different base colors."
                    comparison_result = "different"
            else:
                if abs(bright1 - bright2) < 0.05:
                    answer = "Yes, the red and green points have similar base colors."
                    comparison_result = "same"
                else:
                    answer = "No, the red and green points have different base colors."
                    comparison_result = "different"
        else:
            return "", "", "unknown"

        return question, answer, comparison_result

    def _update_answer_for_colors(self, answer, comparison_result, red_is_first, modality):
        """Update answer to reflect the actual color assignment."""
        if comparison_result == "unknown" or "same" in answer.lower() or "similar" in answer.lower():
            return answer

        winner_color = "red" if red_is_first else "green"

        if modality == "depth":
            answer = f"Based on visual depth cues in the image, the {winner_color} point appears closer to the camera."
        elif modality == "normal" and "facing" in answer.lower():
            answer = f"The {winner_color} point has a surface more facing towards the camera."
        elif modality == "irradiance":
            answer = f"The {winner_color} point is more illuminated."
        elif modality == "albedo":
            answer = f"The {winner_color} point has a more reflective surface."

        return answer

    def _draw_points_pytorch(self, rgb_pil, point_coords, color1, color2):
        """Draw hollow circles with center points using PyTorch tensors."""
        point1, point2 = point_coords
        point1_x, point1_y = point1
        point2_x, point2_y = point2

        rgb_np = np.array(rgb_pil).astype(np.float32)
        rgb_tensor = torch.from_numpy(rgb_np / 127.5 - 1.0)
        rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0)

        circle_radius = 5.0
        center_radius = 1.0
        line_width = 1.0

        h, w = rgb_tensor.shape[-2:]
        y_coords = torch.arange(h).float()
        x_coords = torch.arange(w).float()
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        dist1 = torch.sqrt((xx - point1_x)**2 + (yy - point1_y)**2)
        circle_outline1 = torch.abs(dist1 - circle_radius) <= line_width
        circle_mask1 = circle_outline1.float()
        center_mask1 = (dist1 <= center_radius).float()
        point1_mask = torch.clamp(circle_mask1 + center_mask1, 0, 1)

        dist2 = torch.sqrt((xx - point2_x)**2 + (yy - point2_y)**2)
        circle_outline2 = torch.abs(dist2 - circle_radius) <= line_width
        circle_mask2 = circle_outline2.float()
        center_mask2 = (dist2 <= center_radius).float()
        point2_mask = torch.clamp(circle_mask2 + center_mask2, 0, 1)

        color1_tensor = torch.tensor(color1, dtype=torch.float32) / 127.5 - 1.0
        color2_tensor = torch.tensor(color2, dtype=torch.float32) / 127.5 - 1.0

        for c in range(3):
            rgb_tensor[0, c] = rgb_tensor[0, c] * (1 - point1_mask) + color1_tensor[c] * point1_mask
            rgb_tensor[0, c] = rgb_tensor[0, c] * (1 - point2_mask) + color2_tensor[c] * point2_mask

        rgb_tensor = torch.clamp(rgb_tensor, -1.0, 1.0)
        rgb_np = ((rgb_tensor[0].permute(1, 2, 0) + 1.0) * 127.5).numpy().astype(np.uint8)
        return Image.fromarray(rgb_np)

    def _convert_to_display_format(self, img_array):
        """Convert processed intrinsic image to displayable uint8 format."""
        tensor_format = img_array * 2.0 - 1.0
        display_img = 255 * ((tensor_format + 1) / 2)
        display_img = display_img.clip(0, 255).astype(np.uint8)
        return display_img

    def _save_intrinsic_with_points(self, intrinsic_img, point_coords, color1, color2, save_path, modality, rgb_with_points=None, question="", answer=""):
        """Save stitched RGB and intrinsic images with points overlaid for quality checking."""
        try:
            if not isinstance(intrinsic_img, np.ndarray):
                intrinsic_img = np.array(intrinsic_img)
            intrinsic_img = np.squeeze(intrinsic_img)

            if len(intrinsic_img.shape) == 1:
                print(f"Warning: 1D array of length {len(intrinsic_img)}, cannot process")
                return
            elif len(intrinsic_img.shape) == 2:
                intrinsic_img = np.stack([intrinsic_img]*3, axis=-1)
            elif len(intrinsic_img.shape) > 3:
                print(f"Warning: Too many dimensions {intrinsic_img.shape}")
                return

            intrinsic_resized = cv2.resize(intrinsic_img, self.image_size, interpolation=cv2.INTER_LANCZOS4)
            display_img = self._convert_to_display_format(intrinsic_resized)

            if len(display_img.shape) != 3 or display_img.shape[2] != 3:
                print(f"Warning: Unexpected final shape {display_img.shape} for PIL")
                return

            intrinsic_pil = Image.fromarray(display_img, mode='RGB')

            text_area_height = 0
            if question and answer:
                intrinsic_pil, text_area_height = self._add_debug_text_with_offset(intrinsic_pil, question, answer, modality)

            intrinsic_adjusted_coords = [(x, y + text_area_height) for x, y in point_coords]
            intrinsic_with_points = self._draw_points_pytorch(intrinsic_pil, intrinsic_adjusted_coords, color1, color2)

            if rgb_with_points is not None:
                rgb_width, rgb_height = rgb_with_points.size
                intrinsic_width, intrinsic_height = intrinsic_with_points.size

                rgb_with_padding = Image.new('RGB', (rgb_width, rgb_height + text_area_height), (40, 40, 40))
                rgb_with_padding.paste(rgb_with_points, (0, text_area_height))

                rgb_adjusted_coords = [(x, y + text_area_height) for x, y in point_coords]
                rgb_with_padding_and_points = self._draw_points_pytorch(rgb_with_padding, rgb_adjusted_coords, color1, color2)

                target_width = min(rgb_width, intrinsic_width)
                target_height = rgb_height + text_area_height

                rgb_resized = rgb_with_padding_and_points.resize((target_width, target_height), Image.LANCZOS)
                intrinsic_resized_pil = intrinsic_with_points.resize((target_width, target_height), Image.LANCZOS)

                total_width = target_width * 2 + 10
                stitched_image = Image.new('RGB', (total_width, target_height), (40, 40, 40))
                stitched_image.paste(rgb_resized, (0, 0))
                stitched_image.paste(intrinsic_resized_pil, (target_width + 10, 0))
                stitched_image.save(save_path, quality=100)
            else:
                intrinsic_with_points.save(save_path, quality=100)

        except Exception as e:
            print(f"Error saving stitched debug image ({modality}): {e}")
            import traceback
            traceback.print_exc()

    def _add_debug_text_with_offset(self, image, question, answer, modality):
        """Add question and answer text in padding area for debugging."""
        try:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            except Exception:
                try:
                    font = ImageFont.truetype("arial.ttf", 14)
                except Exception:
                    font = ImageFont.load_default()

            img_width, img_height = image.size

            question_lines = textwrap.wrap(f"Q: {question}", width=65)
            answer_lines = textwrap.wrap(f"A: {answer}", width=65)
            all_lines = [f"Modality: {modality.upper()}"] + [""] + question_lines + [""] + answer_lines

            line_height = 18
            padding = 15
            text_area_height = len(all_lines) * line_height + 2 * padding

            new_image = Image.new('RGB', (img_width, img_height + text_area_height), (40, 40, 40))
            new_image.paste(image, (0, text_area_height))

            draw = ImageDraw.Draw(new_image)
            y_position = padding

            modality_colors = {
                "depth": (255, 255, 100),
                "normal": (100, 255, 255),
                "albedo": (255, 100, 255),
                "irradiance": (255, 255, 255),
            }
            text_color = modality_colors.get(modality, (255, 255, 255))

            for line in all_lines:
                if line.strip():
                    draw.text((10, y_position), line, fill=text_color, font=font)
                y_position += line_height

            return new_image, text_area_height

        except Exception as e:
            print(f"Error adding debug text: {e}")
            return image, 0

    def generate_dataset(self, data_dir, output_dir, max_samples=None, modalities=None, save_intrinsic_debug=False):
        """Generate complete training dataset from a local folder of intrinsic scene images."""

        if modalities is None:
            modalities = ["depth", "normal", "irradiance", "albedo"]

        os.makedirs(output_dir, exist_ok=True)
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        if save_intrinsic_debug:
            debug_dir = os.path.join(output_dir, "debug_intrinsic")
            os.makedirs(debug_dir, exist_ok=True)
            print(f"Debug intrinsic images will be saved to: {debug_dir}")

        jsonl_path = os.path.join(output_dir, "intrinsic_judgement_train.jsonl")

        path_dict = self.load_local_paths(data_dir, modalities, max_samples=max_samples)
        if not path_dict:
            print("No data found. Exiting.")
            return jsonl_path, 0

        dataset_name = os.path.basename(os.path.normpath(data_dir))
        num_rgb_images = len(path_dict["rgb"])

        print(f"Processing {num_rgb_images} RGB images from '{dataset_name}'")
        print(f"Modalities: {modalities}")
        expected = num_rgb_images * len(modalities)
        print(f"Expected samples (upper bound): {expected}")

        all_samples = []
        sample_id = 0
        successful_samples = 0
        skipped_loading = 0
        modality_stats = {mod: {"success": 0, "failed": 0} for mod in modalities}
        albedo_answer_stats = {"same": 0, "different": 0}

        progress = tqdm(range(num_rgb_images), desc="Processing images", unit="img")

        for i in progress:
            try:
                rgb_path = path_dict["rgb"][i]
                rgb_processed = self.load_local_image(rgb_path, modality="image")
                rgb_img, rgb_valid = rgb_processed

                if not rgb_valid or rgb_img.max() == 0:
                    skipped_loading += 1
                    progress.set_postfix({'ok': successful_samples, 'skip': skipped_loading, 'samples': sample_id})
                    continue

                for modality in modalities:
                    mod_path = path_dict.get(modality, [None] * num_rgb_images)[i]
                    if mod_path is None:
                        modality_stats[modality]["failed"] += 1
                        continue

                    try:
                        intrinsic_processed = self.load_local_image(mod_path, modality=modality)
                        intrinsic_img, intrinsic_valid = intrinsic_processed

                        if not intrinsic_valid or intrinsic_img.max() == 0:
                            modality_stats[modality]["failed"] += 1
                            continue

                        albedo_processed = None
                        if modality in ('irradiance', 'albedo'):
                            albedo_path = path_dict.get('albedo', [None] * num_rgb_images)[i]
                            if albedo_path is not None:
                                try:
                                    albedo_processed = self.load_local_image(albedo_path, modality='albedo')
                                    if not albedo_processed[1]:
                                        albedo_processed = None
                                except Exception:
                                    albedo_processed = None

                        prefer_same_answer = None
                        if modality == 'albedo':
                            total_albedo = albedo_answer_stats["same"] + albedo_answer_stats["different"]
                            if total_albedo > 0:
                                prefer_same_answer = albedo_answer_stats["same"] / total_albedo < 0.4
                            else:
                                prefer_same_answer = True

                        max_retries = 20
                        result = None
                        for _retry in range(max_retries):
                            result = self.create_rgb_image_with_points_and_judgment(
                                rgb_processed, intrinsic_processed, modality,
                                albedo_img_processed=albedo_processed, prefer_same_answer=prefer_same_answer
                            )
                            if result[0] is not None and result[1] != "" and result[2] != "":
                                break
                        else:
                            modality_stats[modality]["failed"] += 1
                            continue

                        rgb_with_points, question, answer, point_coords, metadata = result

                        image_filename = f"{dataset_name}_{modality}_{sample_id:06d}.png"
                        image_path = os.path.join(images_dir, image_filename)
                        rgb_with_points.save(image_path, quality=100)

                        if save_intrinsic_debug:
                            debug_filename = f"{dataset_name}_{modality}_{sample_id:06d}_intrinsic.png"
                            debug_path = os.path.join(debug_dir, debug_filename)
                            colors_used = metadata.get('colors_used', {'color1': (255, 0, 0), 'color2': (0, 255, 0)})
                            self._save_intrinsic_with_points(
                                intrinsic_img, point_coords,
                                colors_used['color1'], colors_used['color2'],
                                debug_path, modality, rgb_with_points, question, answer
                            )

                        conversations = [
                            {"from": "human", "value": f"<image>\n{question}"},
                            {"from": "gpt", "value": answer}
                        ]

                        sample = {
                            "id": f"{dataset_name}_{modality}_{sample_id:06d}",
                            "image": f"images/{image_filename}",
                            "conversations": conversations,
                            "metadata": {
                                "dataset": dataset_name,
                                "modality": modality,
                                "question": question,
                                "answer": answer,
                                "rgb_path": rgb_path,
                                "intrinsic_path": mod_path,
                                **metadata
                            }
                        }

                        all_samples.append(sample)
                        sample_id += 1
                        modality_stats[modality]["success"] += 1

                        if modality == 'albedo':
                            if 'same' in answer.lower() or 'similar' in answer.lower():
                                albedo_answer_stats["same"] += 1
                            else:
                                albedo_answer_stats["different"] += 1

                    except Exception as e:
                        print(f"Error processing {modality} for sample {i}: {e}")
                        modality_stats[modality]["failed"] += 1
                        continue

                successful_samples += 1
                progress.set_postfix({'ok': successful_samples, 'skip': skipped_loading, 'samples': sample_id})

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                skipped_loading += 1
                continue

        # --- Statistics ---
        print(f"\nProcessed {successful_samples}/{num_rgb_images} RGB images")
        if skipped_loading > 0:
            print(f"  Skipped {skipped_loading} images due to loading failures")
        for mod in modalities:
            s = modality_stats[mod]
            total = s["success"] + s["failed"]
            rate = s["success"] / total * 100 if total > 0 else 0
            print(f"  {mod}: {s['success']} ok, {s['failed']} failed ({rate:.1f}%)")

        if 'albedo' in modalities and sum(albedo_answer_stats.values()) > 0:
            total_a = sum(albedo_answer_stats.values())
            print(f"  Albedo balance: {albedo_answer_stats['same']} same ({albedo_answer_stats['same']/total_a*100:.1f}%), "
                  f"{albedo_answer_stats['different']} different ({albedo_answer_stats['different']/total_a*100:.1f}%)")

        # tain / val split 
        random.shuffle(all_samples)
        val_split_ratio = 0.1

        samples_by_modality = {}
        for sample in all_samples:
            m = sample['metadata']['modality']
            samples_by_modality.setdefault(m, []).append(sample)

        train_samples, val_samples = [], []
        for m, ms in samples_by_modality.items():
            num_val = max(1, int(len(ms) * val_split_ratio))
            val_samples.extend(ms[:num_val])
            train_samples.extend(ms[num_val:])

        train_jsonl_path = jsonl_path
        with open(train_jsonl_path, 'w') as f:
            for s in train_samples:
                f.write(json.dumps(s) + '\n')

        val_jsonl_path = jsonl_path.replace('_train.jsonl', '_val.jsonl')
        with open(val_jsonl_path, 'w') as f:
            for s in val_samples:
                f.write(json.dumps(s) + '\n')

        print(f"Dataset generation complete!")
        print(f"Total samples: {len(all_samples)} (train: {len(train_samples)}, val: {len(val_samples)})")
        for mod in modalities:
            tc = sum(1 for s in train_samples if s['metadata']['modality'] == mod)
            vc = sum(1 for s in val_samples if s['metadata']['modality'] == mod)
            print(f"  {mod}: {tc + vc} total -> train {tc}, val {vc}")
        print(f"Output: {output_dir}")
        print(f"Train JSONL: {train_jsonl_path}")
        print(f"Val JSONL:   {val_jsonl_path}")

        return train_jsonl_path, len(all_samples)


def main():
    parser = argparse.ArgumentParser(
        description='Generate intrinsic image judgment training data from a local folder of scene images.')
    parser.add_argument('--data_dir', required=True,
                        help='Directory with {id}_rgb.png, {id}_depth.png, {id}_normal.png, etc.')
    parser.add_argument('--output_dir', default='data/intrinsic_judgements',
                        help='Output directory for generated training data')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit number of RGB images to process')
    parser.add_argument('--modalities', nargs='+',
                        default=["depth", "normal", "irradiance", "albedo"],
                        help='Modalities to generate data for')
    parser.add_argument('--save_intrinsic_debug', action='store_true',
                        help='Save side-by-side debug images with intrinsic ground truth')

    args = parser.parse_args()

    generator = RGBXIntrinsicJudgmentGenerator()
    jsonl_path, num_samples = generator.generate_dataset(
        args.data_dir,
        args.output_dir,
        max_samples=args.max_samples,
        modalities=args.modalities,
        save_intrinsic_debug=args.save_intrinsic_debug
    )

    print(f"\nReady for InternVL2.5-4B LoRA training!")
    print(f"JSONL: {jsonl_path}")
    print(f"Total samples: {num_samples}")


if __name__ == "__main__":
    main()
