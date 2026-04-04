import os
import re
import json
import math
import argparse
import textwrap

import cv2
import numpy as np
from tqdm import tqdm
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from collections import defaultdict
from sklearn.metrics import f1_score, precision_recall_fscore_support


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=512, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=512, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def parse_judgment(response, modality, question):
    """
    Parse the model's response to extract the judgment for different modalities.
    
    Args:
        response: Model's text response
        modality: One of 'depth', 'normal', 'albedo', 'irradiance'
        question: The original question asked
    
    Returns:
        str: "red", "green", "same", "different", "unknown", or "parse_error"
    """
    response_lower = response.lower()
    
    # Common patterns
    red_patterns = [
        r"red.*(?:closer|more|brighter|higher|reflective)",
        r"red.*point.*(?:closer|more|brighter|higher|reflective)",
        r"the red",
        r"red is (?:closer|more|brighter|higher|reflective)"
    ]
    
    green_patterns = [
        r"green.*(?:closer|more|brighter|higher|reflective)",
        r"green.*point.*(?:closer|more|brighter|higher|reflective)",
        r"the green",
        r"green is (?:closer|more|brighter|higher|reflective)"
    ]
    
    # Modality-specific patterns
    if modality == "depth":
        red_patterns.extend([
            r"red.*closer",
            r"red.*appears.*closer",
            r"red.*point.*closer"
        ])
        green_patterns.extend([
            r"green.*closer",
            r"green.*appears.*closer", 
            r"green.*point.*closer"
        ])
    
    elif modality == "normal":
        if "same" in question.lower() or "orientation" in question.lower():
            # For orientation similarity questions
            same_patterns = [
                r"yes.*same.*orientation",
                r"yes.*similar.*orientation",
                r"same.*surface.*orientation",
                r"similar.*surface.*orientation",
                r"both.*same",
                r"both.*similar"
            ]
            different_patterns = [
                r"no.*different.*orientation", 
                r"no.*same.*orientation",
                r"different.*surface.*orientation",
                r"different.*orientation"
            ]
            
            for pattern in same_patterns:
                if re.search(pattern, response_lower):
                    return "same"
            for pattern in different_patterns:
                if re.search(pattern, response_lower):
                    return "different"
        else:
            # For "facing camera" questions
            red_patterns.extend([
                r"red.*facing.*camera",
                r"red.*surface.*facing",
                r"red.*more.*facing"
            ])
            green_patterns.extend([
                r"green.*facing.*camera",
                r"green.*surface.*facing", 
                r"green.*more.*facing"
            ])
    
    elif modality == "albedo":
        if "same" in question.lower() or "base color" in question.lower():
            # For base color similarity questions
            same_patterns = [
                r"yes.*same.*color",
                r"yes.*similar.*color",
                r"same.*base.*color",
                r"similar.*base.*color",
                r"both.*same.*color"
            ]
            different_patterns = [
                r"no.*different.*color",
                r"no.*same.*color", 
                r"different.*base.*color",
                r"different.*color"
            ]
            
            for pattern in same_patterns:
                if re.search(pattern, response_lower):
                    return "same"
            for pattern in different_patterns:
                if re.search(pattern, response_lower):
                    return "different"
        else:
            # For reflectiveness questions - check for "same/similar" responses first
            same_reflectance_patterns = [
                r"similar.*surface.*reflectance",
                r"same.*surface.*reflectance", 
                r"similar.*reflectance",
                r"same.*reflectance",
                r"both.*same.*reflective",
                r"both.*similar.*reflective",
                r"equal.*reflectance",
                r"identical.*reflectance"
            ]
            
            for pattern in same_reflectance_patterns:
                if re.search(pattern, response_lower):
                    return "same"
            
            # Then check for red/green specific patterns
            red_patterns.extend([
                r"red.*reflective",
                r"red.*surface.*reflective",
                r"red.*more.*reflective"
            ])
            green_patterns.extend([
                r"green.*reflective",
                r"green.*surface.*reflective",
                r"green.*more.*reflective"
            ])
    
    elif modality == "irradiance":
        # Check for similar illumination first
        same_illumination_patterns = [
            r"similar.*illumination",
            r"same.*illumination",
            r"similar.*lighting", 
            r"same.*lighting",
            r"both.*same.*illuminated",
            r"both.*similar.*illuminated",
            r"equal.*illumination",
            r"identical.*illumination"
        ]
        
        for pattern in same_illumination_patterns:
            if re.search(pattern, response_lower):
                return "same"
        
        red_patterns.extend([
            r"red.*illuminated",
            r"red.*brighter",
            r"red.*more.*illuminated",
            r"red.*more.*bright"
        ])
        green_patterns.extend([
            r"green.*illuminated",
            r"green.*brighter", 
            r"green.*more.*illuminated",
            r"green.*more.*bright"
        ])
    
    # Check for unknown/uncertain responses
    unknown_patterns = [
        r"difficult.*determine",
        r"cannot.*determine",
        r"hard.*tell",
        r"uncertain",
        r"unclear",
        r"can't.*tell",
        r"unable.*determine"
    ]
    
    for pattern in unknown_patterns:
        if re.search(pattern, response_lower):
            return "unknown"
    
    # Check for red vs green
    red_count = sum(1 for pattern in red_patterns if re.search(pattern, response_lower))
    green_count = sum(1 for pattern in green_patterns if re.search(pattern, response_lower))
    
    if red_count > green_count:
        return "red"
    elif green_count > red_count:
        return "green"
    elif red_count == green_count and red_count > 0:
        return "unknown"  # Ambiguous response
    
    # Simple word counting fallback
    red_mentions = response_lower.count("red")
    green_mentions = response_lower.count("green")
    
    if red_mentions > green_mentions:
        return "red"
    elif green_mentions > red_mentions:
        return "green"
    
    return "parse_error"

def _draw_points_pytorch(rgb_pil, point_coords, color1, color2):
    """Draw hollow circles with center points using PyTorch tensors (differentiable)"""
    point1, point2 = point_coords
    point1_x, point1_y = point1
    point2_x, point2_y = point2
    
    # Convert PIL to tensor [-1, 1] range
    rgb_np = np.array(rgb_pil).astype(np.float32)
    rgb_tensor = torch.from_numpy(rgb_np / 127.5 - 1.0)  # Convert [0, 255] -> [-1, 1]
    rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0)  # [H, W, C] -> [1, C, H, W]
    
    # Parameters for hollow circles with center points
    circle_radius = 5.0   # Circle radius
    center_radius = 1.0   # Center point radius
    line_width = 1.0      # Thickness of circle outline
    
    # Create coordinate grids
    h, w = rgb_tensor.shape[-2:]
    y_coords = torch.arange(h).float()
    x_coords = torch.arange(w).float()
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Point 1 (color1)
    dist1 = torch.sqrt((xx - point1_x)**2 + (yy - point1_y)**2)
    circle_outline1 = torch.abs(dist1 - circle_radius) <= line_width
    circle_mask1 = circle_outline1.float()
    center_mask1 = (dist1 <= center_radius).float()
    point1_mask = torch.clamp(circle_mask1 + center_mask1, 0, 1)
    
    # Point 2 (color2)
    dist2 = torch.sqrt((xx - point2_x)**2 + (yy - point2_y)**2)
    circle_outline2 = torch.abs(dist2 - circle_radius) <= line_width
    circle_mask2 = circle_outline2.float()
    center_mask2 = (dist2 <= center_radius).float()
    point2_mask = torch.clamp(circle_mask2 + center_mask2, 0, 1)
    
    # Convert colors to [-1, 1] range
    color1_tensor = torch.tensor(color1, dtype=torch.float32) / 127.5 - 1.0  # [0, 255] -> [-1, 1]
    color2_tensor = torch.tensor(color2, dtype=torch.float32) / 127.5 - 1.0
    
    # Apply colors to RGB tensor
    for c in range(3):  # RGB channels
        # Apply point1 color
        rgb_tensor[0, c] = rgb_tensor[0, c] * (1 - point1_mask) + color1_tensor[c] * point1_mask
        
        # Apply point2 color 
        rgb_tensor[0, c] = rgb_tensor[0, c] * (1 - point2_mask) + color2_tensor[c] * point2_mask
    
    # Convert back to PIL
    rgb_tensor = torch.clamp(rgb_tensor, -1.0, 1.0)
    rgb_np = ((rgb_tensor[0].permute(1, 2, 0) + 1.0) * 127.5).numpy().astype(np.uint8)
    rgb_pil_modified = Image.fromarray(rgb_np)
    
    return rgb_pil_modified

def _add_debug_text_with_comparison(image, question, ground_truth_answer, vlm_response, predicted_judgment, gt_judgment, modality, is_correct):
    """Add question, ground truth, and VLM response text in padding area for debugging"""
    try:
        # Try to use a reasonable font, fallback to default if not available
        try:
            font_size = 12
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            bold_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 12)
                bold_font = ImageFont.truetype("arialbd.ttf", 12)
            except:
                font = ImageFont.load_default()
                bold_font = font
        
        # Get original image dimensions
        img_width, img_height = image.size
        
        # Wrap text content
        question_lines = textwrap.wrap(f"Q: {question}", width=80)  # Wider for debug display
        gt_lines = textwrap.wrap(f"Ground Truth: {ground_truth_answer}", width=80)
        vlm_lines = textwrap.wrap(f"VLM Response: {vlm_response}", width=80)
        
        # Create header with modality and correctness info
        correctness_symbol = "OK" if is_correct else "WRONG"
        header = f"Modality: {modality.upper()} | Predicted: {predicted_judgment} | GT: {gt_judgment} | {correctness_symbol}"
        header_lines = textwrap.wrap(header, width=80)
        
        # Combine all text lines
        all_lines = header_lines + [""] + question_lines + [""] + gt_lines + [""] + vlm_lines
        
        # Calculate text area dimensions
        line_height = 16  # Pixels between lines
        padding = 12  # Top and bottom padding
        text_area_height = len(all_lines) * line_height + 2 * padding
        
        # Create new image with padding at the top
        new_height = img_height + text_area_height
        new_image = Image.new('RGB', (img_width, new_height), (30, 30, 30))  # Dark background
        
        # Paste original image below the text area
        new_image.paste(image, (0, text_area_height))
        
        # Draw text in the padded area
        draw = ImageDraw.Draw(new_image)
        y_position = padding
        
        for i, line in enumerate(all_lines):
            if line.strip():  # Skip empty lines but preserve spacing
                # Choose text color and font based on content
                if i < len(header_lines):
                    # Header with correctness info
                    text_color = (100, 255, 100) if is_correct else (255, 100, 100)  # Green for correct, red for incorrect
                    current_font = bold_font
                elif "Q:" in line:
                    text_color = (255, 255, 150)  # Light yellow for questions
                    current_font = font
                elif "Ground Truth:" in line:
                    text_color = (150, 255, 150)  # Light green for ground truth
                    current_font = font
                elif "VLM Response:" in line:
                    text_color = (150, 200, 255)  # Light blue for VLM response
                    current_font = font
                else:
                    # Choose color based on modality for other lines
                    if modality == "depth":
                        text_color = (255, 255, 100)  # Bright yellow for depth
                    elif modality == "normal":
                        text_color = (100, 255, 255)  # Cyan for normals
                    elif modality == "albedo":
                        text_color = (255, 100, 255)  # Magenta for albedo
                    elif modality == "irradiance":
                        text_color = (255, 255, 255)  # White for irradiance
                    else:
                        text_color = (255, 255, 255)  # Default white
                    current_font = font
                
                draw.text((8, y_position), line, fill=text_color, font=current_font)
            
            y_position += line_height
        
        return new_image, text_area_height
        
    except Exception as e:
        print(f"Error adding debug text: {e}")
        return image, 0  # Return original image with no offset if text overlay fails

def _save_debug_image(result, sample, image_path, debug_dir):
    """Save debug image with points, question, ground truth, and VLM response"""
    try:
        # Load original image
        rgb_image = Image.open(image_path).convert('RGB')
        
        # Get point coordinates from metadata
        point_coords = sample['metadata'].get('point_coordinates', [(100, 100), (200, 200)])  # Fallback coordinates
        
        # Get colors used from metadata
        colors_used = sample['metadata'].get('colors_used', {'color1': [255, 0, 0], 'color2': [0, 255, 0]})
        color1 = tuple(colors_used['color1']) if isinstance(colors_used['color1'], list) else (255, 0, 0)
        color2 = tuple(colors_used['color2']) if isinstance(colors_used['color2'], list) else (0, 255, 0)
        
        # Draw points on the image
        rgb_with_points = _draw_points_pytorch(rgb_image, point_coords, color1, color2)
        
        # Add debug text with comparison
        debug_image, text_area_height = _add_debug_text_with_comparison(
            rgb_with_points,
            result['question'],
            result['ground_truth_answer'],
            result['response'],
            result['predicted'],
            result['ground_truth'],
            result['modality'],
            result['correct']
        )
        
        # Generate filename
        sample_id = result['sample_id']
        correctness = "correct" if result['correct'] else "incorrect"
        debug_filename = f"{sample_id}_{correctness}_debug.png"
        debug_path = os.path.join(debug_dir, debug_filename)
        
        # Save debug image
        debug_image.save(debug_path, quality=95)
        
        return debug_path
        
    except Exception as e:
        print(f"Error saving debug image for {result['sample_id']}: {e}")
        return None

def load_dataset(jsonl_path, modalities, max_samples=None):
    """Load the generated dataset from JSONL file, filtering by modalities."""
    samples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            sample = json.loads(line.strip())
            # Filter by modality
            if sample['metadata']['modality'] in modalities:
                samples.append(sample)
                if max_samples and len(samples) >= max_samples:
                    break
    return samples

def get_ground_truth_label(sample):
    """Extract ground truth label from sample metadata."""
    comparison_result = sample['metadata']['comparison_result']
    modality = sample['metadata']['modality']
    
    # Handle different comparison result types
    if comparison_result in ["same", "different"]:
        return comparison_result
    elif comparison_result == "unknown":
        return "unknown"
    elif comparison_result in ["point1", "point2"]:
        # Need to map point1/point2 to red/green based on color assignment
        colors_used = sample['metadata'].get('colors_used', {})
        if colors_used:
            # point1 gets color1, point2 gets color2
            if comparison_result == "point1":
                return "red" if colors_used['color1'] == [255, 0, 0] else "green"
            else:  # point2
                return "red" if colors_used['color2'] == [255, 0, 0] else "green"
        else:
            # Fallback: assume standard mapping based on answer text
            answer = sample['conversations'][1]['value'].lower()
            if "red" in answer:
                return "red"
            elif "green" in answer:
                return "green"
            else:
                return "unknown"
    else:
        return "unknown"

def calculate_metrics(predictions, ground_truths, labels):
    """Calculate accuracy and F1 scores."""
    # Convert to numerical labels for sklearn
    label_to_num = {label: i for i, label in enumerate(labels)}
    
    pred_nums = [label_to_num.get(p, len(labels)) for p in predictions]
    gt_nums = [label_to_num.get(gt, len(labels)) for gt in ground_truths]
    
    # Filter out unknown predictions for accuracy calculation
    valid_indices = [i for i, p in enumerate(pred_nums) if p < len(labels)]
    
    if not valid_indices:
        return 0.0, 0.0, 0.0, 0.0
    
    valid_preds = [pred_nums[i] for i in valid_indices]
    valid_gts = [gt_nums[i] for i in valid_indices]
    
    # Accuracy
    accuracy = sum(p == gt for p, gt in zip(valid_preds, valid_gts)) / len(valid_preds)
    
    # F1 scores
    precision, recall, f1, support = precision_recall_fscore_support(
        valid_gts, valid_preds, labels=list(range(len(labels))), average=None, zero_division=0
    )
    
    macro_f1 = f1_score(valid_gts, valid_preds, labels=list(range(len(labels))), average='macro', zero_division=0)
    
    return accuracy, macro_f1, f1, support

def evaluate_model(model, tokenizer, dataset_samples, data_root, modalities, device='cuda', save_debug_images=False, debug_dir=None, max_debug_images=50):
    """
    Evaluate the fine-tuned model on the intrinsic judgment dataset.
    
    Args:
        model: Fine-tuned InternVL model
        tokenizer: Model tokenizer  
        dataset_samples: List of dataset samples from JSONL
        data_root: Root directory containing the images
        modalities: List of modalities to evaluate
        device: Device to run inference on
        save_debug_images: Whether to save debug images with VLM vs ground truth comparison
        debug_dir: Directory to save debug images (created if None)
        max_debug_images: Maximum number of debug images to save per modality
    
    Returns:
        dict: Evaluation results and metrics
    """
    results = []
    
    # Setup debug image saving if requested
    debug_image_counts = {}
    if save_debug_images:
        if debug_dir is None:
            debug_dir = "debug_evaluation_images"
        os.makedirs(debug_dir, exist_ok=True)
        print(f"Debug images will be saved to: {debug_dir}")
        for modality in modalities:
            debug_image_counts[modality] = 0
    
    # Track metrics by modality
    metrics_by_modality = {}
    for modality in modalities:
        metrics_by_modality[modality] = {
            'predictions': [],
            'ground_truths': [],
            'correct': 0,
            'total': 0,
            'parse_errors': 0
        }
    
    generation_config = dict(max_new_tokens=128, do_sample=False, temperature=0.0)
    
    print(f"Evaluating model on {len(dataset_samples)} samples across modalities: {modalities}")
    
    for i, sample in enumerate(tqdm(dataset_samples)):
        try:
            # Load image
            image_path = os.path.join(data_root, sample['image'])
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
                
            pixel_values = load_image(image_path, max_num=1).to(torch.bfloat16).to(device)
            
            # Get question from conversation
            question = sample['conversations'][0]['value'].replace('<image>\n', '')
            modality = sample['metadata']['modality']
            
            # Run model inference
            with torch.no_grad():
                response = model.chat(tokenizer, pixel_values, question, generation_config)
            
            # Parse model response
            predicted_judgment = parse_judgment(response, modality, question)
            
            # Get ground truth
            gt_judgment = get_ground_truth_label(sample)
            
            # Record result
            result = {
                'sample_id': sample['id'],
                'image_path': image_path,
                'modality': modality,
                'question': question,
                'response': response,
                'predicted': predicted_judgment,
                'ground_truth': gt_judgment,
                'ground_truth_answer': sample['conversations'][1]['value'],  # Full ground truth answer text
                'correct': predicted_judgment == gt_judgment,
                'distance_type': sample['metadata'].get('distance_type', 'unknown'),
                'dataset': sample['metadata']['dataset']
            }
            results.append(result)
            
            # Save debug image if requested
            if save_debug_images and debug_image_counts.get(modality, 0) < max_debug_images:
                debug_path = _save_debug_image(result, sample, image_path, debug_dir)
                if debug_path:
                    debug_image_counts[modality] += 1
                    if debug_image_counts[modality] == 1:  # Print only for first saved image per modality
                        print(f"Saving debug images for {modality} modality...")
            
            # Update modality-specific metrics
            if modality in metrics_by_modality:
                metrics_by_modality[modality]['predictions'].append(predicted_judgment)
                metrics_by_modality[modality]['ground_truths'].append(gt_judgment)
                metrics_by_modality[modality]['total'] += 1
                
                if predicted_judgment == "parse_error":
                    metrics_by_modality[modality]['parse_errors'] += 1
                elif predicted_judgment == gt_judgment:
                    metrics_by_modality[modality]['correct'] += 1
                    
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    # Calculate metrics for each modality
    final_metrics = {}
    for modality in modalities:
        if modality in metrics_by_modality:
            data = metrics_by_modality[modality]
            
            # Determine possible labels for this modality
            all_labels = list(set(data['predictions'] + data['ground_truths']))
            all_labels = [label for label in all_labels if label not in ['parse_error', 'unknown']]
            
            if len(all_labels) >= 2:  # Need at least 2 classes for meaningful metrics
                accuracy, macro_f1, f1_scores, support = calculate_metrics(
                    data['predictions'], data['ground_truths'], all_labels
                )
                
                final_metrics[modality] = {
                    'accuracy': accuracy,
                    'macro_f1': macro_f1,
                    'f1_scores': dict(zip(all_labels, f1_scores)),
                    'support': dict(zip(all_labels, support)),
                    'total_samples': data['total'],
                    'parse_errors': data['parse_errors'],
                    'labels': all_labels
                }
            else:
                final_metrics[modality] = {
                    'accuracy': 0.0,
                    'macro_f1': 0.0,
                    'f1_scores': {},
                    'support': {},
                    'total_samples': data['total'],
                    'parse_errors': data['parse_errors'],
                    'labels': all_labels
                }
    
    result_dict = {
        'results': results,
        'metrics_by_modality': final_metrics,
        'total_samples': len(results)
    }
    
    # Add debug image counts if debug images were saved
    if save_debug_images:
        result_dict['debug_image_counts'] = debug_image_counts
    
    return result_dict

def print_results(eval_results):
    """Print formatted evaluation results."""
    print(f"\nIntrinsic Judgment Evaluation Results:")
    print(f"Total Samples Evaluated: {eval_results['total_samples']}")
    
    print(f"\nResults by Modality:")
    for modality, metrics in eval_results['metrics_by_modality'].items():
        print(f"\n  {modality.upper()}")
        print(f"    Accuracy:      {metrics['accuracy']:.3f}")
        print(f"    Macro F1:      {metrics['macro_f1']:.3f}")
        print(f"    Total Samples: {metrics['total_samples']}")
        print(f"    Parse Errors:  {metrics['parse_errors']}")
        
        if metrics['f1_scores']:
            print(f"    F1 Scores by Class:")
            for label, f1 in metrics['f1_scores'].items():
                support = metrics['support'].get(label, 0)
                print(f"    {label:>10}: {f1:.3f} (n={support})")

def save_results(results_data, output_path):
    """Save detailed evaluation results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    print(f"Detailed results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned InternVL model on intrinsic judgment dataset')
    parser.add_argument('--model_path', default="pretrained/internvl_2_5_4b_intrinsic_judge", 
                       help='Path to fine-tuned (merged) InternVL model')
    parser.add_argument('--dataset_path', default="data/intrinsic_judgements/intrinsic_judgement_val.jsonl",
                       help='Path to dataset JSONL file')
    parser.add_argument('--data_root', default="data/intrinsic_judgements",
                       help='Root directory containing the dataset images')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Limit evaluation to this many samples per modality')
    parser.add_argument('--modalities', nargs='+', default=["depth", "normal", "irradiance", "albedo"],
                       choices=["depth", "normal", "irradiance", "albedo"],
                       help='Modalities to evaluate (default: all)')
    parser.add_argument('--output_results', default='intrinsic_evaluation_results.json',
                       help='Path to save detailed results JSON')
    parser.add_argument('--save_debug_images', action='store_true',
                       help='Save debug images showing VLM vs ground truth comparison')
    parser.add_argument('--debug_dir', default='debug_evaluation_images',
                       help='Directory to save debug images')
    parser.add_argument('--max_debug_images', type=int, default=50,
                       help='Maximum number of debug images to save per modality (default: 50)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    
    args = parser.parse_args()
    
    print("Loading fine-tuned InternVL intrinsic judgment model...")
    print(f"Model path: {args.model_path}")
    print(f"Evaluating modalities: {args.modalities}")
    
    # Load model
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True
    ).eval()
    
    if args.device == 'cuda':
        model = model.cuda()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    
    # Load and evaluate on dataset
    if not os.path.exists(args.dataset_path):
        print(f"Dataset file not found: {args.dataset_path}")
        return
    
    if not os.path.exists(args.data_root):
        print(f"Data root directory not found: {args.data_root}")
        return
    
    # Load dataset
    dataset_samples = load_dataset(args.dataset_path, args.modalities, args.max_samples)
    print(f"Loaded {len(dataset_samples)} samples from dataset")
    
    # Run evaluation
    results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        dataset_samples=dataset_samples,
        data_root=args.data_root,
        modalities=args.modalities,
        device=args.device,
        save_debug_images=args.save_debug_images,
        debug_dir=args.debug_dir,
        max_debug_images=args.max_debug_images
    )
    
    # Print and save results
    print_results(results)
    save_results(results, args.output_results)
    
    # Print debug image summary if enabled
    if args.save_debug_images and 'debug_image_counts' in results:
        print(f"\nDebug Images Summary:")
        print(f"Saved to: {args.debug_dir}")
        debug_counts = results['debug_image_counts']
        for modality in args.modalities:
            count = debug_counts.get(modality, 0)
            print(f"  {modality.capitalize()}: {count} images saved")
    
    print(f"\nEvaluation complete!")
    
    overall_accuracy = np.mean([m['accuracy'] for m in results['metrics_by_modality'].values()])
    overall_f1 = np.mean([m['macro_f1'] for m in results['metrics_by_modality'].values()])
    print(f"Overall Average Accuracy: {overall_accuracy:.3f}")
    print(f"Overall Average Macro F1: {overall_f1:.3f}")

if __name__ == "__main__":
    main()
