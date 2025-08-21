import cv2
import gc
import torch
import json
from pathlib import Path

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

def extract_regions_from_image(mask_generator, image_path):
    """
    Extract regions from a single image using SAM2
    
    Args:
        image_path: Path to the input image
        
    Returns:
        List of region dictionaries with bounding box coordinates
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return []

    height, width = image.shape[:2]
    image_area = height * width

    # Convert BGR to RGB for SAM
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Generate masks
    masks = mask_generator.generate(image_rgb)
    
    regions = []
    for i, mask_data in enumerate(masks):
        # Get bounding box from mask
        bbox = mask_data['bbox']  # [x, y, width, height]
        
        # Extract region parameters
        region_info = {
            'region_id': i,
            'x': int(bbox[0]),
            'y': int(bbox[1]),
            'width': int(bbox[2]),
            'height': int(bbox[3]),
            'area': int(mask_data['area'])
        }
        region_ratio = region_info['width'] / region_info['height'] if region_info['height'] > 0 else 0
        if region_ratio >= 10 or region_ratio <= 0.1:
            continue
        if region_info['area'] > image_area//6:
            continue

        regions.append(region_info)
    
    return regions

def process_image_folder(mask_generator,input_folder, output_file, image_extensions=None):
    """
    Process all images in a folder and save region parameters to a text file
    
    Args:
        input_folder: Path to folder containing images
        output_file: Path to output text file
        image_extensions: List of valid image extensions
    """
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    input_path = Path(input_folder)
    all_results = {}
    
    # Get all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for i, image_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {image_path.name}")
        
        regions = extract_regions_from_image(mask_generator, str(image_path))
        
        if regions:
            all_results[str(image_path.name)] = {
                'image_path': str(image_path),
                'num_regions': len(regions),
                'regions': regions
            }
            print(f"  Found {len(regions)} regions")
        else:
            print("  No regions found")
    
        # Save results to file
        save_results(all_results, output_file)
        print(f"\nResults saved to {output_file}")
        torch.cuda.empty_cache()
        gc.collect()
    
    return all_results
    
def save_results(results, output_file):
    """
    Save results to both JSON and human-readable text formats
    
    Args:
        results: Dictionary containing all results
        output_file: Path to output file (without extension)
    """
    output_path = Path(output_file)
    
    # Save as JSON for programmatic use
    json_file = output_path.with_suffix('.json')
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)

def main():
    """
    Example usage
    """

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    checkpoint_path = "checkpoints/sam_vit_b_01ec64.pth"
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=64,
        points_per_batch=64,
        pred_iou_thresh=0.70,
        stability_score_thresh=0.9,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=500,
    )
    
    # Process all images in a folder

    input_folder = "sam_detector/flat_images"  # Change this to your image folder
    output_file = "detected_regions"  # Will create .json and .txt files
    process_image_folder(mask_generator, input_folder, output_file)

if __name__ == "__main__":
    main()