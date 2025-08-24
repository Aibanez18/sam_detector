import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pickle
from typing import Dict, List, Tuple
import warnings
import random
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import tempfile
import os

warnings.filterwarnings('ignore')


class COCOStyleEncoderEvaluator:
    def __init__(self, all_regions_data: str, ground_truth_annotations: str, 
                 images_dir: str = None, confidence_threshold: float = 0.5):
        """
        Initialize COCO-style evaluator
        
        Args:
            all_regions_data: Path to SAM region data with features
            ground_truth_annotations: Path to COCO format ground truth annotations
            images_dir: Directory containing images
            confidence_threshold: Threshold for converting similarities to confidence scores
        """
        self.images_dir = Path(images_dir) if images_dir else None
        self.all_regions_data = all_regions_data
        self.confidence_threshold = confidence_threshold
        
        # Load data
        self.sam_regions = self.load_sam_regions()
        self.ground_truth = self.load_ground_truth(ground_truth_annotations)
        
        # Initialize encoders
        self.feature_extractors = self._initialize_encoders()
        
    def load_sam_regions(self) -> Dict:
        """Load SAM regions with features"""
        with open(self.all_regions_data, "rb") as f:
            all_features = pickle.load(f)
        
        sam_regions = {}
        for image_id, image_data in all_features["processed_images"].items():
            clean_image_id = image_id.replace('_flat.png', '')
            if "regions" in image_data:
                sam_regions[clean_image_id] = []
                for region in image_data["regions"]:
                    region_data = {
                        'region_id': region['region_id'],
                        'bbox': region.get('bbox', []),  # [x, y, width, height]
                        'area': region.get('area', 0),
                        'features': np.array(region.get('features', [])),
                        'mask': region.get('mask', None)  # If available
                    }
                    sam_regions[clean_image_id].append(region_data)
        
        print(f"Loaded SAM regions for {len(sam_regions)} images")
        return sam_regions
    
    def load_ground_truth(self, gt_path: str) -> COCO:
        """Load ground truth in COCO format"""
        return COCO(gt_path)
    
    def _initialize_encoders(self) -> Dict:
        """Initialize all available encoders"""
        extractors = {}
        output_path = Path("D:\\Universidad\\Proyecto de Titulo\\sam-dino_detector\\annotated_output")
        
        encoder_configs = {
            'dino_v2': 'dinov2',
            'clip_vit_l14': 'clip', 
            'resnet50': 'resnet50',
            'idoc_base': 'idoc_teacher',
            'ibot_scratch_horae': 'ibot_scratch_horae_teacher',
            'ibot_horae_lora_7e-3_30': 'ibot_horae_lora_7e-3_30'
        }
        
        for name, model_name in encoder_configs.items():
            try:
                extractors[name] = self.create_extractor(model_name)
                print(f"✓ {name} encoder loaded")
            except Exception as e:
                print(f"✗ Could not load {name}: {e}")
        
        return extractors

    def create_extractor(self, model_name: str):
        """Create feature extractor for specified model"""
        output_path = Path("D:\\Universidad\\Proyecto de Titulo\\sam-dino_detector\\annotated_output")
        
        with open(output_path / f"all_features_{model_name}.pkl", "rb") as f:
            all_features = pickle.load(f)

        def extract_features(image_id: str, region_id: int) -> np.ndarray:
            image_key = f"{image_id}_flat.png"
            if image_key in all_features["processed_images"]:
                regions = all_features["processed_images"][image_key].get("regions", [])
                for region in regions:
                    if region["region_id"] == region_id:
                        return np.array(region.get("features", []))
            raise ValueError(f"Features not found for {image_id}_{region_id}")

        return extract_features
    
    def calculate_region_similarity(self, query_features: np.ndarray, 
                                  target_features: np.ndarray) -> float:
        """Calculate cosine similarity between region features"""
        query_norm = normalize(query_features.reshape(1, -1), norm='l2')
        target_norm = normalize(target_features.reshape(1, -1), norm='l2')
        return cosine_similarity(query_norm, target_norm)[0][0]
    
    def create_query_templates(self, category_id: int, num_templates: int = 5) -> List[np.ndarray]:
        """Create feature templates for a category from ground truth regions"""
        templates = []
        gt_regions = []
        
        # Find ground truth regions of this category
        for ann_id in self.ground_truth.getAnnIds(catIds=[category_id]):
            ann = self.ground_truth.loadAnns([ann_id])[0]
            image_info = self.ground_truth.loadImgs([ann['image_id']])[0]
            image_name = os.path.splitext(image_info['file_name'])[0]
            
            # Find corresponding SAM region (by IoU or closest bbox)
            if image_name in self.sam_regions:
                best_sam_region = self.find_matching_sam_region(ann, self.sam_regions[image_name])
                if best_sam_region and len(best_sam_region['features']) > 0:
                    gt_regions.append(best_sam_region['features'])
        
        # Sample templates
        if gt_regions:
            sampled_indices = random.sample(range(len(gt_regions)), 
                                          min(num_templates, len(gt_regions)))
            templates = [gt_regions[i] for i in sampled_indices]
        
        return templates
    
    def find_matching_sam_region(self, gt_annotation: Dict, sam_regions: List[Dict]) -> Dict:
        """Find SAM region that best matches ground truth annotation by IoU"""
        gt_bbox = gt_annotation['bbox']  # [x, y, width, height]
        best_region = None
        best_iou = 0.0
        
        for region in sam_regions:
            if not region['bbox']:
                continue
            
            iou = self.calculate_bbox_iou(gt_bbox, region['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_region = region
        
        return best_region if best_iou > 0.1 else None  # Minimum IoU threshold
    
    def calculate_bbox_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU between two bounding boxes in [x, y, width, height] format"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def generate_predictions(self, encoder_name: str, image_id: str, 
                           category_templates: Dict[int, List[np.ndarray]]) -> List[Dict]:
        """Generate predictions for an image using feature similarity"""
        predictions = []
        
        if image_id not in self.sam_regions:
            return predictions
        
        for region in self.sam_regions[image_id]:
            if len(region['features']) == 0:
                continue
            
            # Calculate similarity to each category template
            best_category = None
            best_score = 0.0
            
            for category_id, templates in category_templates.items():
                if not templates:
                    continue
                
                # Calculate average similarity to all templates in category
                similarities = []
                for template in templates:
                    try:
                        sim = self.calculate_region_similarity(region['features'], template)
                        similarities.append(sim)
                    except:
                        continue
                
                if similarities:
                    avg_similarity = np.mean(similarities)
                    if avg_similarity > best_score:
                        best_score = avg_similarity
                        best_category = category_id
            
            # Convert similarity to confidence score (0-1 range)
            confidence = max(0.0, min(1.0, (best_score + 1) / 2))  # Normalize [-1,1] to [0,1]
            
            if confidence >= self.confidence_threshold and best_category is not None:
                prediction = {
                    'image_id': int(self.ground_truth.imgs[list(self.ground_truth.imgs.keys())[0]]['id']),  # Get proper image ID
                    'category_id': int(best_category),
                    'bbox': region['bbox'],
                    'score': float(confidence),
                    'area': float(region['area'])
                }
                predictions.append(prediction)
        
        return predictions
    
    def evaluate_encoder_coco(self, encoder_name: str) -> Dict:
        """Evaluate encoder using COCO metrics"""
        print(f"Evaluating encoder: {encoder_name}")
        
        # Create category templates from ground truth
        category_ids = self.ground_truth.getCatIds()
        category_templates = {}
        
        for cat_id in category_ids:
            templates = self.create_query_templates(cat_id, num_templates=10)
            category_templates[cat_id] = templates
            print(f"Created {len(templates)} templates for category {cat_id}")
        
        # Generate predictions for all images
        all_predictions = []
        image_ids = list(self.ground_truth.imgs.keys())
        
        for img_id in image_ids:
            img_info = self.ground_truth.loadImgs([img_id])[0]
            image_name = os.path.splitext(img_info['file_name'])[0]
            
            predictions = self.generate_predictions(encoder_name, image_name, category_templates)
            
            # Update image_id to match COCO format
            for pred in predictions:
                pred['image_id'] = img_id
            
            all_predictions.extend(predictions)
        
        print(f"Generated {len(all_predictions)} predictions")
        
        if not all_predictions:
            return {
                'encoder_name': encoder_name,
                'AP': 0.0, 'AP50': 0.0, 'AP75': 0.0,
                'AR1': 0.0, 'AR10': 0.0, 'AR100': 0.0
            }
        
        # Save predictions in COCO format
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(all_predictions, f)
            pred_file = f.name
        
        try:
            # Load predictions and evaluate
            coco_pred = self.ground_truth.loadRes(pred_file)
            coco_eval = COCOeval(self.ground_truth, coco_pred, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Extract metrics
            results = {
                'encoder_name': encoder_name,
                'AP': float(coco_eval.stats[0]),      # AP @ IoU=0.50:0.95
                'AP50': float(coco_eval.stats[1]),    # AP @ IoU=0.50
                'AP75': float(coco_eval.stats[2]),    # AP @ IoU=0.75
                'APs': float(coco_eval.stats[3]),     # AP for small objects
                'APm': float(coco_eval.stats[4]),     # AP for medium objects
                'APl': float(coco_eval.stats[5]),     # AP for large objects
                'AR1': float(coco_eval.stats[6]),     # AR @ max detections=1
                'AR10': float(coco_eval.stats[7]),    # AR @ max detections=10
                'AR100': float(coco_eval.stats[8]),   # AR @ max detections=100
                'ARs': float(coco_eval.stats[9]),     # AR for small objects
                'ARm': float(coco_eval.stats[10]),    # AR for medium objects
                'ARl': float(coco_eval.stats[11])     # AR for large objects
            }
            
        finally:
            # Cleanup
            os.unlink(pred_file)
        
        return results
    
    def compare_encoders_coco(self) -> pd.DataFrame:
        """Compare all encoders using COCO metrics"""
        all_results = []
        
        for encoder_name in self.feature_extractors.keys():
            print(f"\n{'='*50}")
            print(f"Evaluating: {encoder_name}")
            print('='*50)
            
            try:
                results = self.evaluate_encoder_coco(encoder_name)
                all_results.append(results)
                
                print(f"Results:")
                print(f"  AP: {results['AP']:.4f}")
                print(f"  AP50: {results['AP50']:.4f}")
                print(f"  AP75: {results['AP75']:.4f}")
                print(f"  AR100: {results['AR100']:.4f}")
                
            except Exception as e:
                print(f"Error evaluating {encoder_name}: {e}")
                all_results.append({
                    'encoder_name': encoder_name,
                    'AP': 0.0, 'AP50': 0.0, 'AP75': 0.0,
                    'AR1': 0.0, 'AR10': 0.0, 'AR100': 0.0
                })
        
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('AP', ascending=False)
        
        print(f"\n{'='*60}")
        print("FINAL ENCODER COMPARISON (COCO-style)")
        print('='*60)
        print(results_df[['encoder_name', 'AP', 'AP50', 'AP75', 'AR100']].to_string(index=False, float_format='%.4f'))
        
        return results_df


def main_coco_evaluation():
    """Main function for COCO-style evaluation"""
    
    # Update these paths
    ALL_REGIONS_DATA = "D:\\Universidad\\Proyecto de Titulo\\sam-dino_detector\\annotated_output\\all_features_dinov2.pkl"
    GROUND_TRUTH_ANNOTATIONS = "path/to/your/ground_truth_coco_format.json"  # Your COCO format annotations
    IMAGES_DIR = "D:\\Universidad\\Proyecto de Titulo\\sam-dino_detector\\flat_images"
    
    random.seed(42)
    
    evaluator = COCOStyleEncoderEvaluator(
        all_regions_data=ALL_REGIONS_DATA,
        ground_truth_annotations=GROUND_TRUTH_ANNOTATIONS,
        images_dir=IMAGES_DIR,
        confidence_threshold=0.5
    )
    
    results_df = evaluator.compare_encoders_coco()
    results_df.to_csv("coco_style_encoder_results.csv", index=False)
    print("\nResults saved to coco_style_encoder_results.csv")
    
    return results_df


if __name__ == "__main__":
    main_coco_evaluation()