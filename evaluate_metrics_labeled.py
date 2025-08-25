import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pickle
from typing import Dict, List, Set
import cv2
import warnings
warnings.filterwarnings('ignore')


class EnhancedSparseEncoderEvaluator:
    def __init__(self, labeled_data_dir: str, all_regions_data: str, images_dir: str = None):
        """
        Initialize encoder evaluator for sparse labeling scenario
        
        Args:
            labeled_data_dir: Directory containing CSV files from your labeling tool
            all_regions_data: Path to your complete region data (for getting all available regions)
            images_dir: Directory containing your original images (needed for CLIP and other vision models)
        """
        self.labeled_data_dir = Path(labeled_data_dir)
        self.images_dir = Path(images_dir) if images_dir else None
        self.all_regions_data = all_regions_data
        self.labeled_data = self.load_labeled_data()
        self.all_regions = self.load_all_regions(all_regions_data)
        
        # Cache for loaded images and region crops
        self.image_cache = {}
        self.region_cache = {}
        
        # Initialize available encoders
        self.feature_extractors = self._initialize_encoders()
        
    def load_labeled_data(self) -> pd.DataFrame:
        """Load all labeled CSV files and combine them"""
        all_data = []
        for csv_file in self.labeled_data_dir.glob("*.csv"):
            df = pd.read_csv(csv_file)
            all_data.append(df)
        
        if not all_data:
            raise ValueError("No CSV files found in labeled data directory")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Loaded {len(combined_df)} labeled pairs from {len(all_data)} query sets")
        return combined_df
    
    def load_all_regions(self, all_regions_path: str) -> Set[str]:
        """Load all available regions from your complete dataset"""
        with open(all_regions_path, "rb") as f:
            all_features = pickle.load(f)
        
        regions = set()
        for image_id, image_data in all_features["processed_images"].items():
            if "regions" in image_data:
                for region in image_data["regions"]:
                    clean_image_id = image_id.replace('_flat.png', '')
                    regions.add(f"{clean_image_id}_{region['region_id']}")
        
        print(f"Found {len(regions)} total regions in dataset")
        return regions
    
    def _initialize_encoders(self) -> Dict:
        """Initialize all available encoders"""
        extractors = {}
        
        # Add DINO extractor if you have it
        try:
            #extractors['dino_v2'] = self.create_extractor('dino_output')
            print("✓ DINO v2 encoder loaded")
        except Exception as e:
            print(f"✗ Could not load DINO v2: {e}")
        
        # Add CLIP extractors
        try:
            #extractors['clip_vit_l14'] = self.create_extractor('clip_output')
            print("✓ CLIP encoders loaded")
        except Exception as e:
            print(f"✗ Could not load CLIP: {e}")

        # Add ResNet extractors
        try:
            #extractors['resnet50'] = self.create_extractor('resnet50_output')
            print("✓ ResNet50 encoder loaded")
        except Exception as e:
            print(f"✗ Could not load ResNet50: {e}")

        # Add LoRA extractors
        try:
            extractors['idoc_base'] = self.create_extractor('idoc_teacher')
            #extractors['ibot_scratch_horae'] = self.create_extractor('ibot_scratch_horae_teacher')
            #extractors['ibot_horae_lora_7e-3_30'] = self.create_extractor('ibot_horae_lora_7e-3_30')
            extractors['idoc_lora_7e-3_15'] = self.create_extractor('idoc_lora_7e-3_15')
            extractors['idoc_lora_7e-3_30'] = self.create_extractor('idoc_lora_7e-3_30')
            extractors['idoc_lora_7e-3_45'] = self.create_extractor('idoc_lora_7e-3_45')

            print("✓ LoRA encoders loaded")
        except Exception as e:
            print(f"✗ Could not load LoRA: {e}")
        
        print(f"Total encoders available: {len(extractors)}")
        return extractors

    def create_extractor(self, model_name = None):
        """Create LoRA-based feature extractor"""
        output_path = Path(f"sam_detector/")

        if model_name:
            with open(output_path / f"all_features_{model_name}.pkl", "rb") as f:
                all_features = pickle.load(f)
        else:
            with open(output_path / "all_features.pkl", "rb") as f:
                all_features = pickle.load(f)

        def extract_features(image_id: str, region_id: str) -> np.ndarray:
            image_key = f"{image_id}_flat.png"
            if image_key in all_features["processed_images"]:
                regions = all_features["processed_images"][image_key].get("regions", [])
                for region in regions:
                    if region["region_id"] == int(region_id):
                        return np.array(region.get("features", []))
            raise ValueError(f"Features not found for {image_id}_{region_id}")

        return extract_features
    
    # Add your evaluation methods here (same as before)
    def calculate_average_precision(self, labels):
        precisions = []
        for idx, label in enumerate(labels):
            if label == 1:
                precision = labels[:idx + 1].count(label) / (idx + 1)
                precisions.append(precision)
    
        if precisions:
            return sum(precisions) / len(precisions)
        return 0.0

    def evaluate_query_labeled_only(self, query_data: pd.DataFrame, encoder_name: str) -> float:
        """Evaluate using only labeled data (original approach)"""
        query_id = query_data['query_id'].iloc[0]
        query_image_id, query_region_id = query_id.split('_')
        
        # Extract features
        query_features = self.feature_extractors[encoder_name](query_image_id, query_region_id)
        
        similarities = []
        labels = []
        
        for _, row in query_data.iterrows():
            target_image_id, target_region_id = row['region_id'].split('_')
            target_features = self.feature_extractors[encoder_name](target_image_id, target_region_id)
            
            # Compute similarity
            query_norm = normalize(query_features.reshape(1, -1), norm='l2')
            target_norm = normalize(target_features.reshape(1, -1), norm='l2')
            sim = cosine_similarity(query_norm, target_norm)[0][0]
            
            similarities.append(sim)
            labels.append(1 if row['label'] == 'positive' else 0)

        indices = np.argsort(similarities)[::-1]  # Sort by similarity descending
        labels = [labels[i] for i in indices]
        
        # Calculate metrics on labeled subset only
        if sum(labels) == 0:
            return 0.0

        ap = self.calculate_average_precision(labels)

        return ap
    
    def evaluate_query_full_pool(self, query_data: pd.DataFrame, encoder_name: str) -> float:
        """
        Evaluate against all available regions.
        All unlabeled regions are assumed to be negative.
        """
        query_id = query_data['query_id'].iloc[0]
        query_image_id, query_region_id = query_id.split('_')

        # Extract query features
        query_features = self.feature_extractors[encoder_name](query_image_id, query_region_id)
        query_norm = normalize(query_features.reshape(1, -1), norm='l2')

        # Get ground truth positives for this query
        positives = set(query_data.loc[query_data['label'] == 'positive', 'region_id'])

        similarities = []
        labels = []

        for region_id in self.all_regions:
            # Skip the query region itself
            if region_id == query_id:
                continue

            target_image_id, target_region_id = region_id.split('_')
            target_features = self.feature_extractors[encoder_name](target_image_id, target_region_id)
            target_norm = normalize(target_features.reshape(1, -1), norm='l2')
            sim = cosine_similarity(query_norm, target_norm)[0][0]

            similarities.append(sim)
            labels.append(1 if region_id in positives else 0)

        # Sort by similarity
        indices = np.argsort(similarities)[::-1]
        labels = [labels[i] for i in indices]

        # If no positives, AP is zero
        if sum(labels) == 0:
            return 0.0

        ap = self.calculate_average_precision(labels)

        return ap
    
    def evaluate_encoder(self, encoder_name: str) -> Dict:
        """Conservative evaluation: only use labeled data or assume negative for unlabeled"""
        print(f"Evaluating encoder: {encoder_name}")
        
        query_groups = self.labeled_data.groupby('query_id')
        query_results = []
        
        for query_id, query_data in query_groups:
            try:
                result = {}
                result['AP@Labeled'] = self.evaluate_query_labeled_only(query_data, encoder_name)
                result['AP@All'] = self.evaluate_query_full_pool(query_data, encoder_name)
                result['query_id'] = query_id
                query_results.append(result)
            except Exception as e:
                print(f"Error evaluating query {query_id}: {e}")
                continue
        
        # Aggregate results
        if not query_results:
            return {
                'encoder_name': encoder_name,
                'mAP@Labeled': 0.0,
                'mAP@All': 0.0,
                'query_results': []
            }

        mean_ap_labeled = np.mean([r['AP@Labeled'] for r in query_results])
        mean_ap_all = np.mean([r['AP@All'] for r in query_results])

        return {
            'encoder_name': encoder_name,
            'mAP@Labeled': mean_ap_labeled,
            'mAP@All': mean_ap_all,
            'query_results': query_results
        }
    
    def compare_encoders(self) -> pd.DataFrame:
        """Compare all available encoders using conservative evaluation"""
        all_results = []
        
        for encoder_name in self.feature_extractors.keys():
            print(f"\n{'='*50}")
            print(f"Evaluating: {encoder_name}")
            print('='*50)
            
            try:
                results = self.evaluate_encoder(encoder_name)
                all_results.append({
                    'encoder': encoder_name,
                    'mAP@Labeled': results['mAP@Labeled'],
                    'mAP@All': results['mAP@All']
                })

                print(f"Results - mAP: {results['mAP@Labeled']:.4f}/{results['mAP@All']:.4f}")

                results_df = pd.DataFrame(all_results)
                results_df.to_csv("encoder_comparison_results_temp.csv", index=False)
                print("\nResults saved to encoder_comparison_results_temp.csv")

            except Exception as e:
                print(f"Error evaluating {encoder_name}: {e}")
                all_results.append({
                    'encoder': encoder_name,
                    'mAP@Labeled': 0.0,
                    'mAP@All': 0.0
                })
        
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('mAP@Labeled', ascending=False)

        print(f"\n{'='*60}")
        print("FINAL ENCODER COMPARISON")
        print('='*60)
        print(results_df.to_string(index=False, float_format='%.4f'))
        
        return results_df
    
    def analyze_labeling_coverage(self):
        """Analyze how much of your dataset you've actually labeled"""
        total_queries = len(self.labeled_data['query_id'].unique())
        total_labeled_pairs = len(self.labeled_data)
        total_possible_pairs = len(self.all_regions) * total_queries
        
        coverage = total_labeled_pairs / total_possible_pairs

        positive_data = self.labeled_data[self.labeled_data['label'] == 'positive']
        print(positive_data['similarity'].mean())
        print(positive_data['similarity'].std())

        negative_data = self.labeled_data[self.labeled_data['label'] == 'negative']
        print(negative_data['similarity'].mean())
        print(negative_data['similarity'].std())
        
        print("\nLabeling Coverage Analysis:")
        print(f"Total regions in dataset: {len(self.all_regions)}")
        print(f"Total queries: {total_queries}")
        print(f"Total labeled pairs: {total_labeled_pairs}")
        print(f"Coverage: {coverage:.6f} ({coverage*100:.4f}%)")
        
        # Per-query statistics
        query_stats = self.labeled_data.groupby('query_id').agg({
            'label': ['count', lambda x: sum(x == 'positive'), lambda x: sum(x == 'negative')]
        }).round(2)
        
        query_stats.columns = ['total_labeled', 'positive', 'negative']
        print("\nPer-query labeling stats:")
        print(query_stats.describe())

def main_enhanced_evaluation():
    """Main function to run the enhanced evaluation"""
    
    LABELED_DATA_DIR = "sam-dino_detector/similarity_csv"
    ALL_REGIONS_DATA = "sam-dino_detector/sam_output/all_features_resnet50.pkl"
    IMAGES_DIR = "sam-dino_detector/flat_images"

    # Initialize evaluator
    evaluator = EnhancedSparseEncoderEvaluator(
        labeled_data_dir=LABELED_DATA_DIR,
        all_regions_data=ALL_REGIONS_DATA,
        images_dir=IMAGES_DIR
    )
    
    # Analyze labeling coverage
    evaluator.analyze_labeling_coverage()
    
    # Compare all encoders
    results_df = evaluator.compare_encoders()

    return results_df

if __name__ == "__main__":
    main_enhanced_evaluation()