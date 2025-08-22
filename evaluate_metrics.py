import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pickle
from typing import Dict, List
import warnings
import random
warnings.filterwarnings('ignore')


class PatternBasedEncoderEvaluator:
    def __init__(self, all_regions_data: str, images_dir: str = None, num_queries: int = 80):
        """
        Initialize encoder evaluator using pattern pattern_ids as ground truth
        
        Args:
            all_regions_data: Path to your complete region data (contains pattern_ids)
            images_dir: Directory containing your original images
            num_queries: Number of random queries to evaluate (default: 80)
        """
        self.images_dir = Path(images_dir) if images_dir else None
        self.all_regions_data = all_regions_data
        self.num_queries = num_queries
        
        # Load region data with patterns
        self.regions_with_patterns = self.load_regions_with_patterns()
        self.pattern_groups = self.group_by_patterns()
        
        # Generate query set
        self.query_regions = self.generate_query_set()
        
        # Cache for loaded images and region crops
        self.image_cache = {}
        self.region_cache = {}
        
        # Initialize available encoders
        self.feature_extractors = self._initialize_encoders()
        
    def load_regions_with_patterns(self) -> Dict[str, int]:
        """Load all regions and their pattern pattern_ids"""
        with open(self.all_regions_data, "rb") as f:
            all_features = pickle.load(f)
        
        regions_patterns = {}
        for image_id, image_data in all_features["processed_images"].items():
            if "regions" in image_data:
                for region in image_data["regions"]:
                    clean_image_id = image_id.replace('_flat.png', '')
                    region_key = f"{clean_image_id}_{region['region_id']}"
                    
                    # Get pattern_id (pattern id)
                    pattern_id = region.get('pattern_id', None)
                    if pattern_id is not None:
                        regions_patterns[region_key] = pattern_id
        
        print(f"Loaded {len(regions_patterns)} regions with pattern labels")
        return regions_patterns
    
    def group_by_patterns(self) -> Dict[int, List[str]]:
        """Group regions by their pattern pattern_id"""
        pattern_groups = {}
        for region_id, pattern_id in self.regions_with_patterns.items():
            if pattern_id not in pattern_groups:
                pattern_groups[pattern_id] = []
            pattern_groups[pattern_id].append(region_id)
        
        print(f"Found {len(pattern_groups)} unique patterns:")
        for pattern_id, regions in pattern_groups.items():
            print(f"  Pattern {pattern_id}: {len(regions)} regions")
        
        return pattern_groups
    
    def generate_query_set(self) -> List[str]:
        """Generate a set of query regions, ensuring representation across patterns"""
        query_regions = []
        
        # Calculate queries per pattern (roughly equal distribution)
        patterns = list(self.pattern_groups.keys())
        queries_per_pattern = max(1, self.num_queries // len(patterns))
        remaining_queries = self.num_queries - (queries_per_pattern * len(patterns))
        
        for i, pattern_id in enumerate(patterns):
            # Number of queries for this pattern
            num_queries_this_pattern = queries_per_pattern
            if i < remaining_queries:  # Distribute remaining queries
                num_queries_this_pattern += 1
            
            # Randomly sample regions from this pattern
            pattern_regions = self.pattern_groups[pattern_id]
            sampled = random.sample(pattern_regions, 
                                  min(num_queries_this_pattern, len(pattern_regions)))
            query_regions.extend(sampled)
        
        # If we still need more queries, sample randomly from all regions
        if len(query_regions) < self.num_queries:
            remaining_regions = [r for r in self.regions_with_patterns.keys() 
                               if r not in query_regions]
            additional = random.sample(remaining_regions, 
                                     self.num_queries - len(query_regions))
            query_regions.extend(additional)
        
        print(f"Generated {len(query_regions)} query regions")
        return query_regions[:self.num_queries]  # Ensure exact number
    
    def _initialize_encoders(self) -> Dict:
        """Initialize all available encoders"""
        extractors = {}
        
        # Add DINO extractor
        try:
            #extractors['dino_v2'] = self.create_extractor('dinov2')
            print("✓ DINO v2 encoder loaded")
        except Exception as e:
            print(f"✗ Could not load DINO v2: {e}")
        
        # Add CLIP extractors
        try:
            #extractors['clip_vit_l14'] = self.create_extractor('clip')
            print("✓ CLIP encoders loaded")
        except Exception as e:
            print(f"✗ Could not load CLIP: {e}")

        # Add ResNet extractors
        try:
            #extractors['resnet50'] = self.create_extractor('resnet50')
            print("✓ ResNet50 encoder loaded")
        except Exception as e:
            print(f"✗ Could not load ResNet50: {e}")

        # Add iBOT extractors
        try:
            extractors['idoc_base'] = self.create_extractor('idoc_teacher')
            #extractors['ibot_scratch_horae'] = self.create_extractor('ibot_scratch_horae_teacher')
            #extractors['ibot_horae_lora_7e-3_30'] = self.create_extractor('ibot_horae_lora_7e-3_30')
            extractors['idoc_lora_7e-3_15'] = self.create_extractor('idoc_lora_7e-3_15')
            extractors['idoc_lora_7e-3_30'] = self.create_extractor('idoc_lora_7e-3_30')
            extractors['idoc_lora_7e-3_45'] = self.create_extractor('idoc_lora_7e-3_45')
            print("✓ iBOT encoders loaded")
        except Exception as e:
            print(f"✗ Could not load iBOT: {e}")

        print(f"Total encoders available: {len(extractors)}")
        return extractors

    def create_extractor(self, model_name=None):
        """Create feature extractor for specified model"""
        output_path = Path("annotated_output/")

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
    
    def calculate_average_precision(self, labels):
        """Calculate Average Precision from ranked labels"""
        precisions = []
        for idx, label in enumerate(labels):
            if label == 1:
                precision = labels[:idx + 1].count(label) / (idx + 1)
                precisions.append(precision)
    
        if precisions:
            return sum(precisions) / len(precisions)
        return 0.0

    def evaluate_query(self, query_region: str, encoder_name: str) -> float:
        """
        Evaluate a single query using pattern-based ground truth
        Positive: same pattern as query
        Negative: different pattern from query
        """
        query_image_id, query_region_id = query_region.split('_')
        query_pattern = self.regions_with_patterns[query_region]
        
        # Extract query features
        query_features = self.feature_extractors[encoder_name](query_image_id, query_region_id)
        query_norm = normalize(query_features.reshape(1, -1), norm='l2')

        similarities = []
        labels = []

        # Compare against all other regions
        for region_id, pattern_id in self.regions_with_patterns.items():
            # Skip the query region itself
            if region_id == query_region:
                continue

            target_image_id, target_region_id = region_id.split('_')
            target_features = self.feature_extractors[encoder_name](target_image_id, target_region_id)
            target_norm = normalize(target_features.reshape(1, -1), norm='l2')
            sim = cosine_similarity(query_norm, target_norm)[0][0]

            similarities.append(sim)
            # Label as positive if same pattern, negative otherwise
            labels.append(1 if pattern_id == query_pattern else 0)

        # Sort by similarity (descending)
        indices = np.argsort(similarities)[::-1]
        labels = [labels[i] for i in indices]

        # Calculate AP
        if sum(labels) == 0:  # No positives found
            return 0.0

        ap = self.calculate_average_precision(labels)
        return ap
    
    def evaluate_encoder(self, encoder_name: str) -> Dict:
        """Evaluate encoder using pattern-based ground truth"""
        print(f"Evaluating encoder: {encoder_name}")
        
        query_results = []
        
        for query_region in self.query_regions:
            try:
                ap = self.evaluate_query(query_region, encoder_name)
                query_pattern = self.regions_with_patterns[query_region]
                
                result = {
                    'query_region': query_region,
                    'query_pattern': query_pattern,
                    'AP': ap
                }
                query_results.append(result)
                
            except Exception as e:
                print(f"Error evaluating query {query_region}: {e}")
                continue
        
        # Calculate mean AP
        if not query_results:
            return {
                'encoder_name': encoder_name,
                'mAP': 0.0,
                'query_results': []
            }

        mean_ap = np.mean([r['AP'] for r in query_results])
        
        # Calculate per-pattern performance
        pattern_performance = {}
        for result in query_results:
            pattern = result['query_pattern']
            if pattern not in pattern_performance:
                pattern_performance[pattern] = []
            pattern_performance[pattern].append(result['AP'])
        
        pattern_stats = {pattern: {
            'mean_AP': np.mean(aps),
            'std_AP': np.std(aps),
            'count': len(aps)
        } for pattern, aps in pattern_performance.items()}

        return {
            'encoder_name': encoder_name,
            'mAP': mean_ap,
            'pattern_performance': pattern_stats,
            'query_results': query_results
        }
    
    def compare_encoders(self) -> pd.DataFrame:
        """Compare all available encoders using pattern-based evaluation"""
        all_results = []
        detailed_results = {}
        
        for encoder_name in self.feature_extractors.keys():
            print(f"\n{'='*50}")
            print(f"Evaluating: {encoder_name}")
            print('='*50)
            
            try:
                results = self.evaluate_encoder(encoder_name)
                detailed_results[encoder_name] = results
                
                all_results.append({
                    'encoder': encoder_name,
                    'mAP': results['mAP']
                })

                print(f"Results - mAP: {results['mAP']:.4f}")
                
                # Print per-pattern performance
                print("\nPer-pattern performance:")
                for pattern, stats in results['pattern_performance'].items():
                    print(f"  Pattern {pattern}: {stats['mean_AP']:.4f} ± {stats['std_AP']:.4f} ({stats['count']} queries)")

                # Save intermediate results
                results_df = pd.DataFrame(all_results)
                results_df.to_csv("pattern_encoder_comparison_temp.csv", index=False)

            except Exception as e:
                print(f"Error evaluating {encoder_name}: {e}")
                all_results.append({
                    'encoder': encoder_name,
                    'mAP': 0.0
                })
        
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('mAP', ascending=False)

        print(f"\n{'='*60}")
        print("FINAL ENCODER COMPARISON (Pattern-based)")
        print('='*60)
        print(results_df.to_string(index=False, float_format='%.4f'))
        
        # Save detailed results
        with open("detailed_pattern_results.pkl", "wb") as f:
            pickle.dump(detailed_results, f)
        
        return results_df
    
    def analyze_dataset_patterns(self):
        """Analyze the pattern distribution in your dataset"""
        print("\nPattern Distribution Analysis:")
        print(f"Total regions: {len(self.regions_with_patterns)}")
        print(f"Unique patterns: {len(self.pattern_groups)}")
        
        pattern_counts = {pattern: len(regions) for pattern, regions in self.pattern_groups.items()}
        
        print("\nPattern distribution:")
        for pattern in sorted(pattern_counts.keys()):
            count = pattern_counts[pattern]
            percentage = (count / len(self.regions_with_patterns)) * 100
            print(f"  Pattern {pattern}: {count} regions ({percentage:.2f}%)")
        
        print(f"\nQuery distribution:")
        query_patterns = [self.regions_with_patterns[q] for q in self.query_regions]
        query_pattern_counts = {}
        for pattern in query_patterns:
            query_pattern_counts[pattern] = query_pattern_counts.get(pattern, 0) + 1
        
        for pattern in sorted(query_pattern_counts.keys()):
            count = query_pattern_counts[pattern]
            percentage = (count / len(self.query_regions)) * 100
            print(f"  Pattern {pattern}: {count} queries ({percentage:.2f}%)")


def main_pattern_evaluation():
    """Main function to run pattern-based evaluation"""
    
    # Update this path to point to your all_features.pkl file that contains pattern_ids
    ALL_REGIONS_DATA = "annotated_output/all_features_idoc_teacher.pkl"
    IMAGES_DIR = "flat_images/"
    
    # Set random seed for reproducible results
    random.seed()
    
    # Initialize evaluator
    evaluator = PatternBasedEncoderEvaluator(
        all_regions_data=ALL_REGIONS_DATA,
        images_dir=IMAGES_DIR,
        num_queries=100  # Same as your manual evaluation
    )
    
    # Analyze dataset patterns
    evaluator.analyze_dataset_patterns()
    
    # Compare all encoders
    results_df = evaluator.compare_encoders()
    
    # Save final results
    results_df.to_csv("pattern_based_encoder_results.csv", index=False)
    print("\nFinal results saved to pattern_based_encoder_results.csv")

    return results_df

if __name__ == "__main__":
    main_pattern_evaluation()