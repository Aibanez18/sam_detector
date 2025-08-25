import pickle
import random
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.metrics.pairwise import cosine_similarity
import json

# ----------------------
# Load precomputed data
# ----------------------
def load_features(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# Example format expected:
# annotated.pkl -> {image_id: [{"bbox": [x,y,w,h], "category_id": int, "feature": np.array}, ...]}
# sam.pkl       -> {image_id: [{"bbox": [x,y,w,h], "feature": np.array}, ...]}

encoders = ["idoc_lora_7e-3_30","ibot_horae_lora_7e-3_30","resnet50"]#"ibot_horae_base", "idoc_base", "clip", "dinov2",
SAMPLE_RATIO = 0.4
SCORE_THRESHOLD = 0.75

# Load COCO ground-truth (your expert annotations)
coco_gt = COCO("annotations.json")
with open("annotations.json", "r") as f:
    ann_json = json.load(f)

results = []

for encoder in encoders:
    annotated = load_features(f"annotated_output/all_features_{encoder}.pkl")["processed_images"]
    sam = load_features(f"sam_output/all_features_{encoder}.pkl")["processed_images"]

    detections = []

    for image_name, image_data in annotated.items():
        ann_regions = image_data["regions"]
        if image_name not in sam:
            continue

        image_ann_dict = next((image for image in ann_json["images"] if image["file_name"] == image_name), None)
        image_id = image_ann_dict["id"]

        # Random subsample of annotated regions
        random.seed(99)
        ann_sample = random.sample(ann_regions, 
                                   max(1, int(len(ann_regions) * SAMPLE_RATIO)))

        sam_regions = sam[image_name]["regions"]
        sam_feats = np.array([s["features"] for s in sam_regions])

        for ann in ann_sample:
            ann_feat = ann["features"].reshape(1, -1)
            sims = cosine_similarity(ann_feat, sam_feats)[0]

            # normalize cosine similarity to [0,1]
            sims = (sims + 1) / 2.0

            for sam_region, score in zip(sam_regions, sims):
                if score >= SCORE_THRESHOLD:
                    detections.append({
                        "image_id": image_id,
                        "category_id": ann["pattern_id"],
                        "bbox": sam_region["bbox"],
                        "score": float(score)
                    })

    with open("detections.json", "w") as f:
        json.dump(detections, f)

    # ----------------------
    # Evaluate with COCO
    # ----------------------
    coco_dt = coco_gt.loadRes("detections.json")
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    results.append({
        'encoder_name': encoder,
        'AP': float(coco_eval.stats[0]),      # AP @ IoU=0.50:0.95
        'AP50': float(coco_eval.stats[1]),    # AP @ IoU=0.50
        'AP75': float(coco_eval.stats[2]),    # AP @ IoU=0.75
        'AR1': float(coco_eval.stats[6]),     # AR @ max detections=1
        'AR10': float(coco_eval.stats[7]),    # AR @ max detections=10
        'AR100': float(coco_eval.stats[8]),   # AR @ max detections=100
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('AP', ascending=False)

results_df.to_csv("coco_style_encoder_results.csv", index=False)

print(f"\n{'='*60}")
print("FINAL ENCODER COMPARISON (COCO-style)")
print('='*60)
print(results_df[['encoder_name', 'AP', 'AP50', 'AP75', 'AR1', 'AR10', 'AR100']].to_string(index=False, float_format='%.4f'))