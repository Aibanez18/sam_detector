import os
import shutil
import pickle

from pathlib import Path

def collect_pngs_and_save(root_dir, output_dir="data", pkl_name="file_paths.pkl"):
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith(".png"):
                src_path = Path(dirpath) / file
                
                # Avoid overwriting files with the same name
                image_id = src_path.parent.parent.name[:4]
                dest_path = output_dir / f"{image_id}_{src_path.stem[-4:]}{src_path.suffix}"

                shutil.copy2(src_path, dest_path)
                saved_paths.append(str(dest_path).replace('\\', '/'))

    # Save the list of saved paths to a .pkl file
    with open(pkl_name, "wb") as f:
        pickle.dump(saved_paths, f)

    print(f"Copied {len(saved_paths)} PNG files to '{output_dir}' and saved paths to '{pkl_name}'")

# Example usage:
collect_pngs_and_save("dino_output", "data", "file_paths.pkl")
