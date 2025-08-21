import pickle
import json
import numpy as np
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

class DatasetLabeler:
    def __init__(self, output_path, k=160):
        self.output_path = Path(output_path)
        self.k = k
        self.all_features = None
        self.current_query = None
        self.current_similarities = None
        self.current_labels = {}
        self.results = []
        self.current_page = 0
        self.items_per_page = 16
        self.load_data()
        self.setup_gui()
    
    def load_data(self):
        """Load all features and metadata"""
        print("Loading features...")
        with open(self.output_path / "all_features.pkl", "rb") as f:
            self.all_features = pickle.load(f)
        
        with open(self.output_path / "combined_metadata.json", "r") as f:
            self.metadata = json.load(f)
        
        print(f"Loaded features for {len(self.all_features['processed_images'])} images")
    
    def get_all_regions(self):
        """Get all available regions with their identifiers"""
        regions = []
        for image_id, image_data in self.all_features["processed_images"].items():
            if "regions" in image_data:
                for region in image_data["regions"]:
                    regions.append({
                        'image_id': image_id.replace('_flat.png', ''),
                        'region_id': region['region_id'],
                        'full_id': f"{image_id.replace('_flat.png', '')}_{region['region_id']}"
                    })
        return regions
    
    def get_region_features(self, image_id, region_id):
        """Get features for a specific region"""
        image_key = f"{image_id}_flat.png"
        if "features" in self.all_features["processed_images"][image_key]["regions"][int(region_id)]:
            return self.all_features["processed_images"][image_key]["regions"][int(region_id)]["features"]
        return None
    
    def get_region_thumbnail_path(self, image_id, region_id):
        """Get path to region thumbnail"""
        return self.output_path / f"{image_id}_flat" / "thumbnails" / f"region_{int(region_id):04d}.png"
    
    def compute_similarities(self, query_image_id, query_region_id, top_k=160):
        """Compute similarities between query and all other regions"""
        query_features = self.get_region_features(query_image_id, query_region_id)
        if query_features is None:
            raise ValueError(f"Query region {query_image_id}_{query_region_id} not found")
        
        query_features = np.array(query_features).reshape(1, -1)
        query_features = normalize(query_features, norm='l2')
        
        similarities = []
        all_regions = self.get_all_regions()
        
        print(f"Computing similarities for {len(all_regions)} regions...")
        
        for region in all_regions:
            # Skip the query region itself
            if region['image_id'] == query_image_id and region['region_id'] == query_region_id:
                continue
            
            features = self.get_region_features(region['image_id'], region['region_id'])
            if features is not None:
                features = np.array(features).reshape(1, -1)
                features = normalize(features, norm='l2')
                
                similarity = cosine_similarity(query_features, features)[0][0]
                similarities.append({
                    'image_id': region['image_id'],
                    'region_id': region['region_id'],
                    'full_id': region['full_id'],
                    'similarity': similarity,
                    'distance': 1 - similarity
                })
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def setup_gui(self):
        """Setup the GUI interface"""
        self.root = tk.Tk()
        self.root.title("Dataset Labeling Tool")
        self.root.geometry("1400x900")
        
        # Create main frames
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.content_frame = ttk.Frame(self.root)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.setup_control_panel()
        self.setup_content_panel()
    
    def setup_control_panel(self):
        """Setup control panel with query selection and navigation"""
        # Query selection
        query_frame = ttk.LabelFrame(self.control_frame, text="Query Selection")
        query_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(query_frame, text="Image ID:").grid(row=0, column=0, padx=5, pady=5)
        self.image_id_var = tk.StringVar()
        self.image_id_entry = ttk.Entry(query_frame, textvariable=self.image_id_var, width=15)
        self.image_id_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(query_frame, text="Region ID:").grid(row=0, column=2, padx=5, pady=5)
        self.region_id_var = tk.StringVar()
        self.region_id_entry = ttk.Entry(query_frame, textvariable=self.region_id_var, width=10)
        self.region_id_entry.grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Button(query_frame, text="Load Query", command=self.load_query).grid(row=0, column=4, padx=5, pady=5)
        ttk.Button(query_frame, text="Random Query", command=self.load_random_query).grid(row=0, column=5, padx=5, pady=5)
        
        # Navigation and control
        nav_frame = ttk.LabelFrame(self.control_frame, text="Navigation & Control")
        nav_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(nav_frame, text="Previous Page", command=self.prev_page).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(nav_frame, text="Next Page", command=self.next_page).grid(row=0, column=1, padx=5, pady=5)
        
        self.page_label = ttk.Label(nav_frame, text="Page: 0/0")
        self.page_label.grid(row=0, column=2, padx=20, pady=5)
        
        ttk.Button(nav_frame, text="Mark All Visible as Negative", command=self.mark_all_negative).grid(row=0, column=3, padx=5, pady=5)
        ttk.Button(nav_frame, text="Save Results", command=self.save_results).grid(row=0, column=4, padx=5, pady=5)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(nav_frame, textvariable=self.status_var)
        self.status_label.grid(row=0, column=5, padx=20, pady=5)
    
    def setup_content_panel(self):
        """Setup content panel with query image and similar regions"""
        # Query display
        self.query_frame = ttk.LabelFrame(self.content_frame, text="Query Region")
        self.query_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        self.query_label = ttk.Label(self.query_frame, text="No query selected")
        self.query_label.pack(padx=10, pady=10)
        
        # Similar regions display
        self.regions_frame = ttk.LabelFrame(self.content_frame, text="Similar Regions")
        self.regions_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5)
        
        # Create scrollable frame for regions
        self.canvas = tk.Canvas(self.regions_frame)
        self.scrollbar = ttk.Scrollbar(self.regions_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
    
    def load_query(self):
        """Load query based on user input"""
        image_id = self.image_id_var.get().strip()
        region_id = self.region_id_var.get().strip()
        
        if not image_id or not region_id:
            messagebox.showerror("Error", "Please enter both image ID and region ID")
            return
        
        try:
            self.current_query = {'image_id': image_id, 'region_id': region_id}
            self.current_similarities = self.compute_similarities(image_id, region_id, self.k)
            self.current_labels = {}
            self.current_page = 0
            
            self.update_query_display()
            self.update_regions_display()
            self.status_var.set(f"Loaded query {image_id}_{region_id} with {len(self.current_similarities)} similar regions")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load query: {str(e)}")
    
    def load_random_query(self):
        """Load a random query region"""
        all_regions = self.get_all_regions()
        if not all_regions:
            messagebox.showerror("Error", "No regions available")
            return
        
        import random
        random_region = random.choice(all_regions)
        
        self.image_id_var.set(random_region['image_id'])
        self.region_id_var.set(random_region['region_id'])
        self.load_query()
    
    def update_query_display(self):
        """Update query region display"""
        if not self.current_query:
            return
        
        try:
            thumbnail_path = self.get_region_thumbnail_path(
                self.current_query['image_id'], 
                self.current_query['region_id']
            )
            
            if thumbnail_path.exists():
                image = Image.open(thumbnail_path)
                image = image.resize((150, 150), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                
                self.query_label.configure(image=photo, text="")
                self.query_label.image = photo  # Keep a reference
            else:
                self.query_label.configure(image="", text=f"Query: {self.current_query['image_id']}_{self.current_query['region_id']}\n(Image not found)")
        except Exception as e:
            self.query_label.configure(image="", text=f"Query: {self.current_query['image_id']}_{self.current_query['region_id']}\n(Error loading image)")
    
    def update_regions_display(self):
        """Update similar regions display"""
        # Clear existing widgets
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        if not self.current_similarities:
            return
        
        # Calculate pagination
        total_items = len(self.current_similarities)
        total_pages = (total_items + self.items_per_page - 1) // self.items_per_page
        start_idx = self.current_page * self.items_per_page
        end_idx = min(start_idx + self.items_per_page, total_items)
        
        self.page_label.configure(text=f"Page: {self.current_page + 1}/{total_pages}")
        
        # Create grid of regions
        cols = 8
        current_similarities_page = self.current_similarities[start_idx:end_idx]
        
        for i, region in enumerate(current_similarities_page):
            row = i // cols
            col = i % cols
            
            region_frame = ttk.Frame(self.scrollable_frame)
            region_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            
            # Load thumbnail
            try:
                thumbnail_path = self.get_region_thumbnail_path(region['image_id'], region['region_id'])
                if thumbnail_path.exists():
                    image = Image.open(thumbnail_path)
                    image = image.resize((120, 120), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(image)
                    
                    img_label = ttk.Label(region_frame, image=photo)
                    img_label.image = photo  # Keep reference
                    img_label.pack()
                else:
                    img_label = ttk.Label(region_frame, text="No Image", width=15)
                    img_label.pack()
            except:
                img_label = ttk.Label(region_frame, text="Error", width=15)
                img_label.pack()
            
            # Region info
            info_text = f"{region['image_id']}_{region['region_id']}\nSim: {region['similarity']:.3f}"
            info_label = ttk.Label(region_frame, text=info_text, font=("Arial", 8))
            info_label.pack()
            
            # Buttons
            button_frame = ttk.Frame(region_frame)
            button_frame.pack()
            
            # Get current label
            current_label = self.current_labels.get(region['full_id'], 'negative')
            
            pos_btn = ttk.Button(button_frame, text="Positive", 
                               command=lambda r=region: self.label_region(r, 'positive'))
            pos_btn.pack(side=tk.LEFT, padx=2)
            
            neg_btn = ttk.Button(button_frame, text="Negative",
                               command=lambda r=region: self.label_region(r, 'negative'))
            neg_btn.pack(side=tk.LEFT, padx=2)
            
            # Highlight current selection
            if current_label == 'positive':
                pos_btn.configure(style='Positive.TButton')
            else:
                neg_btn.configure(style='Negative.TButton')
        
        # Configure grid weights
        for i in range(cols):
            self.scrollable_frame.columnconfigure(i, weight=1)
    
    def label_region(self, region, label):
        """Label a region as positive or negative"""
        self.current_labels[region['full_id']] = label
        self.update_regions_display()  # Refresh to show updated button states
    
    def mark_all_negative(self):
        """Mark all currently visible regions as negative"""
        if not self.current_similarities:
            return
        
        start_idx = self.current_page * self.items_per_page
        end_idx = min(start_idx + self.items_per_page, len(self.current_similarities))
        
        for region in self.current_similarities[start_idx:end_idx]:
            self.current_labels[region['full_id']] = 'negative'
        
        self.update_regions_display()
        self.status_var.set("Marked all visible regions as negative")
    
    def prev_page(self):
        """Go to previous page"""
        if self.current_page > 0:
            self.current_page -= 1
            self.update_regions_display()
    
    def next_page(self):
        """Go to next page"""
        if self.current_similarities:
            total_pages = (len(self.current_similarities) + self.items_per_page - 1) // self.items_per_page
            if self.current_page < total_pages - 1:
                self.current_page += 1
                self.update_regions_display()
    
    def save_results(self):
        """Save labeling results to CSV"""
        if not self.current_query or not self.current_similarities:
            messagebox.showerror("Error", "No query loaded")
            return
        
        # Prepare results
        results = []
        query_id = f"{self.current_query['image_id']}_{self.current_query['region_id']}"
        
        filename = query_id + ".csv"
        
        if not filename:
            return
        for region in self.current_similarities:
            region_id = f"{region['image_id']}_{region['region_id']}"
            label = self.current_labels.get(region['full_id'], 'negative')
            
            results.append({
                'query_id': query_id,
                'region_id': region_id,
                'label': label,
                'similarity': region['similarity'],
                'distance': region['distance']
            })
        
        # Save to CSV
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        
        messagebox.showinfo("Success", f"Results saved to {filename}")
        self.status_var.set(f"Saved {len(results)} results to CSV")
    
    def run(self):
        """Run the application"""
        # Configure button styles
        style = ttk.Style()
        style.configure('Positive.TButton', background='lightgreen')
        style.configure('Negative.TButton', background='lightcoral')
        
        self.root.mainloop()

def main():
    # Get the path to output folder
    output_path = "D:\\Universidad\\Proyecto de Titulo\\sam-dino_detector\\resnet50_output"
    if not os.path.exists(output_path):
        print(f"Error: Path {output_path} does not exist")
        return
    
    # Check if required files exist
    required_files = ["all_features.pkl", "combined_metadata.json"]
    for file in required_files:
        if not os.path.exists(os.path.join(output_path, file)):
            print(f"Error: Required file {file} not found in {output_path}")
            return
    
    # Create and run the labeler
    labeler = DatasetLabeler(output_path, k=80)
    labeler.run()

if __name__ == "__main__":
    main()