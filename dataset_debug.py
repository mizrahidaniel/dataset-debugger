#!/usr/bin/env python3
"""
Dataset Debugger - Automatic dataset health checker
Detects label imbalance, outliers, duplicates, and potential mislabels.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from collections import Counter
from datetime import datetime
import hashlib

try:
    import numpy as np
    from sklearn.ensemble import IsolationForest
    from PIL import Image
    import imagehash
except ImportError:
    print("❌ Missing dependencies. Install with: pip install numpy scikit-learn pillow imagehash")
    sys.exit(1)


class DatasetDebugger:
    def __init__(self, data_path, labels_file=None, output_dir="report"):
        self.data_path = Path(data_path)
        self.labels_file = labels_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.images = []
        self.labels = {}
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "dataset_path": str(data_path),
            "total_samples": 0,
            "label_distribution": {},
            "outliers": [],
            "duplicates": [],
            "warnings": []
        }
    
    def load_data(self):
        """Load images and labels"""
        print(f"📂 Loading data from {self.data_path}")
        
        # Find all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        for img_path in self.data_path.rglob('*'):
            if img_path.suffix.lower() in image_extensions:
                self.images.append(img_path)
        
        self.results["total_samples"] = len(self.images)
        print(f"   Found {len(self.images)} images")
        
        # Load labels if provided
        if self.labels_file:
            self._load_labels()
    
    def _load_labels(self):
        """Load labels from CSV or JSON"""
        labels_path = Path(self.labels_file)
        
        if not labels_path.exists():
            print(f"⚠️  Labels file not found: {self.labels_file}")
            return
        
        # Simple CSV format: filename,label
        if labels_path.suffix == '.csv':
            with open(labels_path) as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        filename, label = parts[0], parts[1]
                        self.labels[filename] = label
        
        # JSON format: {"filename": "label"}
        elif labels_path.suffix == '.json':
            with open(labels_path) as f:
                self.labels = json.load(f)
        
        print(f"   Loaded {len(self.labels)} labels")
    
    def analyze_label_distribution(self):
        """Check for class imbalance"""
        print("\n📊 Analyzing label distribution...")
        
        if not self.labels:
            print("   ⚠️  No labels provided, skipping")
            return
        
        label_counts = Counter(self.labels.values())
        total = len(self.labels)
        
        distribution = {}
        for label, count in label_counts.most_common():
            percentage = (count / total) * 100
            distribution[label] = {
                "count": count,
                "percentage": round(percentage, 1)
            }
            
            # Flag severe imbalance
            if percentage > 70 or percentage < 5:
                self.results["warnings"].append(
                    f"⚠️  Class '{label}': {percentage:.1f}% - IMBALANCED"
                )
        
        self.results["label_distribution"] = distribution
        
        # Print summary
        print(f"\n   Label Distribution:")
        for label, stats in distribution.items():
            emoji = "⚠️" if stats["percentage"] > 70 or stats["percentage"] < 5 else "✅"
            print(f"   {emoji} {label}: {stats['count']} ({stats['percentage']}%)")
    
    def detect_outliers(self, contamination=0.05):
        """Use Isolation Forest to detect image outliers"""
        print(f"\n🚨 Detecting outliers (contamination={contamination})...")
        
        if len(self.images) < 10:
            print("   ⚠️  Too few images for outlier detection")
            return
        
        # Extract simple features: file size, dimensions, aspect ratio
        features = []
        valid_images = []
        
        for img_path in self.images:
            try:
                # File size
                file_size = img_path.stat().st_size
                
                # Image dimensions
                with Image.open(img_path) as img:
                    width, height = img.size
                    aspect_ratio = width / height if height > 0 else 1.0
                
                features.append([file_size, width, height, aspect_ratio])
                valid_images.append(img_path)
            except Exception as e:
                print(f"   ⚠️  Error processing {img_path.name}: {e}")
        
        if len(features) < 10:
            print("   ⚠️  Not enough valid images")
            return
        
        # Run Isolation Forest
        X = np.array(features)
        clf = IsolationForest(contamination=contamination, random_state=42)
        predictions = clf.fit_predict(X)
        
        # Find outliers (prediction = -1)
        outliers = []
        for i, pred in enumerate(predictions):
            if pred == -1:
                img_path = valid_images[i]
                outliers.append({
                    "file": str(img_path.relative_to(self.data_path)),
                    "size_bytes": features[i][0],
                    "dimensions": f"{int(features[i][1])}x{int(features[i][2])}",
                    "aspect_ratio": round(features[i][3], 2)
                })
        
        self.results["outliers"] = outliers
        print(f"   Found {len(outliers)} suspicious examples")
        
        if outliers:
            print(f"\n   Top outliers:")
            for outlier in outliers[:5]:
                print(f"   - {outlier['file']} ({outlier['dimensions']}, {outlier['size_bytes']} bytes)")
    
    def find_duplicates(self):
        """Find duplicate/near-duplicate images using perceptual hashing"""
        print("\n🔍 Finding duplicates...")
        
        if len(self.images) < 2:
            print("   ⚠️  Not enough images")
            return
        
        # Compute perceptual hashes
        hashes = {}
        for img_path in self.images:
            try:
                with Image.open(img_path) as img:
                    # Use average hash (fast, good for duplicates)
                    img_hash = imagehash.average_hash(img)
                    
                    # Store hash and path
                    hash_str = str(img_hash)
                    if hash_str not in hashes:
                        hashes[hash_str] = []
                    hashes[hash_str].append(str(img_path.relative_to(self.data_path)))
            except Exception as e:
                print(f"   ⚠️  Error hashing {img_path.name}: {e}")
        
        # Find duplicates (same hash = duplicate)
        duplicates = []
        for hash_val, paths in hashes.items():
            if len(paths) > 1:
                duplicates.append({
                    "hash": hash_val,
                    "count": len(paths),
                    "files": paths
                })
        
        self.results["duplicates"] = duplicates
        print(f"   Found {len(duplicates)} duplicate groups ({sum(d['count'] for d in duplicates)} total files)")
        
        if duplicates:
            print(f"\n   Duplicate groups:")
            for dup in duplicates[:3]:
                print(f"   - {dup['count']} files: {', '.join(dup['files'][:3])}...")
    
    def generate_report(self):
        """Generate markdown report"""
        print(f"\n📝 Generating report...")
        
        report_path = self.output_dir / "report.md"
        
        with open(report_path, 'w') as f:
            f.write("# 🔍 Dataset Health Report\n\n")
            f.write(f"**Generated:** {self.results['timestamp']}\n\n")
            f.write(f"**Dataset:** `{self.results['dataset_path']}`\n\n")
            f.write(f"**Total Samples:** {self.results['total_samples']}\n\n")
            
            # Warnings
            if self.results["warnings"]:
                f.write("## ⚠️ Warnings\n\n")
                for warning in self.results["warnings"]:
                    f.write(f"- {warning}\n")
                f.write("\n")
            
            # Label distribution
            if self.results["label_distribution"]:
                f.write("## 📊 Label Distribution\n\n")
                f.write("| Label | Count | Percentage |\n")
                f.write("|-------|-------|------------|\n")
                for label, stats in self.results["label_distribution"].items():
                    f.write(f"| {label} | {stats['count']} | {stats['percentage']}% |\n")
                f.write("\n")
            
            # Outliers
            if self.results["outliers"]:
                f.write(f"## 🚨 Outliers ({len(self.results['outliers'])} found)\n\n")
                f.write("| File | Dimensions | Size |\n")
                f.write("|------|------------|------|\n")
                for outlier in self.results["outliers"][:10]:
                    f.write(f"| `{outlier['file']}` | {outlier['dimensions']} | {outlier['size_bytes']} bytes |\n")
                f.write("\n")
            
            # Duplicates
            if self.results["duplicates"]:
                f.write(f"## 📈 Duplicates ({len(self.results['duplicates'])} groups)\n\n")
                for dup in self.results["duplicates"][:5]:
                    f.write(f"**Group {dup['hash'][:8]}...** ({dup['count']} files):\n")
                    for file in dup['files']:
                        f.write(f"- `{file}`\n")
                    f.write("\n")
        
        # Also save JSON
        json_path = self.output_dir / "report.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"   ✅ Report saved to: {report_path}")
        print(f"   ✅ JSON data saved to: {json_path}")
    
    def run(self):
        """Run full analysis pipeline"""
        print("🔍 Dataset Debugger - Automatic Health Check\n")
        print("=" * 60)
        
        self.load_data()
        self.analyze_label_distribution()
        self.detect_outliers()
        self.find_duplicates()
        self.generate_report()
        
        print("\n" + "=" * 60)
        print("✅ Analysis complete!\n")


def main():
    parser = argparse.ArgumentParser(
        description="Dataset Debugger - Automatic dataset health checker"
    )
    parser.add_argument(
        "data_path",
        help="Path to image dataset directory"
    )
    parser.add_argument(
        "--labels",
        help="Path to labels file (CSV or JSON)"
    )
    parser.add_argument(
        "--output",
        default="report",
        help="Output directory for report (default: report/)"
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Outlier detection contamination (default: 0.05)"
    )
    
    args = parser.parse_args()
    
    debugger = DatasetDebugger(
        data_path=args.data_path,
        labels_file=args.labels,
        output_dir=args.output
    )
    debugger.run()


if __name__ == "__main__":
    main()
