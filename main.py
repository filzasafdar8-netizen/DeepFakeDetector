import os
from PIL import Image
import imagehash
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# Paths
# ------------------------------
IMAGE_FOLDER = "images/"
REFERENCE_FOLDER = os.path.join(IMAGE_FOLDER, "reference/")
REPORT_FILE = "report.csv"

# ------------------------------
# Load Reference Images
# ------------------------------
reference_hashes = {}
for ref_file in os.listdir(REFERENCE_FOLDER):
    if ref_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        ref_path = os.path.join(REFERENCE_FOLDER, ref_file)
        try:
            img = Image.open(ref_path).convert('RGB')
            reference_hashes[ref_file] = imagehash.phash(img)
        except Exception as e:
            print(f"Failed to load reference {ref_file}: {e}")

print(f"Loaded reference images: {list(reference_hashes.keys())}")

# ------------------------------
# Function to Compute Similarity
# ------------------------------
def compute_similarity(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        img_hash = imagehash.phash(img)
    except Exception as e:
        print(f"Failed to load test image {img_path}: {e}")
        return None, 0

    best_score = 0
    best_ref = None
    for ref_name, ref_hash in reference_hashes.items():
        diff = img_hash - ref_hash
        sim = max(0, 100 - diff*5)  # similarity %
        if sim > best_score:
            best_score = sim
            best_ref = ref_name
    return best_ref, float(best_score)

# ------------------------------
# Process Test Images
# ------------------------------
results = []

for file in os.listdir(IMAGE_FOLDER):
    file_path = os.path.join(IMAGE_FOLDER, file)
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        if file in reference_hashes:  # skip reference images
            continue
        ref_name, sim_score = compute_similarity(file_path)
        if sim_score >= 90:
            label = "Real"
        elif sim_score >= 70:
            label = "Suspicious"
        else:
            label = "Fake"
        results.append([file, label, sim_score])
        print(f"{file}: {label} ({sim_score:.2f}%) vs {ref_name}")

if not results:
    print("No test images found. Please check your 'images/' folder.")
    exit()

# ------------------------------
# Save CSV Report
# ------------------------------
df = pd.DataFrame(results, columns=["Image", "Prediction", "Confidence"])
df.to_csv(REPORT_FILE, index=False)
print(f"\nReport saved to {REPORT_FILE}")

# ------------------------------
# Visualization
# ------------------------------
labels = [r[0] for r in results]
confidences = [r[2] for r in results]
colors = [
    'green' if r[1].lower() == 'real' else
    'orange' if r[1].lower() == 'suspicious' else
    'red'
    for r in results
]

plt.figure(figsize=(8,5))
bars = plt.bar(range(len(labels)), confidences, color=colors)

plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
plt.ylabel("Confidence %")
plt.title("Deepfake / Manipulated Image Detection")

# Add % text on top of bars
for bar, conf in zip(bars, confidences):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{conf:.1f}%", ha='center', va='bottom')

plt.tight_layout()
plt.show()