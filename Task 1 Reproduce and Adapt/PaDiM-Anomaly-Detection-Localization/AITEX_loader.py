import os
import shutil
import random
import re

# --- Configuration ---
# !!! MODIFY THESE PATHS !!!
SOURCE_BASE_DIR = r'C:\Users\tsatt\Downloads\repo\PADIM\New folder' # Directory containing NODefect_images, Defect_images, Mask_images
TARGET_BASE_DIR = r'C:\Users\tsatt\Downloads\repo\PADIM\AITEX_Simplified' # Where the organized dataset will be created

# Source folder names (relative to SOURCE_BASE_DIR)
NO_DEFECT_FOLDER = 'NODefect_images'
DEFECT_FOLDER = 'Defect_images'
MASK_FOLDER = 'Mask_images'

# Target folder names (relative to TARGET_BASE_DIR)
TRAIN_GOOD_FOLDER = os.path.join('train', 'good')
TEST_GOOD_FOLDER = os.path.join('test', 'good')
TEST_BAD_FOLDER = os.path.join('test', 'bad')
GROUND_TRUTH_BAD_FOLDER = os.path.join('ground_truth', 'bad')

# Train/Test Split Ratio for the 'good' (NoDefect) images
# 0.8 means 80% for training, 20% for testing
TRAIN_RATIO = 0.8

# --- End Configuration ---

# --- Helper Function ---
def create_dir_if_not_exists(dir_path):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")
    else:
        print(f"Directory already exists: {dir_path}")

# --- Main Script Logic ---

print("Starting dataset organization...")

# 1. Define Full Source Paths
source_no_defect_dir = os.path.join(SOURCE_BASE_DIR, NO_DEFECT_FOLDER)
source_defect_dir = os.path.join(SOURCE_BASE_DIR, DEFECT_FOLDER)
source_mask_dir = os.path.join(SOURCE_BASE_DIR, MASK_FOLDER)

# 2. Define Full Target Paths
target_train_good_dir = os.path.join(TARGET_BASE_DIR, TRAIN_GOOD_FOLDER)
target_test_good_dir = os.path.join(TARGET_BASE_DIR, TEST_GOOD_FOLDER)
target_test_bad_dir = os.path.join(TARGET_BASE_DIR, TEST_BAD_FOLDER)
target_gt_bad_dir = os.path.join(TARGET_BASE_DIR, GROUND_TRUTH_BAD_FOLDER)

# 3. Create Target Directories
print("\nCreating target directories...")
create_dir_if_not_exists(target_train_good_dir)
create_dir_if_not_exists(target_test_good_dir)
create_dir_if_not_exists(target_test_bad_dir)
create_dir_if_not_exists(target_gt_bad_dir)

# 4. Process NoDefect (Good) Images
print("\nProcessing NoDefect images...")
good_images = []
# Walk through all subdirectories in the NODefect_images source folder
for root, _, files in os.walk(source_no_defect_dir):
    for filename in files:
        # Check if it's a PNG file and follows the NoDefect naming convention (_000_)
        if filename.lower().endswith('.png') and '_000_' in filename:
            full_path = os.path.join(root, filename)
            good_images.append(full_path)

print(f"Found {len(good_images)} 'good' (NoDefect) images.")

# Shuffle and split
random.shuffle(good_images)
split_index = int(len(good_images) * TRAIN_RATIO)
train_good_files = good_images[:split_index]
test_good_files = good_images[split_index:]

print(f"Splitting into {len(train_good_files)} training and {len(test_good_files)} testing 'good' images.")

# Copy 'good' images to train and test folders
print("Copying 'good' images...")
for src_path in train_good_files:
    filename = os.path.basename(src_path)
    dst_path = os.path.join(target_train_good_dir, filename)
    shutil.copy2(src_path, dst_path) # copy2 preserves metadata

for src_path in test_good_files:
    filename = os.path.basename(src_path)
    dst_path = os.path.join(target_test_good_dir, filename)
    shutil.copy2(src_path, dst_path)

print("'Good' image copying complete.")

# 5. Process Defect (Bad) Images
print("\nProcessing Defect images...")
bad_image_count = 0
if os.path.exists(source_defect_dir):
    for filename in os.listdir(source_defect_dir):
        # Assuming all png images in this folder are defect images
        if filename.lower().endswith('.png'):
            # Double-check it's not accidentally a 'good' image if naming was inconsistent
            if '_000_' not in filename:
                src_path = os.path.join(source_defect_dir, filename)
                dst_path = os.path.join(target_test_bad_dir, filename)
                shutil.copy2(src_path, dst_path)
                bad_image_count += 1
            else:
                 print(f"Warning: Found image '{filename}' with '_000_' in Defect folder. Skipping.")
    print(f"Copied {bad_image_count} 'bad' (Defect) images to {target_test_bad_dir}.")
else:
    print(f"Warning: Source Defect directory not found: {source_defect_dir}")

# 6. Process Mask Images (Ground Truth)
print("\nProcessing Mask images...")
mask_count = 0
if os.path.exists(source_mask_dir):
    for filename in os.listdir(source_mask_dir):
        if filename.lower().endswith('.png') and '_mask' in filename.lower():
            src_path = os.path.join(source_mask_dir, filename)

            # --- Basic Mask Filename Handling ---
            # This copies the mask with its original name.
            # You might need to rename masks later depending on your model's requirements.
            # For example, removing "_mask1" or "_mask2" suffixes if needed.
            # Example: Check for problematic suffixes
            if filename.lower().endswith(('_mask1.png', '_mask2.png')):
                 print(f"  Warning: Found potential double mask: {filename}. Copied as is. Review needed.")
            # -------------------------------------

            dst_path = os.path.join(target_gt_bad_dir, filename)
            shutil.copy2(src_path, dst_path)
            mask_count += 1
    print(f"Copied {mask_count} mask images to {target_gt_bad_dir}.")
    print("  IMPORTANT: Please manually check mask filenames in the target directory.")
    print("  Ensure they correspond correctly to images in 'test/bad/'.")
    print("  You may need to rename or merge masks like '..._mask1.png', '..._mask2.png'.")
    print(f"  Also verify if masks exist for all images in 'test/bad/' (e.g., 0100_025_08.png might be missing one).")

else:
    print(f"Warning: Source Mask directory not found: {source_mask_dir}")


print("\n-------------------------------------")
print("Dataset organization script finished.")
print(f"Organized data is located in: {TARGET_BASE_DIR}")
print("-------------------------------------")
print("\nNEXT STEPS:")
print("1. Manually verify the contents of each folder, especially the masks in 'ground_truth/bad/'.")
print("2. Remember to implement image preprocessing (e.g., tiling/resizing 4096x256 images) in your data loader.")