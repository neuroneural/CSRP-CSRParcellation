import os
import nibabel as nib
import numpy as np
import trimesh
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import cKDTree

# Function to calculate Dice Score
def calculate_dice_score(labels1, labels2):
    dice_scores = {}
    unique_labels = np.unique(labels1)
    
    for label in unique_labels:
        intersection = np.sum((labels1 == label) & (labels2 == label))
        size1 = np.sum(labels1 == label)
        size2 = np.sum(labels2 == label)
        dice_score = 2 * intersection / (size1 + size2) if (size1 + size2) > 0 else 0.0
        dice_scores[label] = dice_score
    
    return dice_scores

# Adjusted function to ensure labels match vertex counts
def load_freesurfer_surface_and_labels(surface_path, label_path):
    coords, faces = nib.freesurfer.read_geometry(surface_path)
    labels, _, _ = nib.freesurfer.read_annot(label_path)
    #print('debug',len(labels),coords.shape[0],surface_path,label_path)
    assert len(labels) == coords.shape[0]
    if len(labels) != coords.shape[0]:
        raise ValueError(f"Label and vertex count mismatch: labels {len(labels)}, vertices {coords.shape[0]}")
    mesh = trimesh.Trimesh(vertices=coords, faces=faces)
    #print("mesh shape",mesh.vertices.shape)
    assert mesh.vertices.shape == coords.shape
    return mesh, labels

# Function to calculate Dice Score for Parcellation using KD-Tree mapping
def calculate_dice_score_with_mapping(labels1, labels2, vertices1, vertices2):
    if len(labels1) != vertices1.shape[0]:
        raise ValueError(f"Mismatch between labels1 and vertices1: labels1 {len(labels1)}, vertices1 {vertices1.shape[0]}")
    if len(labels2) != vertices2.shape[0]:
        raise ValueError(f"Mismatch between labels2 and vertices2: labels2 {len(labels2)}, vertices2 {vertices2.shape[0]}")
    
    if len(labels1) > len(labels2):
        temp = labels1
        labels1 = labels2
        labels2 = temp
        temp = vertices1
        vertices1 = vertices2
        vertices2 = temp
    assert len(labels2) >= len(labels1), "swap should have ensured this"
    assert vertices2.shape[0] >= vertices1.shape[0], "swap should have ensured this"
    tree = cKDTree(vertices1)
    distances, indices = tree.query(vertices2, k=1)
    
    assert len(indices) == vertices2.shape[0]
    
    # Debugging the mapping process
    print(f'Min distance: {np.min(distances)}, Max distance: {np.max(distances)}, Mean distance: {np.mean(distances)}')

    # Check if indices are unique when vertices1 == vertices2
    if np.array_equal(vertices1, vertices2):
        unique_indices = np.unique(indices)
        if len(unique_indices) != len(indices):
            print(f"Warning: Non-unique indices found when vertices1 == vertices2. Unique indices: {len(unique_indices)}, Total indices: {len(indices)}")
        else:
            print("All indices are unique when vertices1 == vertices2.")
    mapped_labels2=np.zeros_like(labels2)
    for index,assignmentIndex in enumerate(indices):
        #print (a,b )
        mapped_labels2[index] =labels1[assignmentIndex]
    
    assert labels2.shape == mapped_labels2.shape
    assert np.array_equal(vertices1,vertices2) == np.array_equal(labels2,mapped_labels2)
    # if not np.array_equal(vertices1,vertices2):
    #     assert not np.array_equal(labels2,mapped_labels2)
    return calculate_dice_score(labels2, mapped_labels2)

# Paths to the required data
freesurfer_subject_path = '/data/users2/washbee/speedrun/cortexode-data-rp/test/201818'
fastsurfer_subject_path = '/data/users2/washbee/fastsurfer-output/test/201818/201818'

# Load FreeSurfer pial surfaces and labels
try:
    fs_lh_pial, fs_lh_labels = load_freesurfer_surface_and_labels(
        os.path.join(freesurfer_subject_path, 'surf', 'lh.pial'),
        os.path.join(freesurfer_subject_path, 'label', 'lh.aparc.DKTatlas40.annot')
    )
    
except ValueError as e:
    print(f"Error loading FreeSurfer left hemisphere: {e}")

try:
    fs_rh_pial, fs_rh_labels = load_freesurfer_surface_and_labels(
        os.path.join(freesurfer_subject_path, 'surf', 'rh.pial'),
        os.path.join(freesurfer_subject_path, 'label', 'rh.aparc.DKTatlas40.annot')
    )
except ValueError as e:
    print(f"Error loading FreeSurfer right hemisphere: {e}")

# Load FastSurfer pial surfaces and labels
try:
    fast_lh_pial, fast_lh_labels = load_freesurfer_surface_and_labels(
        os.path.join(fastsurfer_subject_path, 'surf', 'lh.pial'),
        os.path.join(fastsurfer_subject_path, 'label', 'lh.aparc.DKTatlas.mapped.annot')
    )
except ValueError as e:
    print(f"Error loading FastSurfer left hemisphere: {e}")

try:
    fast_rh_pial, fast_rh_labels = load_freesurfer_surface_and_labels(
        os.path.join(fastsurfer_subject_path, 'surf', 'rh.pial'),
        os.path.join(fastsurfer_subject_path, 'label', 'rh.aparc.DKTatlas.mapped.annot')
    )
except ValueError as e:
    print(f"Error loading FastSurfer right hemisphere: {e}")


# Compute Dice scores with mapping
try:
    print('fs_lh_labels len',len(fs_lh_labels))
    print('fast_lh_labels len',len(fast_lh_labels))
    print('fs_lh_pial shape',fs_lh_pial.vertices.shape)
    print('fast_lh_pial shape',fast_lh_pial.vertices.shape)
    lh_dice_scores = calculate_dice_score_with_mapping(fs_lh_labels, fast_lh_labels, fs_lh_pial.vertices, fast_lh_pial.vertices)
except ValueError as e:
    print(f"Error calculating left hemisphere Dice scores: {e}")

try:
    print('fs_rh_labels len',len(fs_rh_labels))
    print('fast_rh_labels len',len(fast_rh_labels))
    print('fs_rh_pial shape',fs_rh_pial.vertices.shape)
    print('fast_rh_pial shape',fast_rh_pial.vertices.shape)

    rh_dice_scores = calculate_dice_score_with_mapping(fs_rh_labels, fast_rh_labels, fs_rh_pial.vertices, fast_rh_pial.vertices)
except ValueError as e:
    print(f"Error calculating right hemisphere Dice scores: {e}")

# KD-Tree Self-Comparison test
try:
    lh_self_dice_scores_kdtree = calculate_dice_score_with_mapping(fs_lh_labels, fs_lh_labels, fs_lh_pial.vertices, fs_lh_pial.vertices)
except ValueError as e:
    print(f"Error calculating left hemisphere KD-Tree self-comparison Dice scores: {e}")

try:
    rh_self_dice_scores_kdtree = calculate_dice_score_with_mapping(fs_rh_labels, fs_rh_labels, fs_rh_pial.vertices, fs_rh_pial.vertices)
except ValueError as e:
    print(f"Error calculating right hemisphere KD-Tree self-comparison Dice scores: {e}")

# Print Dice scores in a tabular format
print("Left Hemisphere Dice Scores:")
print("{:<10} {:<10}".format('Label', 'Dice Score'))
for label, score in lh_dice_scores.items():
    print("{:<10} {:<10.4f}".format(label, score))

print("\nRight Hemisphere Dice Scores:")
print("{:<10} {:<10}".format('Label', 'Dice Score'))
for label, score in rh_dice_scores.items():
    print("{:<10} {:<10.4f}".format(label, score))

# Print self-comparison Dice scores in a tabular format using KD-Tree
print("\nLeft Hemisphere KD-Tree Self-Comparison Dice Scores:")
print("{:<10} {:<10}".format('Label', 'Dice Score'))
for label, score in lh_self_dice_scores_kdtree.items():
    print("{:<10} {:<10.4f}".format(label, score))

print("\nRight Hemisphere KD-Tree Self-Comparison Dice Scores:")
print("{:<10} {:<10}".format('Label', 'Dice Score'))
for label, score in rh_self_dice_scores_kdtree.items():
    print("{:<10} {:<10.4f}".format(label, score))
