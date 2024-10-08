import numpy as np
import pickle
import nibabel as nib
import os
from plyfile import PlyData, PlyElement
import argparse

def create_binary_mask_from_annotation(annotation_data, num_vertices):
    mask = np.zeros(num_vertices, dtype=bool)
    mask[np.isin(annotation_data, [-1, 4])] = True
    return mask

# Function to save vertices to a pickle file
def save_vertices_to_pickle(vertices, file_path):
    with open(file_path, 'wb') as pkl_file:
        pickle.dump(vertices, pkl_file)

# Function to save vertices to a PLY file
def save_vertices_to_ply(vertices, file_path):
    vertex_array = np.array([(vertex[0], vertex[1], vertex[2]) for vertex in vertices],
                            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex_array, 'vertex')
    PlyData([el]).write(file_path)

# Function to save the full surface as PLY file
def save_full_surface_to_ply(vertices, faces, file_path):
    vertex_array = np.array([(vertex[0], vertex[1], vertex[2]) for vertex in vertices],
                            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    face_array = np.array([(face,) for face in faces],
                          dtype=[('vertex_indices', 'i4', (3,))])
    el_verts = PlyElement.describe(vertex_array, 'vertex')
    el_faces = PlyElement.describe(face_array, 'face')
    PlyData([el_verts, el_faces]).write(file_path)

# Function to create medial wall files
def createMedialWallFiles(subjects_dir, condition, hemi, surf_type, output_dir):
    if surf_type not in ['pial', 'white']:
        raise ValueError("Invalid surf_type. Must be 'pial' or 'white'.")

    # Step 1: Read the annotation file
    annot_path = os.path.join(subjects_dir, condition, 'wm_gt', hemi, f'{hemi}.{surf_type}.annot')
    if not os.path.exists(annot_path):
        raise FileNotFoundError(f"Annotation file not found: {annot_path}")

    # Load surface geometry
    surf_path = os.path.join(subjects_dir, condition, 'wm_gt', hemi, f'{hemi}.{surf_type}.surf')
    vertices, faces = nib.freesurfer.read_geometry(surf_path)
    annot_data, ctab, names = nib.freesurfer.read_annot(annot_path)

    # Step 2: Create a binary mask from the annotation data
    medial_wall_mask = create_binary_mask_from_annotation(annot_data, vertices.shape[0])

    # Step 3: Modify the surface by removing faces associated with medial wall
    mw_faces = [face for face in faces if np.any(medial_wall_mask[face])]

    # Extract unique vertices from mw_faces
    unique_vertex_indices = np.unique(np.array(mw_faces).flatten())
    mw_vertices = vertices[unique_vertex_indices]

    # Save medial wall vertices to pickle and PLY files
    os.makedirs(output_dir, exist_ok=True)
    pkl_path = os.path.join(output_dir, f'{condition}.{hemi}.{surf_type}.medial_wall.pkl')
    save_vertices_to_pickle(mw_vertices, pkl_path)
    print(f"Medial wall vertices saved to {pkl_path}")

    ply_path = os.path.join(output_dir, f'{condition}.{hemi}.{surf_type}.medial_wall.ply')
    save_vertices_to_ply(mw_vertices, ply_path)
    print(f"Medial wall vertices saved to {ply_path}")

    # Save the full surface to PLY file
    full_surface_ply_path = os.path.join(output_dir, f'{condition}.{hemi}.{surf_type}.full_surface.ply')
    save_full_surface_to_ply(vertices, faces, full_surface_ply_path)
    print(f"Full surface saved to {full_surface_ply_path}")

def main():
    parser = argparse.ArgumentParser(description='Create medial wall files.')
    parser.add_argument('--subjects_dir', required=True, help='Path to the subjects directory.')
    parser.add_argument('--conditions', nargs='+', required=True, help='List of conditions to process (e.g., a b cortexode).')
    parser.add_argument('--hemis', nargs='+', required=True, help='List of hemispheres to process (e.g., lh rh).')
    parser.add_argument('--output_dir', required=True, help='Output directory for medial wall files.')
    parser.add_argument('--surf_types', nargs='+', default=['pial', 'white'], help='List of surface types to process (default: pial white).')
    args = parser.parse_args()

    for condition in args.conditions:
        for hemi in args.hemis:
            for surf_type in args.surf_types:
                try:
                    createMedialWallFiles(args.subjects_dir, condition, hemi, surf_type, args.output_dir)
                except Exception as e:
                    print(f"Error processing condition {condition}, hemisphere {hemi}, surf_type {surf_type}: {e}")

if __name__ == "__main__":
    main()
