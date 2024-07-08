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
def createMedialWallFiles(subjects_dir, subject_id, hemi, output_dir, surf_type='pial'):
    if surf_type not in ['pial', 'white']:
        raise ValueError("Invalid surf_type. Must be 'pial' or 'white'.")
    
    # Step 1: Read the annotation file
    annot_path = os.path.join(subjects_dir, subject_id, 'label', f'{hemi}.aparc.DKTatlas40.annot')
    if not os.path.exists(annot_path):
        raise FileNotFoundError(f"Annotation file not found: {annot_path}")
    
    vertices, faces = nib.freesurfer.read_geometry(os.path.join(subjects_dir, subject_id, 'surf', f'{hemi}.{surf_type}'))
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
    pkl_path = os.path.join(output_dir, f'{subject_id}.{hemi}.{surf_type}.medial_wall.pkl')
    save_vertices_to_pickle(mw_vertices, pkl_path)
    print(f"Medial wall vertices saved to {pkl_path}")

    ply_path = os.path.join(output_dir, f'{subject_id}.{hemi}.{surf_type}.medial_wall.ply')
    save_vertices_to_ply(mw_vertices, ply_path)
    print(f"Medial wall vertices saved to {ply_path}")

    # Save the full surface to PLY file
    full_surface_ply_path = os.path.join(output_dir, f'{subject_id}.{hemi}.{surf_type}.full_surface.ply')
    save_full_surface_to_ply(vertices, faces, full_surface_ply_path)
    print(f"Full surface saved to {full_surface_ply_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create medial wall files.')
    parser.add_argument('--subjects_dir', required=True, help='Path to the subjects directory.')
    parser.add_argument('--subject_id', required=True, help='Subject ID.')
    parser.add_argument('--output_dir', required=True, help='Output directory.')
    
    args = parser.parse_args()
    
    createMedialWallFiles(args.subjects_dir, args.subject_id, 'rh', args.output_dir, surf_type='pial')
    createMedialWallFiles(args.subjects_dir, args.subject_id, 'rh', args.output_dir, surf_type='white')
    createMedialWallFiles(args.subjects_dir, args.subject_id, 'lh', args.output_dir, surf_type='pial')
    createMedialWallFiles(args.subjects_dir, args.subject_id, 'lh', args.output_dir, surf_type='white')
