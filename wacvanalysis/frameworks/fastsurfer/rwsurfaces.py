import os
import argparse
import nibabel as nib
import numpy as np
from nibabel.freesurfer.io import read_geometry, write_geometry
from stl import mesh

def read_surface(file_path):
    vertices, faces = read_geometry(file_path)
    return vertices, faces

def write_stl(vertices, faces, file_path):
    surface_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            surface_mesh.vectors[i][j] = vertices[f[j],:]
    surface_mesh.save(file_path)

def process_subject(subject_dir, dest_dir, subject):
    subject_path = os.path.join(subject_dir, subject, subject)
    dest_subject_path = os.path.join(dest_dir, subject)
    os.makedirs(dest_subject_path, exist_ok=True)
    
    surfaces = ['lh.pial', 'rh.pial', 'lh.white', 'rh.white']
    
    for surface in surfaces:
        surface_path = os.path.join(subject_path, 'surf', surface)
        if os.path.exists(surface_path):
            vertices, faces = read_surface(surface_path)
            
            # Write FreeSurfer format
            fs_file_path = os.path.join(dest_subject_path, surface)
            write_geometry(fs_file_path, vertices, faces)
            
            # Write STL format
            stl_file_path = os.path.join(dest_subject_path, surface + '.stl')
            write_stl(vertices, faces, stl_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process FastSurfer surfaces and convert to STL and FreeSurfer formats")
    parser.add_argument("--subject-dir", type=str, required=True, help="Path to FastSurfer subjects directory")
    parser.add_argument("--dest-dir", type=str, required=True, help="Path to destination subjects directory")
    parser.add_argument("--subject", type=str, required=True, help="Subject ID to process")
    
    args = parser.parse_args()
    
    process_subject(args.subject_dir, args.dest_dir, args.subject)
