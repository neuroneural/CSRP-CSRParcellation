import numpy as np
import os
import nibabel as nib
from plyfile import PlyData, PlyElement

def save_vertices_to_ply(vertices, file_path):
    vertex_array = np.array([(vertex[0], vertex[1], vertex[2]) for vertex in vertices],
                            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex_array, 'vertex')
    PlyData([el]).write(file_path)

def save_vertices_to_freesurfer(vertices, faces, file_path):
    # Create nibabel triangular mesh object
    mesh = nib.freesurfer.Surface(vertices, faces)
    # Save as FreeSurfer surface
    nib.freesurfer.io.write_geometry(file_path, vertices, faces)

def save_vertices_and_faces_to_stl(vertices, faces, file_path):
    # Convert vertices and faces to STL format
    stl_data = np.zeros(len(vertices), dtype=[('normals', np.float32, 3), ('vertices', np.float32, 3)])
    stl_data['vertices'] = vertices
    stl_data['normals'] = np.zeros((len(vertices), 3), dtype=np.float32)  # Dummy normals for STL
    stl_faces = np.array([(3, face) for face in faces], dtype=[('binary', np.uint8, 4), ('indices', np.uint32, (3,))])
    el = PlyElement.describe(stl_data, 'vertex', comments=['vertices'])
    el_faces = PlyElement.describe(stl_faces, 'face', comments=['faces'])
    PlyData([el, el_faces]).write(file_path)

def _createMedialWallPly(file_path):
    # Extract subject_id from directory structure
    subject_id = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
    print('subject_id', subject_id)

    # Extract hemisphere (hemi) from the file name
    file_name = os.path.basename(file_path)
    print('file_name', file_name)
    if file_name.startswith('lh'):
        hemi = 'lh'
    elif file_name.startswith('rh'):
        hemi = 'rh'
    else:
        raise ValueError("Hemisphere not recognized in the file name.")

    # Dynamically determine the subjects directory from the file_path
    subjects_dir = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))
    print('subjects_dir', subjects_dir)

    # Step 1: Read medial wall label directly from annotation
    annotation_file = os.path.join(subjects_dir, subject_id, 'label', f'{hemi}.aparc.a2009s.annot')
    annot = nib.freesurfer.read_annot(annotation_file)

    # Extract medial wall indices from the annotation
    medial_wall_label = np.where(annot[0] == annot[0].max())[0]

    # Step 2: Load surface data (pial and white surfaces)
    pial_path = os.path.join(subjects_dir, subject_id, 'surf', f'{hemi}.pial')
    pial_surface = nib.freesurfer.io.read_geometry(pial_path)
    white_path = os.path.join(subjects_dir, subject_id, 'surf', f'{hemi}.white')
    white_surface = nib.freesurfer.io.read_geometry(white_path)

    # Create binary mask for medial wall
    num_vertices = pial_surface[0].shape[0]
    medial_wall_mask = np.zeros(num_vertices, dtype=bool)
    medial_wall_mask[medial_wall_label] = True

    # Step 3: Extract vertices and faces associated with the medial wall
    vertices, faces = pial_surface
    mw_faces = [face for face in faces if np.any(medial_wall_mask[face])]
    unique_vertex_indices = np.unique(np.array(mw_faces).flatten())
    mw_vertices = vertices[unique_vertex_indices]

    # Save medial wall vertices to PLY, FreeSurfer, and STL formats
    ply_path = os.path.join(subjects_dir, subject_id, 'surf', f'{hemi}.pial.medial_wall.ply')
    freesurfer_path = os.path.join(subjects_dir, subject_id, 'surf', f'{hemi}.pial.medial_wall.surf')
    stl_path = os.path.join(subjects_dir, subject_id, 'surf', f'{hemi}.pial.medial_wall.stl')

    save_vertices_to_ply(mw_vertices, ply_path)
    save_vertices_to_freesurfer(mw_vertices, mw_faces, freesurfer_path)
    save_vertices_and_faces_to_stl(mw_vertices, mw_faces, stl_path)

    print(f"Medial wall vertices saved to {ply_path}, {freesurfer_path}, and {stl_path}")

    # Repeat the process for the white surface
    vertices, faces = white_surface
    mw_faces = [face for face in faces if np.any(medial_wall_mask[face])]
    unique_vertex_indices = np.unique(np.array(mw_faces).flatten())
    mw_vertices = vertices[unique_vertex_indices]

    ply_path = os.path.join(subjects_dir, subject_id, 'surf', f'{hemi}.white.medial_wall.ply')
    freesurfer_path = os.path.join(subjects_dir, subject_id, 'surf', f'{hemi}.white.medial_wall.surf')
    stl_path = os.path.join(subjects_dir, subject_id, 'surf', f'{hemi}.white.medial_wall.stl')

    save_vertices_to_ply(mw_vertices, ply_path)
    save_vertices_to_freesurfer(mw_vertices, mw_faces, freesurfer_path)
    save_vertices_and_faces_to_stl(mw_vertices, mw_faces, stl_path)

    print(f"Medial wall vertices saved to {ply_path}, {freesurfer_path}, and {stl_path}")
