import pyvista as pv
import pickle as pkl
import numpy as np
from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree

from vtkmodules.vtkCommonDataModel import vtkIterativeClosestPointTransform
import os
import numpy as np
from sklearn.decomposition import PCA
import pyvista as pv

def save_mesh(mesh, file_path, file_format='vtk'):
    """
    Save a PyVista mesh to a file.

    Parameters:
    - mesh: The PyVista mesh object to be saved.
    - file_path: The path (including filename) where the mesh will be saved.
    - file_format: The format of the file. Default is 'vtk'. Other formats can be 'stl', 'obj', 'ply', etc.
    """
    if not isinstance(mesh, pv.core.pointset.PolyData):
        raise ValueError("The provided mesh is not a valid PyVista PolyData object.")
    
    # # Check if the mesh is valid
    # if not isinstance(mesh, pv.core.pointset.PointSet):
    #     raise ValueError("The provided mesh is not a valid PyVista mesh object.")

    # Check the file format and save accordingly
    if file_format == 'vtk':
        mesh.save(file_path, binary=True)
    elif file_format == 'stl':
        mesh.save(file_path, binary=True)
    elif file_format == 'obj':
        mesh.save(file_path, binary=False)  # OBJ is typically non-binary
    elif file_format == 'ply':
        mesh.save(file_path, binary=True)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

# Example usage
# my_mesh = pv.Sphere()  # Example: create a sphere mesh
# save_mesh(my_mesh, 'path_to_save/sphere_mesh.vtk', 'vtk')

def scaleToMatchBoundingBox(source, target):
    source_bounds = source.bounds
    target_bounds = target.bounds

    scale_factors = [(target_bounds[i+1] - target_bounds[i]) / (source_bounds[i+1] - source_bounds[i]) for i in range(0, len(source_bounds), 2)]

    scale_matrix = np.diag(scale_factors + [1])
    source_scaled = source.copy().transform(scale_matrix)
    return source_scaled, scale_matrix


# scale a mesh around its center of mass (in-place)
def scaleAmesh(mesh, scale):
    points = mesh.points - mesh.center_of_mass()
    points = scale*points
    points = points + mesh.center_of_mass()
    mesh.points = points
    return mesh


def withinBounds(points, bounds):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    xx = (points[:,0] > xmin)
    yy = ((points[:,1] > ymin) & (points[:,1] < ymax))
    zz = ((points[:,2] > zmin) & (points[:,2] < zmax))
    passing = np.prod(np.c_[xx,yy,zz], axis=1).astype('bool')
    return points[passing, :], passing

def cleanDebris(mesh):
    # Label the connected components
    labeled_mesh = mesh.connectivity(largest=False)

    # Check if 'RegionId' scalar is available; if not, label the components
    if 'RegionId' not in labeled_mesh.point_data:
        labeled_mesh = labeled_mesh.cell_data_to_point_data()

    # Extract the largest connected component
    largest_component = labeled_mesh.connectivity(largest=True)

    # Ensure the result is a PolyData object
    if not isinstance(largest_component, pv.PolyData):
        largest_component = pv.PolyData(largest_component)

    return largest_component



def minuspatch(meshA, patch, K=1):
    pmesh = pv.PolyData(patch)
    region = pmesh.delaunay_3d().extract_surface()
    scaleAmesh(region, 1.1)
    mp, passing = withinBounds(meshA.points, region.bounds)
    tree = KDTree(mp)
    nnidx = []
    for point in patch:
        distances, indices = tree.query(np.expand_dims(point,
            axis=0), K)
        nnidx.append({'idx': indices,
                      'dist': distances})
    nearest = np.array([x['idx'] for x in nnidx]).flatten()
    idxs = np.where(passing == True)[0][nearest]
    return cleanDebris(meshA.remove_points(idxs)[0])

def minuspatch_optimized(meshA, patch, K=1):
    # Calculate centroids of faces in the mesh
    faces = meshA.faces.reshape((-1, 4))[:, 1:4]
    centroids = np.mean(meshA.points[faces], axis=1)

    # Build KDTree using centroids
    kd_tree = cKDTree(centroids)

    # Find nearest neighbor faces for each point in the patch
    face_indices = set()
    for point in patch:
        _, indices = kd_tree.query(point, k=K)
        face_indices.update(indices)

    # Convert set to list for indexing
    face_indices_to_remove = list(face_indices)

    # Remove the faces from the mesh in a batch
    clean_mesh = meshA.copy()
    clean_mesh.remove_cells(face_indices_to_remove, inplace=True)
    #return clean_mesh
    if isinstance(clean_mesh, pv.UnstructuredGrid):
        clean_mesh = clean_mesh.cast_to_polydata()

    return cleanDebris(clean_mesh)

# Usage example
# meshA = pv.read('/path/to/mesh/file.stl')
# patch = load_ply_points('/path/to/patch/points.ply')
# optimized_mesh = minuspatch_optimized(meshA, patch, K=5)


##################3
def _alignMeshesAndGetMatrix(target, source, rigid=True):
    icp = vtkIterativeClosestPointTransform()
    icp.SetSource(source)
    icp.SetTarget(target)
    if rigid:
        icp.GetLandmarkTransform().SetModeToRigidBody()
    else:
        icp.GetLandmarkTransform().SetModeToSimilarity() 
    icp.SetMaximumNumberOfLandmarks(1000)
    icp.SetMaximumMeanDistance(.00001)
    icp.SetMaximumNumberOfIterations(5000)
    icp.CheckMeanDistanceOn()
    icp.StartByMatchingCentroidsOn()
    icp.Update()

    # Extract the transformation matrix
    vtk_matrix = icp.GetMatrix()
    matrix = np.array([vtk_matrix.GetElement(i, j) for i in range(4) for j in range(4)]).reshape(4, 4)
    return source.transform(vtk_matrix), matrix

def alignCentersAndGetMatrix(target, source):
    target_center = np.array(target.center_of_mass())
    source_center = np.array(source.center_of_mass())
    translation = target_center - source_center

    # Creating the translation matrix
    translation_matrix = np.identity(4)
    translation_matrix[:3, 3] = translation

    # Applying the translation
    translated_source = source.copy()
    translated_source.points = source.points + translation
    return translated_source, translation_matrix

def alignMeshes(target, source, scale=True):
    # Align centers and get translation matrix
    aligned_source, translation_matrix = alignCentersAndGetMatrix(target, source)

    # Perform ICP alignment and get transformation matrix
    if scale:
        aligned_source, icp_matrix = _alignMeshesAndGetMatrix(target, aligned_source, rigid=False)
    else:
        aligned_source, icp_matrix = _alignMeshesAndGetMatrix(target, aligned_source)

    # Order of transformations: First apply translation, then ICP
    return aligned_source, [translation_matrix, icp_matrix]


# Example of using the function
# target_mesh = ...
# source_mesh = ...
# aligned_mesh, transformations = alignMeshes(target_mesh, source_mesh)

# To apply these transformations:
# 1. Apply translation_matrix to the mesh
# 2. Apply icp_matrix to the mesh


