import numpy as np
import pytorch3d
from pytorch3d.structures import Meshes
import trimesh
from trimesh.exchange.obj import export_obj
from scipy.spatial import cKDTree
import torch


import torch
from pytorch3d.structures import Meshes


def compute_normal_and_edges(v, f):
    """
    Compute normals and edges of a mesh, adding a batch dimension.

    Args:
    v (Tensor): Vertices of the mesh. Shape: [num_meshes, num_vertices, 3]
    f (Tensor): Faces of the mesh. Shape: [num_meshes, num_faces, 3]

    Returns:
    Tuple[Tensor, Tensor]: A tuple containing:
        - normals (Tensor): Normal vectors of the vertices with batch dimension. Shape: [num_meshes, num_vertices, 3]
        - edges (Tensor): Edges of the mesh with batch dimension. Shape: [num_meshes, num_edges, 2]
    """
    mesh = Meshes(verts=list(v), faces=list(f))
    normals = mesh.verts_normals_list()[0]
    edges = mesh.edges_packed()

    
    # Assertions to ensure that only one dimension is added
    assert normals.dim() + 1 == normals.unsqueeze(0).dim(), "More than one dimension added to normals"
    assert edges.dim() + 1 == edges.unsqueeze(0).dim(), "More than one dimension added to edges"

    normals = normals.unsqueeze(0)  # Adding batch dimension
    edges = edges.unsqueeze(0)      # Adding batch dimension
    edges = edges.transpose(1, 2)
    return normals, edges


def normal_similarity_loss_batched(normals, edges, epsilon=1e-6):
    """
    Calculate loss for encouraging normals of adjacent vertices to be parallel for batched input.
    Args:
        normals (Tensor): Normal vectors of the vertices with shape [batch_size, num_vertices, 3].
        edges (Tensor): Edges of the mesh with shape [batch_size, num_edges, 2].
        epsilon (float, optional): Small value to avoid division by zero in normalization. Defaults to 1e-6.
    Returns:
        Tensor: Mean loss over the batch.
    """

    batch_size = normals.shape[0]
    losses = []

    for b in range(batch_size):
        # For each edge, compute the dot product of the normals of its vertices
        normals_a = normals[b][edges[:, 0]]#Todo:add vertices
        normals_b = normals[b][edges[:, 1]]

        # Normalize the normals to ensure they are unit vectors
        normals_a = torch.nn.functional.normalize(normals_a, dim=1, eps=epsilon)
        normals_b = torch.nn.functional.normalize(normals_b, dim=1, eps=epsilon)

        # Use dot product to measure parallelism, and (1 - dot_product) as loss
        dot_products = torch.sum(normals_a * normals_b, dim=1)
        loss = torch.mean(1 - dot_products)

        losses.append(loss)

    # Calculate mean loss over the batch
    total_loss = torch.mean(torch.stack(losses))

    return total_loss



def getSubMesh(v, f, vertex_indices):
    # Ensure v and f are lists of tensors
    mesh = Meshes(verts=v, faces=f)

    # Determine the device from the input tensor
    device = v[0].device

    # Convert vertex_indices to a set for faster lookup
    vertex_indices_set = set(vertex_indices)

    # Get the faces of the original mesh
    faces = mesh.faces_list()[0]

    # Find faces where all three vertices are in the subset
    mask = [all(vertex.item() in vertex_indices_set for vertex in face) for face in faces]
    submesh_faces = faces[mask]

    # Create a mapping from old vertex indices to new ones
    new_vertex_indices = {old_idx: new_idx for new_idx, old_idx in enumerate(vertex_indices)}

    # Re-map vertex indices for the new faces and ensure it's on the same device
    remapped_faces = torch.tensor([[new_vertex_indices[vertex.item()] for vertex in face] for face in submesh_faces], device=device)

    # Extract the vertex positions for the submesh
    submesh_verts = v[0][vertex_indices]

    # Adjust dimensions to match the original mesh's structure
    sub_verts = submesh_verts.unsqueeze(0)  # shape: [1, num_verts, 3]
    sub_faces = remapped_faces.unsqueeze(0)  # shape: [1, num_faces, 3]

    return sub_verts, sub_faces

    # Example usage:
    # v and f are the vertices and faces of your original mesh as lists of tensors.
    # vertex_indices is a 1D tensor or list of indices for the vertices you want to include in the submesh.


def compute_normal(v, f):
    """v, f: Tensors"""
    normal = Meshes(verts=list(v),
                    faces=list(f)).verts_normals_list()[0]
    return normal.unsqueeze(0)


def save_mesh_obj(v, f, n, path):
    mesh_save = trimesh.Trimesh(vertices=v,
                                faces=f,
                                vertex_normals=n)
    obj_save  = export_obj(mesh_save, include_normals=True)
    with open(path, "w") as file:
        print(obj_save, file=file)


def compute_distance(v_pred, v_gt, f_pred, f_gt, n_samples=150000):
    """
    The results are evaluated based on three distances:
    1. Chamfer Distance (CD)
    2. Average Absolute Distance (AD)
    3. Hausdorff Distance (HD)

    Please see DeepCSR paper in details:
    https://arxiv.org/abs/2010.11423
    
    For original code, please see:
    https://bitbucket.csiro.au/projects/CRCPMAX/repos/deepcsr/browse/eval.py
    """
    
    # chamfer distance
    cd = 0
    kdtree = cKDTree(v_pred)
    cd += kdtree.query(v_gt)[0].mean()/2
    kdtree = cKDTree(v_gt)
    cd += kdtree.query(v_pred)[0].mean()/2

    # AD & HD
    mesh_pred = trimesh.Trimesh(vertices=v_pred, faces=f_pred)
    pts_pred = mesh_pred.sample(n_samples)
    mesh_gt = trimesh.Trimesh(vertices=v_gt, faces=f_gt)
    pts_gt = mesh_gt.sample(n_samples)

    _, P2G_dist, _ = trimesh.proximity.closest_point(mesh_pred, pts_gt)
    _, G2P_dist, _ = trimesh.proximity.closest_point(mesh_gt, pts_pred)

    # average absolute distance
    assd = ((P2G_dist.sum() + G2P_dist.sum()) / float(P2G_dist.size + G2P_dist.size))
    
    # Hausdorff distance
    hd = max(np.percentile(P2G_dist, 90), np.percentile(G2P_dist, 90))

    return cd, assd, hd


def compute_hausdorff(v_pred, v_gt, f_pred, f_gt, n_samples=150000):
    """
    The results are evaluated based on three distances:
    1. Chamfer Distance (CD)
    2. Average Absolute Distance (AD)
    3. Hausdorff Distance (HD)

    Please see DeepCSR paper in details:
    https://arxiv.org/abs/2010.11423
    
    For original code, please see:
    https://bitbucket.csiro.au/projects/CRCPMAX/repos/deepcsr/browse/eval.py
    """
    
    
    # AD & HD
    mesh_pred = trimesh.Trimesh(vertices=v_pred, faces=f_pred)
    pts_pred = mesh_pred.sample(n_samples)
    mesh_gt = trimesh.Trimesh(vertices=v_gt, faces=f_gt)
    pts_gt = mesh_gt.sample(n_samples)

    _, P2G_dist, _ = trimesh.proximity.closest_point(mesh_pred, pts_gt)
    _, G2P_dist, _ = trimesh.proximity.closest_point(mesh_gt, pts_pred)

    # Hausdorff distance
    hd = max(np.percentile(P2G_dist, 90), np.percentile(G2P_dist, 90))

    return hd
