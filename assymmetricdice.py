import numpy as np
from scipy.spatial import KDTree

def interpolate_labels(predicted_vertices, target_vertices, target_labels):
    """
    Interpolate labels from predicted vertices to target vertices using nearest neighbors.
    :param predicted_vertices: np.array of shape (N, 3), vertices of the predicted surface
    :param target_vertices: np.array of shape (M, 3), vertices of the target surface
    :param predicted_labels: np.array of shape (N,), labels of the predicted surface
    :return: np.array of shape (?,), interpolated labels for the target surface
    """
    assert target_vertices.shape[0] == target_labels.shape[0]
    assert target_vertices.shape[0] != 1
    tree = KDTree(target_vertices)
    _, indices = tree.query(predicted_vertices)
    interpolated_labels = target_labels[indices]
    return interpolated_labels


predicted_vertices = np.random.rand(1000, 3)  # Replace with actual predicted vertices
target_vertices = np.random.rand(1200, 3)  # Replace with actual target vertices
target_labels = np.random.randint(0, 5, 1200)  # Replace with actual predicted labels
predicted_labels = np.random.randint(0, 5, 1000)  # Replace with actual predicted labels

print('predicted_vertices.shape',predicted_vertices.shape)
print('target_vertices.shape',target_vertices.shape)
print('target_labels.shape',target_labels.shape)

interpolated_ground_truth_labels = interpolate_labels(predicted_vertices, target_vertices, target_labels)
print('interpolated_ground_truth_labels.shape',interpolated_ground_truth_labels.shape)
print('predicted_labels.shape',predicted_labels.shape)
assert predicted_labels.shape == interpolated_ground_truth_labels.shape