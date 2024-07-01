import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.spatial import KDTree
from model.csrvertexclassification import CSRVCNet  # Ensure this matches your model import
from data.vc_dataloader2 import CSRVertexLabeledDataset  # Ensure this matches your data loader import
from util.mesh import compute_dice
from config import load_config


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

def evaluate_model(config):
    # Load configuration
    device = config.device
    model_dir = config.model_dir
    model_file = config.model_file
    num_classes = get_num_classes(config.atlas)

    # Initialize model
    csrvcnet = CSRVCNet(dim_h=config.dim_h, kernel_size=config.kernel_size, n_scale=config.n_scale,
                        gnn_layers=config.gnn_layers, use_gcn=(config.gnn == 'gcn'),
                        gat_heads=config.gat_heads, num_classes=num_classes,
                        use_pytorch3d=(config.use_pytorch3d_normal != 'no')).to(device)

    # Load trained model weights
    model_path = os.path.join(model_dir, model_file)
    csrvcnet.load_state_dict(torch.load(model_path, map_location=device))
    csrvcnet.eval()

    # Load validation dataset
    validset = CSRVertexLabeledDataset(config, 'valid')
    validloader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=4)

    # Evaluate the model
    all_dice_scores = []
    exclude_classes = [4] if config.atlas == 'aparc' else []  # Exclude non-cortex, include medial wall

    with torch.no_grad():
        for data in validloader:
            volume_in, v_in, f_in, labels, subid, color_map = data
            volume_in = volume_in.to(device).float()
            v_in = v_in.to(device)
            f_in = f_in.to(device)
            labels = labels.to(device)

            csrvcnet.set_data(v_in, volume_in, f=f_in)
            logits = csrvcnet(v_in)
            preds = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy()

            predicted_vertices = v_in.squeeze(0).cpu().numpy()
            target_vertices = v_in.squeeze(0).cpu().numpy()  # Assuming the same vertices for simplicity
            target_labels = labels.squeeze(0).cpu().numpy()

            interpolated_ground_truth_labels = interpolate_labels(predicted_vertices, target_vertices, target_labels)
            dice_score = compute_dice(torch.tensor(preds), torch.tensor(interpolated_ground_truth_labels), num_classes, exclude_classes)
            all_dice_scores.append(dice_score)

    mean_dice_score = np.mean(all_dice_scores)
    print(f"Mean Dice Score: {mean_dice_score}")

if __name__ == '__main__':
    config = load_config()
    evaluate_model(config)
