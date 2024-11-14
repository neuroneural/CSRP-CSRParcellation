import sys
import os
import copy

# Determine the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))

# Add the project root to sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed

# Import your dataset class and configuration loader
from data.csrandvcdataloader import BrainDataset
from config import load_config

def compute_cortical_thickness(fs_white_coords, fs_pial_coords):
    """
    Compute cortical thickness as the Euclidean distance between corresponding vertices
    on the white matter and pial surfaces.
    """
    thickness = np.linalg.norm(fs_pial_coords - fs_white_coords, axis=1)
    return thickness

def compute_face_areas(coords, faces):
    """
    Compute the area of each face (triangle) on the surface mesh.
    """
    v0 = coords[faces[:, 0]]
    v1 = coords[faces[:, 1]]
    v2 = coords[faces[:, 2]]
    # Compute the vectors representing two sides of the triangle
    vec1 = v1 - v0
    vec2 = v2 - v0
    # Compute the cross product of these vectors
    cross_prod = np.cross(vec1, vec2)
    # The area of the triangle is half the magnitude of the cross product
    face_areas = np.linalg.norm(cross_prod, axis=1) * 0.5
    return face_areas

def process_subject(subject_data):
    """
    Process a single subject's data to compute regional and whole-cortex cortical thickness and area.
    """
    subject_id = subject_data['subject_id']
    group = subject_data['group']
    hemisphere = subject_data['hemisphere']
    labels = subject_data['labels']
    white_coords = subject_data['white_coords']
    pial_coords = subject_data['pial_coords']
    faces = subject_data['faces']

    # Compute cortical thickness
    thickness = compute_cortical_thickness(white_coords, pial_coords)

    # Compute face areas on the pial surface
    face_areas = compute_face_areas(pial_coords, faces)

    # Map faces to region labels
    face_labels = labels[faces]
    # Assign faces to regions where all three vertices have the same label
    face_regions = np.where(
        (face_labels[:, 0] == face_labels[:, 1]) & (face_labels[:, 1] == face_labels[:, 2]),
        face_labels[:, 0],
        -1
    )

    # Filter valid faces (exclude faces with label -1 or 4)
    valid_faces = (face_regions != -1) & (face_regions != 4)
    face_regions_filtered = face_regions[valid_faces]
    face_areas_filtered = face_areas[valid_faces]

    # Use region labels directly
    face_region_labels = face_regions_filtered

    # Aggregate area per region label
    area_df = pd.DataFrame({
        'Region': face_region_labels,
        'Area': face_areas_filtered
    })
    area_by_region = area_df.groupby('Region')['Area'].sum().reset_index()

    # Map vertex thickness to region labels (exclude vertices with label -1 or 4)
    valid_vertices = (labels != -1) & (labels != 4)
    vertex_region_labels = labels[valid_vertices]
    thickness_filtered = thickness[valid_vertices]
    thickness_df = pd.DataFrame({
        'Region': vertex_region_labels,
        'Thickness': thickness_filtered
    })
    thickness_by_region = thickness_df.groupby('Region')['Thickness'].mean().reset_index()

    # Merge area and thickness data per region label
    region_stats = pd.merge(area_by_region, thickness_by_region, on='Region')

    # Calculate whole-cortex metrics (excluding regions with labels -1 and 4)
    whole_cortex_area = face_areas_filtered.sum()
    whole_cortex_thickness = thickness_filtered.mean()

    # Add whole-cortex data
    whole_cortex_data = pd.DataFrame({
        'Region': ['WholeCortex'],
        'Area': [whole_cortex_area],
        'Thickness': [whole_cortex_thickness]
    })

    # Combine regional and whole-cortex data
    region_stats = pd.concat([region_stats, whole_cortex_data], ignore_index=True)

    # Add subject metadata
    region_stats['Subject_ID'] = subject_id
    region_stats['Group'] = group
    region_stats['Hemisphere'] = hemisphere
    region_stats['Model'] = 'GroundTruth'  # Since we're using ground truth data

    return region_stats


def load_data_for_analysis(config, data_usage='test'):
    """
    Load data using the DataLoader from the training code.
    """
    # Create separate configs for WM and GM surfaces
    config_wm = copy.deepcopy(config)
    config_wm.surf_type = 'wm'

    config_gm = copy.deepcopy(config)
    config_gm.surf_type = 'gm'

    # Load the datasets
    dataset_wm = BrainDataset(config_wm, data_usage, affCtab=True)
    dataset_gm = BrainDataset(config_gm, data_usage, affCtab=True)

    # Ensure both datasets have the same subjects
    assert len(dataset_wm) == len(dataset_gm), "WM and GM datasets have different lengths."

    # Create DataLoaders
    dataloader_wm = DataLoader(dataset_wm, batch_size=1, shuffle=False, num_workers=4)
    dataloader_gm = DataLoader(dataset_gm, batch_size=1, shuffle=False, num_workers=4)

    return dataloader_wm, dataloader_gm

# Main analysis function
if __name__ == '__main__':
    # Load your configuration
    config = load_config()

    # Update the config for the test set
    config.data_usage = 'test'

    # Define group labels for each subject (you need to provide this mapping)
    subject_groups = {
        # 'subj1': 'control',
        # 'subj2': 'disease',
        # Add all subjects with their respective group labels
    }

    # Load data for both WM and GM surfaces
    dataloader_wm, dataloader_gm = load_data_for_analysis(config, data_usage='test')

    # List to hold all subjects' data
    all_subjects_data = []

    # Iterate over both DataLoaders simultaneously
    for (data_wm, data_gm) in zip(dataloader_wm, dataloader_gm):
        # Unpack WM data
        volume_in_wm, v_in_wm, v_gt_wm, f_in_wm, f_gt_wm, labels_wm, aff_wm, ctab_wm, subid_wm = data_wm
        # Unpack GM data
        volume_in_gm, v_in_gm, v_gt_gm, f_in_gm, f_gt_gm, labels_gm, aff_gm, ctab_gm, subid_gm = data_gm

        print('subid_wm',subid_wm,'subid_gm',subid_gm)
        # Ensure the subject IDs match
        subject_id = subid_wm[0]
        hemisphere = config.surf_hemi  # Assuming hemisphere is specified in config
        group = subject_groups.get(subject_id, 'Unknown')  # Replace 'Unknown' with default group if necessary

        # Extract data
        white_coords = v_gt_wm.squeeze(0).numpy()
        pial_coords = v_gt_gm.squeeze(0).numpy()
        faces = f_gt_wm.squeeze(0).numpy()  # Assuming faces are the same for WM and GM

        # Extract labels
        labels = labels_wm.squeeze(0).numpy()
        # No need to extract region names or region_dict since we're using labels directly

        # Prepare subject data
        subject_data = {
            'subject_id': subject_id,
            'group': group,
            'hemisphere': hemisphere,
            'labels': labels,
            'white_coords': white_coords,
            'pial_coords': pial_coords,
            'faces': faces
        }

        # Process the subject
        region_stats = process_subject(subject_data)

        # Append to all subjects data
        all_subjects_data.append(region_stats)

    # Combine all data into a single DataFrame
    combined_data = pd.concat(all_subjects_data, ignore_index=True)

    # Construct the output filename using config.atlas
    output_filename = f'combined_ground_truth_data_{config.atlas}.csv'

    # Save combined data to CSV
    combined_data.to_csv(output_filename, index=False)

    # Now you can proceed with your statistical analysis as before
    # For example, perform group comparisons if group labels are available

    # Separate data into control and disease groups if applicable
    if 'control' in combined_data['Group'].unique() and 'disease' in combined_data['Group'].unique():
        control_data = combined_data[combined_data['Group'] == 'control']
        disease_data = combined_data[combined_data['Group'] == 'disease']

        # Perform t-tests per region and hemisphere
        regions = combined_data['Region'].unique()
        hemispheres = combined_data['Hemisphere'].unique()
        t_test_results = []

        for region in regions:
            for hemisphere in hemispheres:
                control_values = control_data[
                    (control_data['Region'] == region) &
                    (control_data['Hemisphere'] == hemisphere)
                ]
                disease_values = disease_data[
                    (disease_data['Region'] == region) &
                    (disease_data['Hemisphere'] == hemisphere)
                ]
                # Ensure there are enough samples
                if len(control_values) > 1 and len(disease_values) > 1:
                    # Thickness
                    t_stat_thickness, p_value_thickness = ttest_ind(
                        control_values['Thickness'],
                        disease_values['Thickness'],
                        equal_var=False,
                        nan_policy='omit'
                    )
                    # Area
                    t_stat_area, p_value_area = ttest_ind(
                        control_values['Area'],
                        disease_values['Area'],
                        equal_var=False,
                        nan_policy='omit'
                    )
                    # Store results
                    t_test_results.append({
                        'Region': region,
                        'Hemisphere': hemisphere,
                        't_stat_thickness': t_stat_thickness,
                        'p_value_thickness': p_value_thickness,
                        't_stat_area': t_stat_area,
                        'p_value_area': p_value_area
                    })

        # Convert to DataFrame
        t_test_df = pd.DataFrame(t_test_results)

        # Adjust p-values for multiple comparisons using FDR
        # Thickness
        reject_thickness, pvals_corrected_thickness, _, _ = multipletests(
            t_test_df['p_value_thickness'], alpha=0.05, method='fdr_bh'
        )
        t_test_df['pval_corrected_thickness'] = pvals_corrected_thickness
        t_test_df['Significant_Thickness'] = reject_thickness

        # Area
        reject_area, pvals_corrected_area, _, _ = multipletests(
            t_test_df['p_value_area'], alpha=0.05, method='fdr_bh'
        )
        t_test_df['pval_corrected_area'] = pvals_corrected_area
        t_test_df['Significant_Area'] = reject_area

        # Save t-test results to CSV
        t_test_df.to_csv('t_test_results_between_groups.csv', index=False)

        # Display significant results
        print("\nSignificant Regions for Cortical Thickness:")
        print(t_test_df[t_test_df['Significant_Thickness']][
            ['Region', 'Hemisphere', 't_stat_thickness', 'pval_corrected_thickness']
        ])

        print("\nSignificant Regions for Surface Area:")
        print(t_test_df[t_test_df['Significant_Area']][
            ['Region', 'Hemisphere', 't_stat_area', 'pval_corrected_area']
        ])

        # Save significant results to CSV
        significant_thickness = t_test_df[t_test_df['Significant_Thickness']]
        significant_thickness.to_csv('significant_thickness_regions.csv', index=False)

        significant_area = t_test_df[t_test_df['Significant_Area']]
        significant_area.to_csv('significant_area_regions.csv', index=False)
    else:
        print("Group labels 'control' and 'disease' not found in the data. Skipping group comparisons.")
