import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import ttest_ind, ttest_rel
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed

# Define paths and subjects
subjects_dir = 'path_to_subjects_dir'  # Replace with your subjects directory
subject_ids = ['subj1', 'subj2', 'subj3']  # Replace with your subject IDs
hemispheres = ['lh', 'rh']  # Left and right hemispheres

# Define group labels for each subject
subject_groups = {
    'subj1': 'control',
    'subj2': 'disease',
    'subj3': 'control',
    # Add all subjects with their respective group labels
}

# Define paths to your model's outputs and other models' outputs
your_model_dir = 'path_to_your_model_outputs'
other_model_dir = 'path_to_other_model_outputs'

# Initialize a list to hold data from all subjects
all_subjects_data = []

def process_subject(subject_id):
    subject_data = []
    for hemisphere in hemispheres:
        # Construct file paths
        fs_white_path = os.path.join(subjects_dir, subject_id, 'surf', f'{hemisphere}.white')
        fs_pial_path = os.path.join(subjects_dir, subject_id, 'surf', f'{hemisphere}.pial')
        fs_annot_path = os.path.join(subjects_dir, subject_id, 'label', f'{hemisphere}.aparc.annot')
        
        # Load FreeSurfer surfaces
        fs_white_coords, fs_faces = nib.freesurfer.read_geometry(fs_white_path)
        fs_pial_coords, _ = nib.freesurfer.read_geometry(fs_pial_path)
        
        # Load annotations
        labels, ctab, region_names = nib.freesurfer.read_annot(fs_annot_path)
        region_names = [name.decode('utf-8') for name in region_names]
        region_dict = dict(enumerate(region_names))
        
        # Compute cortical thickness (Euclidean distance between corresponding vertices)
        thickness = np.linalg.norm(fs_pial_coords - fs_white_coords, axis=1)
        
        # Compute face areas on the pial surface (areas per face)
        v0 = fs_pial_coords[fs_faces[:, 0]]
        v1 = fs_pial_coords[fs_faces[:, 1]]
        v2 = fs_pial_coords[fs_faces[:, 2]]
        
        # Use vectorized cross product to compute area of each triangle
        cross_prod = np.cross(v1 - v0, v2 - v0)
        face_areas = np.linalg.norm(cross_prod, axis=1) * 0.5
        
        # Map faces to regions
        face_labels = labels[fs_faces]
        
        # Assign faces to regions where all three vertices have the same label
        face_regions = np.where(
            (face_labels[:, 0] == face_labels[:, 1]) & (face_labels[:, 1] == face_labels[:, 2]),
            face_labels[:, 0],
            -1
        )
        
        # Filter valid faces
        valid_faces = face_regions != -1
        face_regions = face_regions[valid_faces]
        face_areas = face_areas[valid_faces]
        
        # Map face regions to region names
        face_region_names = [region_dict[label] for label in face_regions]
        
        # Aggregate area per region
        area_df = pd.DataFrame({
            'Region': face_region_names,
            'Area': face_areas
        })
        area_by_region = area_df.groupby('Region')['Area'].sum().reset_index()
        
        # Map vertex thickness to regions
        vertex_regions = [region_dict[label] if label in region_dict else 'Unknown' for label in labels]
        thickness_df = pd.DataFrame({
            'Region': vertex_regions,
            'Thickness': thickness
        })
        thickness_by_region = thickness_df.groupby('Region')['Thickness'].mean().reset_index()
        
        # Merge area and thickness data
        region_stats = pd.merge(area_by_region, thickness_by_region, on='Region')
        region_stats['Subject_ID'] = subject_id
        region_stats['Group'] = subject_groups[subject_id]
        region_stats['Hemisphere'] = hemisphere
        region_stats['Model'] = 'FreeSurfer'
        
        # Append to subject data
        subject_data.append(region_stats)
    return pd.concat(subject_data, ignore_index=True)

# Use joblib.Parallel for parallel processing of subjects
all_subjects_data = Parallel(n_jobs=-1)(delayed(process_subject)(subj) for subj in subject_ids)

# Combine all data into a single DataFrame
combined_data = pd.concat(all_subjects_data, ignore_index=True)

# Save combined FreeSurfer data to CSV
combined_data.to_csv('combined_freesurfer_data.csv', index=False)

# Now, repeat the same process for your model and other models
def process_model_subject(subject_id, model_name, model_dir):
    subject_data = []
    for hemisphere in hemispheres:
        # Paths to your model's surfaces
        model_white_path = os.path.join(model_dir, subject_id, 'surf', f'{hemisphere}.white')
        model_pial_path = os.path.join(model_dir, subject_id, 'surf', f'{hemisphere}.pial')
        model_annot_path = os.path.join(model_dir, subject_id, 'label', f'{hemisphere}.aparc.annot')
        
        # Load model surfaces
        model_white_coords, model_faces = nib.freesurfer.read_geometry(model_white_path)
        model_pial_coords, _ = nib.freesurfer.read_geometry(model_pial_path)
        
        # Load annotations
        labels, ctab, region_names = nib.freesurfer.read_annot(model_annot_path)
        region_names = [name.decode('utf-8') for name in region_names]
        region_dict = dict(enumerate(region_names))
        
        # Compute cortical thickness
        thickness = np.linalg.norm(model_pial_coords - model_white_coords, axis=1)
        
        # Compute face areas
        v0 = model_pial_coords[model_faces[:, 0]]
        v1 = model_pial_coords[model_faces[:, 1]]
        v2 = model_pial_coords[model_faces[:, 2]]
        cross_prod = np.cross(v1 - v0, v2 - v0)
        face_areas = np.linalg.norm(cross_prod, axis=1) * 0.5
        
        # Map faces to regions
        face_labels = labels[model_faces]
        face_regions = np.where(
            (face_labels[:, 0] == face_labels[:, 1]) & (face_labels[:, 1] == face_labels[:, 2]),
            face_labels[:, 0],
            -1
        )
        valid_faces = face_regions != -1
        face_regions = face_regions[valid_faces]
        face_areas = face_areas[valid_faces]
        face_region_names = [region_dict[label] for label in face_regions]
        
        # Aggregate area per region
        area_df = pd.DataFrame({
            'Region': face_region_names,
            'Area': face_areas
        })
        area_by_region = area_df.groupby('Region')['Area'].sum().reset_index()
        
        # Map vertex thickness to regions
        vertex_regions = [region_dict[label] if label in region_dict else 'Unknown' for label in labels]
        thickness_df = pd.DataFrame({
            'Region': vertex_regions,
            'Thickness': thickness
        })
        thickness_by_region = thickness_df.groupby('Region')['Thickness'].mean().reset_index()
        
        # Merge area and thickness data
        region_stats = pd.merge(area_by_region, thickness_by_region, on='Region')
        region_stats['Subject_ID'] = subject_id
        region_stats['Group'] = subject_groups[subject_id]
        region_stats['Hemisphere'] = hemisphere
        region_stats['Model'] = model_name
        
        # Append to subject data
        subject_data.append(region_stats)
    return pd.concat(subject_data, ignore_index=True)

# Process data for your model
your_model_data = Parallel(n_jobs=-1)(delayed(process_model_subject)(subj, 'YourModel', your_model_dir) for subj in subject_ids)
your_model_combined = pd.concat(your_model_data, ignore_index=True)

# Save your model's data to CSV
your_model_combined.to_csv('your_model_data.csv', index=False)

# Process data for other models
other_model_data = Parallel(n_jobs=-1)(delayed(process_model_subject)(subj, 'OtherModel', other_model_dir) for subj in subject_ids)
other_model_combined = pd.concat(other_model_data, ignore_index=True)

# Save other model's data to CSV
other_model_combined.to_csv('other_model_data.csv', index=False)

# Combine all models' data
all_models_data = pd.concat([combined_data, your_model_combined, other_model_combined], ignore_index=True)

# Prepare data for statistical analysis
# Merge FreeSurfer data with your model's data and other models' data
def prepare_error_data(model_data, model_name):
    error_data = []
    for subject_id in subject_ids:
        for hemisphere in hemispheres:
            # Extract FreeSurfer data for this subject and hemisphere
            fs_data = combined_data[
                (combined_data['Subject_ID'] == subject_id) &
                (combined_data['Hemisphere'] == hemisphere)
            ]
            # Extract model data for this subject and hemisphere
            mdl_data = model_data[
                (model_data['Subject_ID'] == subject_id) &
                (model_data['Hemisphere'] == hemisphere)
            ]
            # Merge on Region
            merged_data = pd.merge(fs_data, mdl_data, on='Region', suffixes=('_FS', f'_{model_name}'))
            # Compute errors
            merged_data['Thickness_Error'] = merged_data[f'Thickness_{model_name}'] - merged_data['Thickness_FS']
            merged_data['Area_Error'] = merged_data[f'Area_{model_name}'] - merged_data['Area_FS']
            merged_data['Subject_ID'] = subject_id
            merged_data['Hemisphere'] = hemisphere
            error_data.append(merged_data[['Region', 'Subject_ID', 'Hemisphere', 'Thickness_Error', 'Area_Error']])
    return pd.concat(error_data, ignore_index=True)

# Compute errors for your model
your_model_errors = prepare_error_data(your_model_combined, 'YourModel')

# Save your model's error data to CSV
your_model_errors.to_csv('your_model_errors.csv', index=False)

# Compute errors for other model
other_model_errors = prepare_error_data(other_model_combined, 'OtherModel')

# Save other model's error data to CSV
other_model_errors.to_csv('other_model_errors.csv', index=False)

# Merge errors for statistical comparison
error_comparison = pd.merge(
    your_model_errors,
    other_model_errors,
    on=['Region', 'Subject_ID', 'Hemisphere'],
    suffixes=('_YourModel', '_OtherModel')
)

# Save error comparison data to CSV
error_comparison.to_csv('model_error_comparisons.csv', index=False)

# Perform paired t-tests across all regions
# Thickness Error Comparison
t_stat_thickness, p_value_thickness = ttest_rel(
    error_comparison['Thickness_Error_OtherModel'],
    error_comparison['Thickness_Error_YourModel']
)
p_value_thickness_one_tailed = p_value_thickness / 2 if t_stat_thickness > 0 else 1 - (p_value_thickness / 2)
print(f"Thickness Errors - t-statistic: {t_stat_thickness}, p-value: {p_value_thickness_one_tailed}")

# Surface Area Error Comparison
t_stat_area, p_value_area = ttest_rel(
    error_comparison['Area_Error_OtherModel'],
    error_comparison['Area_Error_YourModel']
)
p_value_area_one_tailed = p_value_area / 2 if t_stat_area > 0 else 1 - (p_value_area / 2)
print(f"Surface Area Errors - t-statistic: {t_stat_area}, p-value: {p_value_area_one_tailed}")

# Calculate effect sizes
def cohen_d_paired(x, y):
    diff = x - y
    return np.mean(diff) / np.std(diff, ddof=1)

effect_size_thickness = cohen_d_paired(
    error_comparison['Thickness_Error_OtherModel'],
    error_comparison['Thickness_Error_YourModel']
)
print(f"Effect Size (Cohen's d) for Thickness Errors: {effect_size_thickness}")

effect_size_area = cohen_d_paired(
    error_comparison['Area_Error_OtherModel'],
    error_comparison['Area_Error_YourModel']
)
print(f"Effect Size (Cohen's d) for Surface Area Errors: {effect_size_area}")

# Now perform t-tests between disease and control groups for FreeSurfer data
# First, separate FreeSurfer data into control and disease groups
fs_control = combined_data[combined_data['Group'] == 'control']
fs_disease = combined_data[combined_data['Group'] == 'disease']

# Perform t-tests per region and hemisphere
regions = combined_data['Region'].unique()
hemispheres = combined_data['Hemisphere'].unique()
t_test_results = []

for region in regions:
    for hemisphere in hemispheres:
        control_values = fs_control[
            (fs_control['Region'] == region) &
            (fs_control['Hemisphere'] == hemisphere)
        ]
        disease_values = fs_disease[
            (fs_disease['Region'] == region) &
            (fs_disease['Hemisphere'] == hemisphere)
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

