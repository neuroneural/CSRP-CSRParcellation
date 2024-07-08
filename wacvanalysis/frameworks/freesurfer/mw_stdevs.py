import trimesh
import pandas as pd
import numpy as np
import glob

# Function to calculate bounding box volume of a .ply file
def calculate_bounding_box_volume(ply_file):
    mesh = trimesh.load(ply_file)
    bounding_box = mesh.bounding_box_oriented
    extent = bounding_box.extents
    volume = extent[0] * extent[1] * extent[2]
    return volume

# Function to process each subject and its PLY files
def process_subject(subject_id, directory_path):
    volumes = {}
    surfaces = ['lh.pial', 'lh.white', 'rh.pial', 'rh.white']
    
    for surf in surfaces:
        ply_file = glob.glob(f"{directory_path}/{subject_id}.{surf}.medial_wall.ply")
        if ply_file:
            volumes[surf] = calculate_bounding_box_volume(ply_file[0])
        else:
            volumes[surf] = np.nan  # Use NaN for missing files

    return [subject_id, volumes['lh.pial'], volumes['lh.white'], volumes['rh.pial'], volumes['rh.white']]

def main(subjects_file, directory_path, output_csv):
    # Read subject IDs from file
    with open(subjects_file, 'r') as f:
        subject_ids = [line.strip() for line in f]

    # Process each subject
    data = [process_subject(subject_id, directory_path) for subject_id in subject_ids]

    # Create DataFrame from data
    df = pd.DataFrame(data, columns=["subject_id", "bv_lhpial", "bv_lhwhite", "bv_rhpial", "bv_rhwhite"])

    # Calculate population mean and standard deviation for each surface type
    for surf in ['bv_lhpial', 'bv_lhwhite', 'bv_rhpial', 'bv_rhwhite']:
        df[f'{surf}_mean'] = df[surf].mean()
        df[f'{surf}_std'] = df[surf].std()

    # Calculate number of standard deviations from mean for each surface type for each subject
    for surf in ['bv_lhpial', 'bv_lhwhite', 'bv_rhpial', 'bv_rhwhite']:
        df[f'{surf}_stdevfrommean'] = (df[surf] - df[f'{surf}_mean']) / df[f'{surf}_std']

    # Determine the lowest standard deviation from the mean for each subject
    df['min_stdevfrommean'] = df[[f'{surf}_stdevfrommean' for surf in ['bv_lhpial', 'bv_lhwhite', 'bv_rhpial', 'bv_rhwhite']]].min(axis=1)

    # Prepare the final DataFrame for CSV
    final_df = df[["subject_id", "min_stdevfrommean"]]

    # Saving to a CSV file
    final_df.to_csv(output_csv, index=False)

    # Print out the number of subjects greater than and less than -2 standard deviations from the mean
    num_greater_than_minus_2 = len(final_df[final_df['min_stdevfrommean'] > -2])
    num_less_than_minus_2 = len(final_df[final_df['min_stdevfrommean'] < -2])

    print(f"Number of subjects with min_stdevfrommean greater than -2: {num_greater_than_minus_2}")
    print(f"Number of subjects with min_stdevfrommean less than -2: {num_less_than_minus_2}")

if __name__ == "__main__":
    # Hardcoded parameters
    subjects_file = "../test_subs.txt"
    directory_path = "mwremoved/"
    output_csv = "mwremoved/lowerstdevs.csv"

    main(subjects_file, directory_path, output_csv)
