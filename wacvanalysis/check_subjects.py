import pandas as pd
from collections import Counter

# Path to your CSV file
input_file_path = 'fastsurfer_measures.csv'
output_cleaned_file_path = 'cleaned_fastsurfer_measures.csv'
problematic_subjects_file_path = 'remaining.txt'

# Initialize a counter for subject IDs
subject_counter = Counter()

# Read the file line by line
with open(input_file_path, 'r') as file:
    # Read the header
    header = next(file).strip().split(',')
    
    # Count subject IDs
    for line_number, line in enumerate(file, start=2):  # Start at 2 to account for header line
        parts = line.strip().split(',')
        if len(parts) > 1 and any(parts):  # Ensure the line has more than one column and is not empty
            subject_id = parts[1].strip()  # Assuming Subject ID is the second column
            if subject_id and subject_id != '':  # Check if the subject_id is not empty or blank
                subject_counter[subject_id] += 1
            else:
                print(f"Skipping line {line_number} with empty or blank Subject ID: {line.strip()}")
        else:
            print(f"Skipping malformed line {line_number}: {line.strip()}")

# Find subjects with counts different from the expected 70
expected_count = 70
problematic_subjects = {subject_id: count for subject_id, count in subject_counter.items() if count != expected_count}

# Print problematic subjects for debugging
print("Problematic Subjects and their counts:")
with open(problematic_subjects_file_path, 'w') as f:
    for subject_id, count in problematic_subjects.items():
        print(f"Subject ID: '{subject_id}', Count: {count}")
        f.write(f"{subject_id}\n")

# Load the original file into a DataFrame, ensuring Subject ID is read as a string
data = pd.read_csv(input_file_path, dtype={'Subject ID': str})

# Strip any whitespace from the Subject ID column in the DataFrame
data['Subject ID'] = data['Subject ID'].str.strip()

# Exclude rows with blank subject IDs or empty columns
data = data[data['Subject ID'].notna() & (data['Subject ID'] != '')]

# Identify rows to exclude based on problematic subject IDs
cleaned_data = data[~data['Subject ID'].isin(problematic_subjects.keys())]

# Save the cleaned data to a new CSV file
cleaned_data.to_csv(output_cleaned_file_path, index=False)

print(f"Cleaned data saved to {output_cleaned_file_path}")
print(f"Problematic subjects saved to {problematic_subjects_file_path}")
