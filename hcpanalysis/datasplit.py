import random

# Set the seed for reproducibility
random.seed(42)

# Load paths from the text file
with open('bsnip.txt', 'r') as file:
    paths = file.readlines()

# Remove any trailing newline characters
paths = [path.strip() for path in paths]

# Shuffle the paths randomly
random.shuffle(paths)

# Define split ratios (e.g., 70% train, 15% validation, 15% test)
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Calculate the number of samples for each set
total_count = len(paths)
train_count = int(total_count * train_ratio)
val_count = int(total_count * val_ratio)
test_count = total_count - train_count - val_count  # Ensuring the remainder goes to the test set

# Split the data
train_paths = paths[:train_count]
val_paths = paths[train_count:train_count + val_count]
test_paths = paths[train_count + val_count:]

# Write the splits to separate files
with open('bsniptrain.txt', 'w') as train_file:
    for path in train_paths:
        train_file.write(f"{path}\n")

with open('bsnipval.txt', 'w') as val_file:
    for path in val_paths:
        val_file.write(f"{path}\n")

with open('bsniptest.txt', 'w') as test_file:
    for path in test_paths:
        test_file.write(f"{path}\n")

print("Splitting completed successfully with a set seed!")
