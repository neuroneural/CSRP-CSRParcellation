input_file_path = 'fastsurfer_measures.csv'
output_file_path = 'cleaned_fastsurfer_measures.csv'

with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
    header = next(infile)
    outfile.write(header)
    for line_number, line in enumerate(infile, start=2):  # Start at 2 to account for header line
        parts = line.strip().split(',')
        if len(parts) > 1:
            outfile.write(line)
        else:
            print(f"Skipping malformed line {line_number}: {line.strip()}")

print(f"Cleaned file saved to {output_file_path}")
