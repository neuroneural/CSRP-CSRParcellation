import subprocess
import re
import csv
import time
import argparse
import os  # Import the os module

# Setup argument parser to take PID as an argument
parser = argparse.ArgumentParser(description='Monitor GPU Memory Usage.')
args = parser.parse_args()

# Parameters
interval = 0.01  # Polling interval in seconds
output_csv_file = 'gpu_memory_usage_log.csv'

# Check if the CSV file exists and write the header if not
if not os.path.exists(output_csv_file):
    with open(output_csv_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header to only include GPU memory usage
        csvwriter.writerow(['Timestamp', 'GPU Memory Usage (MiB)'])

# Function to get GPU memory usage
def get_gpu_memory_usage():
    smi_output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits']).decode()
    memory_used = int(re.findall(r'\d+', smi_output)[0])
    return memory_used

# Monitoring loop
try:
    while True:
        gpu_memory_used = get_gpu_memory_usage()
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"{timestamp}, GPU Memory Usage: {gpu_memory_used} MiB")

        # Append to CSV
        with open(output_csv_file, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([timestamp, gpu_memory_used])

        time.sleep(interval)

except KeyboardInterrupt:
    print("Monitoring stopped by user.")
