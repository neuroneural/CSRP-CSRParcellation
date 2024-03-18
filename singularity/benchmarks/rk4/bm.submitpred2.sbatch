#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -p qTRDGPU
#SBATCH --gres=gpu:RTX:1
#SBATCH -t 1-00:00
#SBATCH -J ctodeval  # Optionally include task ID in job name if provided
#SBATCH -e jobs/error%A.err
#SBATCH -o jobs/out%A.out
#SBATCH -A psy53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe

# Define CSV files paths
success_csv_file="memory_usage_success.csv"
failure_csv_file="memory_usage_failure.csv"

sleep 5s
echo "aa"
module load singularity/3.10.2

echo "bb"

python monitor.py &
MONITOR_PID=$!

# Capture the start time in seconds with nanoseconds
start_time=$(date +%s.%N)

# Run the command and capture its PID
singularity exec --nv --bind /data/users2/washbee/speedrun/:/speedrun,/data/users2/washbee/speedrun/CortexODE_fork:/cortexode /data/users2/washbee/containers/speedrun/cortexODE_sr_sandbox/ /cortexode/singularity/benchmarks/euler/bm.pred2.sh &
CORTX_PID=$!

# Wait for the singularity command to finish
wait $CORTX_PID
exit_status=$?

# Capture the end time in seconds with nanoseconds
end_time=$(date +%s.%N)

# Calculate the duration in seconds to milliseconds precision
duration=$(echo "$end_time - $start_time" | bc)

# After the command has finished, terminate the monitoring script
kill $MONITOR_PID

# Check if the command was successful
if [ $exit_status -eq 0 ]; then
    # Log success to CSV, using memory value passed to script
    echo "$(date),${1},${duration}" >> "${success_csv_file}"
    echo "Command executed successfully, memory usage and execution time (in seconds) logged to ${success_csv_file}."
else
    # Log failure to CSV, with exit status
    echo "$(date),Failed with exit status ${exit_status}, Memory Allocated: ${1}MB, Execution Time: ${duration} seconds" >> "${failure_csv_file}"
    echo "Command failed, error and execution time logged to ${failure_csv_file}."
fi

echo "CortexODE inference and monitoring have completed."
sleep 5s
