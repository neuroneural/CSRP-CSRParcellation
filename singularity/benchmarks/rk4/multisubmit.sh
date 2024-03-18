#!/bin/bash

# Job parameters
#num_tasks=49
#num_tasks=3
base_mem=250  # Base memory in MB
max_mem=13288  # Max memory in MB
#max_mem=755  # Max memory in MB
#chatgpt, could you calculate the num_tasks using the base_mem and max_mem parameter
job_name="dynamic_mem_job"
file='memory_numbers'

# Calculate num_tasks based on base_mem and max_mem
num_tasks=0
total_mem=0

while [ $total_mem -lt $max_mem ]; do
    num_tasks=$((num_tasks + 1))
    total_mem=$((num_tasks * base_mem))
done
echo "num_tasks ${num_tasks}"

# Initialize or clear the memory allocation log file
echo "" > memory_allocation.log

for task_id in $(seq 1 $num_tasks); do
    # Calculate memory allocation
    mem=$(( (task_id - 1) * base_mem + base_mem ))  # Ensure base memory is allocated to the first task
    mem=$(( mem < max_mem ? mem : max_mem ))
    
    # Generate unique output file name based on task ID
    output_file="job_${task_id}.out"
    
    # Generate and append memory log entry
    echo "Allocating memory: ${mem}MB for task ${task_id}" >> memory_allocation.log
    
    # Submit the job with specified memory
    sbatch --mem=${mem}M bm.submitpred2.sbatch ${mem} {task_id}
done

