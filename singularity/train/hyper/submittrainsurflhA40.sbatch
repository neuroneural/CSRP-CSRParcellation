#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=90g
#SBATCH -p qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH -t 0-01:00
#SBATCH -J ctrlh2A
#SBATCH -e jobs/error%A_%a.err
#SBATCH -o jobs/out%A_%a.out
#SBATCH -A psy53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe

sleep 5s

module load singularity/3.10.2

singularity exec --nv --bind /data,/data/users2/washbee/speedrun/:/speedrun,/data/users2/washbee/CortexODE-CSRFusionNet:/cortexode /data/users2/washbee/containers/csrf_sandbox/ /cortexode/singularity/train/hyper/trainsurflhwm.sh $SLURM_ARRAY_TASK_ID&
pid1=$!  # Store the process ID of the first process

singularity exec --nv --bind /data,/data/users2/washbee/speedrun/:/speedrun,/data/users2/washbee/CortexODE-CSRFusionNet:/cortexode /data/users2/washbee/containers/csrf_sandbox/ /cortexode/singularity/train/hyper/trainsurflhgm.sh $SLURM_ARRAY_TASK_ID&
pid2=$!  # Store the process ID of the second process

wait $pid1
wait $pid2

sleep 10s
