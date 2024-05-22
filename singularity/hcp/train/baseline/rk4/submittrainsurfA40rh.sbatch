#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=55g
#SBATCH -p qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH -t 5-00:00
#SBATCH -J rk4stdr
#SBATCH -e jobs/error%A_%a.err
#SBATCH -o jobs/out%A_%a.out
#SBATCH -A psy53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe

sleep 5s

module load singularity/3.10.2

singularity exec --nv --bind /data,/data/users2/washbee/speedrun/:/speedrun,/data/users2/washbee/CortexODE-CSRFusionNet:/cortexode /data/users2/washbee/containers/csrf_sandbox/ /cortexode/singularity/hcp/train/baseline/rk4/trainsurfrhwm.sh $SLURM_ARRAY_TASK_ID&
pid3=$!  # Store the process ID of the first process

singularity exec --nv --bind /data,/data/users2/washbee/speedrun/:/speedrun,/data/users2/washbee/CortexODE-CSRFusionNet:/cortexode /data/users2/washbee/containers/csrf_sandbox/ /cortexode/singularity/hcp/train/baseline/rk4/trainsurfrhgm.sh $SLURM_ARRAY_TASK_ID&
pid4=$!  # Store the process ID of the second process

wait $pid3
wait $pid4

sleep 10s
