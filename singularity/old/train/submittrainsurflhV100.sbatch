#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=90g
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:V100:1
#SBATCH -t 1-00:00
#SBATCH -J ctrlh2A
#SBATCH -e jobs/error%A.err
#SBATCH -o jobs/out%A.out
#SBATCH -A psy53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe

sleep 5s

module load singularity/3.10.2


singularity exec --nv --bind /data/users2/washbee/speedrun/:/speedrun,/data/users2/washbee/CortexODE-CSRFusionNet:/cortexode /data/users2/washbee/containers/speedrun/csrfusion_sandbox/ /cortexode/singularity/train/trainsurflhwm.sh &
pid1=$!  # Store the process ID of the first process

singularity exec --nv --bind /data/users2/washbee/speedrun/:/speedrun,/data/users2/washbee/CortexODE-CSRFusionNet:/cortexode /data/users2/washbee/containers/speedrun/csrfusion_sandbox/ /cortexode/singularity/train/trainsurflhgm.sh &
pid2=$!  # Store the process ID of the second process

wait $pid1
wait $pid2


sleep 10s
