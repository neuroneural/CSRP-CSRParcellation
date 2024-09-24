#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=55g
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:V100:1
#SBATCH -t 5-00:00
#SBATCH -J vcLGs
#SBATCH -e jobs/error%A_%a.err
#SBATCH -o jobs/out%A_%a.out
#SBATCH -A psy53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe
#SBATCH --exclude=arctrddgxa001

sleep 5s

module load singularity/3.10.2

singularity exec --nv --bind /data,/data/users2/washbee/speedrun/:/speedrun,/data/users2/washbee/CortexODE-CSRFusionNet:/cortexode /data/users2/washbee/containers/csrf_sandbox/ /cortexode/singularity/isbi/csrvcv3/symmetric/trainsurfvclhgm.sh $SLURM_ARRAY_TASK_ID&
pid1=$!  # Store the process ID of the second process

# singularity exec --nv --bind /data,/data/users2/washbee/speedrun/:/speedrun,/data/users2/washbee/CortexODE-CSRFusionNet:/cortexode /data/users2/washbee/containers/csrf_sandbox/ /cortexode/singularity/isbi/csrvcv3/symmetric/trainsurfvcrhgm.sh $SLURM_ARRAY_TASK_ID&
# pid2=$!  # Store the process ID of the second process

wait $pid1
# wait $pid2

sleep 10s
