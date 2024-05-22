#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=30g
#SBATCH -p qTRDGPU
#SBATCH --gres=gpu:RTX:1
#SBATCH -t 5-00:00
#SBATCH -J ctoinitr
#SBATCH -e jobs/error%A.err
#SBATCH -o jobs/out%A.out
#SBATCH -A psy53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe

sleep 5s

module load singularity/3.10.2
singularity exec --nv --bind /data/users2/washbee/speedrun/:/speedrun,/data/users2/washbee/CortexODE-CSRFusionNet:/cortexode /data/users2/washbee/containers/csrf_sandbox/ /cortexode/singularity/init/initsurfrh.sh &

wait

sleep 10s

