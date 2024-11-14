#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=40g
#SBATCH -p qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH -t 2-00:00
#SBATCH -J ctotrseg
#SBATCH -e jobs/error%A.err
#SBATCH -o jobs/out%A.out
#SBATCH -A psy53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe
#SBATCH --exclude=arctrdagn031,arctrdagn032,arctrdagn033,arctrdagn034,arctrdagn035
sleep 5s

module load singularity/3.10.2
singularity exec --nv --bind /data,/data/users2/washbee/speedrun/:/speedrun,/data/users2/washbee/CortexODE-CSRFusionNet:/cortexode /data/users2/washbee/containers/csrf_sandbox/ /cortexode/hcpanalysis/singularity/hcp/seg/trainseg.sh &

wait

sleep 10s
