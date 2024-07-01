#!/bin/bash
#SBATCH -A psy53c17
#SBATCH --output=jobs/out%a_%A.out
#SBATCH --error=jobs/err%a_%A.err
#SBATCH --time=01:00:00
#SBATCH --job-name=sfgnwacv
#SBATCH -c 4
#SBATCH --mem=32g
#SBATCH -p qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe

module load singularity/3.10.2

singularity exec --nv --bind /data,/data/users2/washbee/speedrun/:/speedrun,/data/users2/washbee/CortexODE-CSRFusionNet:/cortexode /data/users2/washbee/containers/csrf_sandbox/ /cortexode/wacvanalysis/frameworks/cortexode/rigidbody.sh

