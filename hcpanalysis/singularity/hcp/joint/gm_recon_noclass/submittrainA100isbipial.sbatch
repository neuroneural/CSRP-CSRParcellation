#!/bin/bash
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=110g
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:A100:1
#SBATCH -t 5-00:00
#SBATCH -J hcpgm_rnc
#SBATCH -e jobs/error%A_%a.err
#SBATCH -o jobs/out%A_%a.out
#SBATCH -A psy53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe
#SBATCH --exclude=arctrddgxa001,arctrdcn001,arctrdcn006,arctrdcn010,arctrdagn031

sleep 5s

module load singularity/3.10.2

singularity exec --nv \
    --bind /data,\
/data/users2/washbee/speedrun/:/speedrun,\
/data/users2/washbee/CortexODE-CSRFusionNet:/cortexode \
    /data/users2/washbee/containers/csrf_sandbox/ \
    /cortexode/hcpanalysis/singularity/hcp/joint/gm_recon_noclass/train_csrvc_lhgm.sh $SLURM_ARRAY_TASK_ID &

pid3=$!  # Store the process ID of the second process

singularity exec --nv \
    --bind /data,\
/data/users2/washbee/speedrun/:/speedrun,\
/data/users2/washbee/CortexODE-CSRFusionNet:/cortexode \
    /data/users2/washbee/containers/csrf_sandbox/ \
    /cortexode/hcpanalysis/singularity/hcp/joint/gm_recon_noclass/train_csrvc_rhgm.sh $SLURM_ARRAY_TASK_ID &

pid4=$!  # Store the process ID of the second process

wait $pid3
wait $pid4


sleep 10s
