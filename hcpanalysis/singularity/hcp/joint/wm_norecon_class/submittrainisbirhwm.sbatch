#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=20g
#SBATCH -p qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH -t 0-03:00
#SBATCH -J csrvcrh
#SBATCH -e jobs/error%A_%a.err
#SBATCH -o jobs/out%A_%a.out
#SBATCH -A psy53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe
#SBATCH --exclude=arctrddgxa001

sleep 5s

module load singularity/3.10.2

# Launch the training script inside the Singularity container
singularity exec --nv \
    --bind /data,\
/data/users2/washbee/speedrun/:/speedrun,\
/data/users2/washbee/CortexODE-CSRFusionNet:/cortexode \
    /data/users2/washbee/containers/csrf_sandbox/ \
    /cortexode/bsnipanalysis/singularity/bsnip/joint/wm_norecon_class/train_csrvc_rhwm.sh $SLURM_ARRAY_TASK_ID &

pid1=$!  # Store the process ID of the second process



wait $pid1




sleep 10s
