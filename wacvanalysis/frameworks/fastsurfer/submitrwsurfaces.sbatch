#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH -A psy53c17
#SBATCH --output=jobs/out%a_%A.out
#SBATCH --error=jobs/err%a_%A.err
#SBATCH --job-name=rwfs
#SBATCH -p qTRD

# Load required modules
source /data/users2/washbee/anaconda3latest/bin/activate wacvanalysis

# Install necessary Python packages

# Define paths
SUBJECTS_DIR=/data/users2/washbee/fastsurfer-output/test
DEST_DIR=/data/users2/washbee/wacv/surfaces/fastsurfer
SUBJECT_LIST=/data/users2/washbee/speedrun/cortexode-data-rp/test/all_subjects.txt

# Get subject ID from the list
SUBJECT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $SUBJECT_LIST)

# Run the Python script
python rwsurfaces.py --subject-dir $SUBJECTS_DIR --dest-dir $DEST_DIR --subject $SUBJECT
