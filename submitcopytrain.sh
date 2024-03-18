#!/bin/bash
#SBATCH -N 1            # number of requested nodes. Set to 1 unless needed.  
#SBATCH -n 1            # number of tasks to run. Set to 1 unless needed. See granular resource allocation below for example.
#SBATCH -c 4            # number of requested CPUs
#SBATCH --mem=10g       # amount of memory requested (g=gigabytes)
#SBATCH -p qTRDGPUH         # partition to run job on. See "cluster and queue information" page for more information.
#SBATCH -t 1440         # time in minutes. After set time the job will be cancelled. See "cluster and queue information" page for limits.
#SBATCH -J hcpc
#SBATCH -e error%A.err  # errors will be written to this file. If saving this file in a separate folder, make sure the folder exists, or the job will fail
#SBATCH -o out%A.out    # output will be written to this file. If saving this file in a separate folder, make sure the folder exists, or the job will fail
#SBATCH -A psy53c17     # user group. See "requesting an account" page for list of groups
#SBATCH --mail-type=ALL # types of emails to send out. See SLURM documentation for more possible values
#SBATCH --mail-user=washbee1@student.gsu.edu # set this email address to receive updates about the job
#SBATCH --oversubscribe # see SLURM documentation for explanation

# it is a good practice to add small delay at the beginning and end of the job- helps to preserve stability of SLURM controller when large number of jobs fail simultaneously 
sleep 10s

# for debugging purpose- in case the job fails, you know where to look for possible cause
echo $HOSTNAME >&2

# run the actual job
cd /data/users2/washbee/speedrun/deepcsr-data
cp -avr $(cat trainpaths.txt) /data/users2/washbee/speedrun/cortexode-data-rp/train
# delay at the end (good practice)
sleep 10s

