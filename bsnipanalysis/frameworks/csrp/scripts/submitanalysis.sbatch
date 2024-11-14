#!/bin/bash
#SBATCH -p qTRDHM
#SBATCH -A psy53c17
#SBATCH --job-name=alyzsub_csrp
#SBATCH --output=jobs/out%A_%a.out
#SBATCH --error=jobs/err%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --mem=5G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe

source /data/users2/washbee/anaconda3_refresh/bin/activate bsnip
cd /data/users2/washbee/CortexODE-CSRFusionNet/bsnipanalysis/frameworks/csrp/scripts/

python analysis.py --data_dir '/data/users2/washbee/CortexODE-CSRFusionNet/bsnipanalysis' --data_name 'bsnip' --init_dir '/data/users2/washbee/CortexODE-CSRFusionNet/ckpts/bsnip/bsnip_0/init' --atlas 'DKTatlas' --device 'cpu'
