#!/bin/bash
#SBATCH --job-name=fedkseed
#SBATCH --account=project_2009050
#SBATCH --partition=gpu
#SBATCH --time=15:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8000
##SBATCH --gres=gpu:v100:1,nvme:10
#SBATCH --output=logs/fedkseed.out
#SBATCH --error=logs/fedkseed.err
#SBATCH --mail-type=BEGIN


#module load myprog/1.2.3
cd /projappl/project_2009050/code/FedKseed
source /projappl/project_2009050/python_envs/bin/activate
srun main.py --fname ./configs/fedk/configs.yaml
