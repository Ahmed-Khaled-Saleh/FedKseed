#!/bin/bash
#SBATCH --account=project_2009050
#SBATCH --job-name=fedkseed
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=320G
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --output=logs/fedkseed.out
#SBATCH --error=logs/fedkseed.err
# # SBATCH --mail-type=BEGIN

module --force purge
module load pytorch
source /projappl/project_2009050/torch/bin/activate
cd /projappl/project_2009050/code/FedKseed

export PYTHONPATH=$PYTHONPATH:/projappl/project_2009050/torch/lib/python3.9/site-packages
echo "Current PYTHONPATH: $PYTHONPATH"
python -c "import rouge; print('rouge module is installed and importable')"

srun torchrun --nnodes=1 --nproc_per_node=2  --rdzv-backend=c10d --rdzv-endpoint=localhost:0 main_dist.py --fname ./configs/fedk/configs_instruct_motivation.yaml
