#!/bin/bash -x

#SBATCH --gres=gpu:1      
#SBATCH -p seas_gpu_requeue
#SBATCH -t 18:00:00         # Runtime in D-HH:MM:SS, minimum of 10 minutes
#SBATCH --mem=6000          # Memory pool for all cores (see also --mem-per-cpu) MBs
#SBATCH -o /n/home10/hongjinlin/outputs/cs288_baseline_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e /n/home10/hongjinlin/outputs/cs288_baseline_%j.err  # File to which STDERR will be written, %j inserts jobid

set -x

module load Anaconda3/2020.11
module load gcc/12.1.0-fasrc01

# baseline with same tree same channel
python3 -m main --name baseline_channel --min_points 1000 --learning_rate 1e-4 --epochs 100 --batch_size 32 --data_dir ../data/MpalaForestGEO_LasClippedtoTreePolygons --label_path labels.csv --top_species 5 --spicy
