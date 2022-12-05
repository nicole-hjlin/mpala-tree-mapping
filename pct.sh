#!/bin/bash -x

#SBATCH --gres=gpu:3      
#SBATCH -p seas_gpu
#SBATCH -t 6:00:00         # Runtime in D-HH:MM:SS, minimum of 10 minutes
#SBATCH --mem=6000          # Memory pool for all cores (see also --mem-per-cpu) MBs
#SBATCH -o /n/home10/hongjinlin/outputs/cs288_pct_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e /n/home10/hongjinlin/outputs/cs288_pct_%j.err  # File to which STDERR will be written, %j inserts jobid

set -x

module load Anaconda3/2020.11
module load cuda/11.7.1-fasrc01
module load gcc/10.2.0-fasrc01
pip install ./PCT_Pytorch/pointnet2_ops_lib

# python3 -m pct_main --exp_name pct --batch_size 32 --test_batch_size 32 --epochs 10 --num_points 1000 --min_points 1000 --dropout 0.5 --data_dir ../data/MpalaForestGEO_LasClippedtoTreePolygons --label_path labels.csv --output_channels 6 --top_species 5
# python3 -m pct_main --exp_name pct --batch_size 32 --test_batch_size 32 --epochs 10 --num_points 1000 --min_points 1000 --dropout 0.5 --data_dir ../data/MpalaForestGEO_LasClippedtoTreePolygons --label_path labels.csv --output_channels 6 --top_species 5 --lr 1e-5
# python3 -m pct_main --exp_name pct --batch_size 32 --test_batch_size 32 --epochs 10 --num_points 1000 --min_points 1000 --dropout 0.2 --data_dir ../data/MpalaForestGEO_LasClippedtoTreePolygons --label_path labels.csv --output_channels 6 --top_species 5 --lr 1e-5
# python3 -m pct_main --exp_name pct --batch_size 32 --test_batch_size 32 --epochs 100 --num_points 1000 --min_points 1000 --dropout 0.5 --data_dir ../data/MpalaForestGEO_LasClippedtoTreePolygons --label_path labels.csv --output_channels 6 --top_species 5
python3 -m pct_main --exp_name pct --batch_size 32 --test_batch_size 32 --epochs 100 --num_points 1000 --min_points 1000 --dropout 0.5 --data_dir ../data/MpalaForestGEO_LasClippedtoTreePolygons --label_path labels.csv --output_channels 6 --top_species 5 --lr 1e-5
# python3 -m pct_main --exp_name pct --batch_size 32 --test_batch_size 32 --epochs 100 --num_points 1000 --min_points 1000 --dropout 0.2 --data_dir ../data/MpalaForestGEO_LasClippedtoTreePolygons --label_path labels.csv --output_channels 6 --top_species 5 --lr 1e-5


