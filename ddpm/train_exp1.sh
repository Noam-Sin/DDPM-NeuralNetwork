#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
apptainer exec --nv --bind /projects:/projects /opt/containers/pytorch-25.04.sif bash -c "cd /projects/nn-bsc/shahar.girtler/DDPM-NeuralNetwork/ddpm && python train.py --config configs/config_exp1.py"
