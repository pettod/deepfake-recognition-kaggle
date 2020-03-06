#!/bin/bash
#
# At first let's give job some descriptive name to distinct from other jobs running on cluster
sbatch -J deepfake
#
# Let's redirect job's out some other file than default slurm-%jobid-out
#sbatch --output=slurm.out
#sbatch --error=slurm.err
#
# We'll want mail notifications from job errors and ending
sbatch --mail-user=peter.todorov@tuni.fi
sbatch --mail-type=END
#
# We'll only need one cpu, because GPU should do most of the work
sbatch --ntasks=1
sbatch --cpus-per-task=1
#
# We'll want to reserve 2GB memory for job (per node), and
# we estimate that job should finish in maximum of 8 hours and 15 minutes.
sbatch --time=8:15:00
sbatch --mem=16384
#
# We'll request one Tesla V100 gpu from the gpu partition.
# (actually you don't even need to specify partition name, because gpu's are only available there anyway.)
sbatch --partition=gpu --gres=gpu:teslav100:1
#
# Let's load cuda environment
module load CUDA
#
# And finally we'll start our program
python resnet_code/resnet50_train.py