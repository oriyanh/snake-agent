#!/bin/csh

#SBATCH --cpus-per-task=2
#SBATCH --output=~/Courses/APML/snake-apml/out
#SBATCH --mem-per-cpu=500M
#SBATCH --account=aml
#SBATCH --constraint="sm"

source /cs/labs/dshahaf/omribloch/env/snake/snake/bin/activate.csh
module load tensorflow

python3 ~/Courses/APML/snake-apml/Snake.py -P "Linear()" -D 5000 -s 1000 -l "~/Courses/APML/snake-apml/out/game_linear.log" -o "~/Courses/APML/snake-apml/out/game_linear.out" -r 0 -plt 0.01 -pat 0.005 -pit 20
