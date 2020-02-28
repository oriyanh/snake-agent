#!/bin/csh

#SBATCH --cpus-per-task=2
#SBATCH --output=~/Courses/APML/snake-apml/out
#SBATCH --mem-per-cpu=500M
#SBATCH --account=aml
#SBATCH --constraint="sm"
#SBATCH --nodelist="sm-17"

source /cs/labs/dshahaf/omribloch/env/snake/snake/bin/activate.csh
module load tensorflow
python3 ~/Courses/APML/snake-apml/Snake.py -P "Linear();Avoid(epsilon=0);Avoid(epsilon=0.05);Avoid(epsilon=0.1);Avoid(epsilon=0.2)"  -D 5000 -s 1000 -l "/cs/usr/oriyanh/Courses/APML/snake-apml/out/game_linear.log" -o "/cs/usr/oriyanh/Courses/APML/snake-apml/out/game_linear.out" -r 0 -plt 0.2 -pat 0.2 -pit 5
python3 ~/Courses/APML/snake-apml/Snake.py -P "Custom();Avoid(epsilon=0);Avoid(epsilon=0.05);Avoid(epsilon=0.1);Avoid(epsilon=0.2)"  -D 50000 -s 5000 -l "/cs/usr/oriyanh/Courses/APML/snake-apml/out/game_custom.2log" -o "/cs/usr/oriyanh/Courses/APML/snake-apml/out/game_custom2.out" -r 0 -plt 0.2 -pat 0.2 -pit 5


