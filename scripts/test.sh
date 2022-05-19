#!/bin/bash

#PBS -P p00
#PBS -q normal
#PBS -l ncpus=240
#PBS -l walltime=48:00:00
#PBS -l software=python
#PBS -l mem=300GB
#PBS -l jobfs=400GB
#PBS -l wd
#PBS -M u6774588@anu.edu.au
#PBS -l storage=scratch/p00+gdata/p00
#PBS -r n
#PBS -j oe

module load intel-mkl/2019.3.199
module load python3/3.9.2
module load hdf5/1.10.5
module load netcdf/4.7.3
module load openmpi/4.0.2

pip install --user scipy
pip install --user scikit-learn
pip install --user tensorflow-cpu
pip install --user keras
pip install --user pandas
pip install --user matplotlib
pip install --user Pillow
pip install --user numpy
pip install --user pandas
pip install --user modAL
pip install --user mpi4py
pip install --user xlrd
pip install --user openpyxl

export OMP_NUM_THREADS=$PBS_NCPUS
export I_MPI_PIN=off
export PYTHONPATH=/g/data/p00/hd8710/tmp/active_learning:$PYTHONPATH
mpirun -np $PBS_NCPUS python3 _parallel.py > $PBS_JOBID.log