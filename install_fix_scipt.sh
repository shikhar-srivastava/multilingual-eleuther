/source /scratch/ssrivas9/miniconda3/bin/activate multi
/scratch/ssrivas9/miniconda3/envs/multi/bin/python -m pip install -U pip setuptools wheel
/scratch/ssrivas9/miniconda3/envs/multi/bin/python -m pip install --only-binary=:all: "pyarrow>=16.1,<18"