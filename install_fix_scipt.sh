/source /localdisk/ssrivas9/miniconda3/bin/activate multi
/localdisk/ssrivas9/miniconda3/envs/multi/bin/python -m pip install -U pip setuptools wheel
/localdisk/ssrivas9/miniconda3/envs/multi/bin/python -m pip install --only-binary=:all: "pyarrow>=16.1,<18"