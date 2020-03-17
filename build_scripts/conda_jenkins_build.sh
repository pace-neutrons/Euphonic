module load conda/3
module load gcc


conda create --name py3 python=3.6.0 -y
conda activate py3

export CC=gcc

python -m pip install --upgrade --user pip
python -m pip install --user numpy
python -m pip install --user .[matplotlib]
python -m pip install --user mock
