module load conda/$2
module load gcc
conda create --name py$1 python=$1 -y

conda activate py$1 &&

export CC=gcc &&

python -m pip install --upgrade --user pip &&
python -m pip install --user numpy &&
python -m pip install --user .[matplotlib] &&
python -m pip install --user mock
