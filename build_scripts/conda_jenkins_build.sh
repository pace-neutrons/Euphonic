pushd ..

module load conda/3
module load gcc


conda create --name py python=$1 -y # $1 holds python version
conda activate py

export CC=gcc

python -m pip install --upgrade --user pip
python -m pip install --user numpy
python -m pip install --user .[matplotlib]
python -m pip install --user mock

popd
