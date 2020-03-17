module load conda/$2
conda activate py$1
pushd test
python -m unittest discover -v .
popd
