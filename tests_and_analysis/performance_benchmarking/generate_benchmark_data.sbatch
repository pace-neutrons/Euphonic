#!/bin/bash
#SBATCH -C scarf18
#SBATCH -p devel
#SBATCH -c 24
#SBATCH -n 1
#SBATCH -t 1:00:00
#SBATCH --job-name=euphonic_perform
#SBATCH --exclusive
#SBATCH -v
#SBATCH -o %j.log
#SBATCH -e %j.err

sudo /usr/local/bin/enable_no_turbo.sh
/usr/local/bin/show_no_turbo.sh

module load python/3.6.5-gcc-4.8.5
python -m pip install --user virtualenv
python -m virtualenv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pushd ../..
python -m pip -v install -e .[matplotlib,phonopy_reader]
popd
echo "Generating data"
python generate_benchmark_data.py 
echo "Generated benchmark data"
deactivate

sudo /usr/local/bin/disable_no_turbo.sh

