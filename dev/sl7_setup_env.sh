# Install pyenv dependencies
sudo yum install -y zlib-devel bzip2 bzip2-devel readline-devel sqlite sqlite-devel openssl-devel xz xz-devel

# Install pyenv to enable installation of different Python versions
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bash_profile
source ~/.bash_profile # Reload shell vars

# Install pyenv-virtualenv as pyenv plugin
git clone https://github.com/pyenv/pyenv-virtualenv.git $PYENV_ROOT/plugins/pyenv-virtualenv
source ~/.bash_profile # Reload shell vars

# 'pyenv install' requires a C compiler
sudo yum install -y gcc

# Install Python versions
pyenv install 2.7.13
pyenv install 3.6.0
pyenv global 2.7.13 # Set pyenv version
pip install --upgrade pip

# Install Jupyter
pip install jupyter

# Start the notebook
jupyter notebook --ip=`hostname` --notebook-dir=../src &

# Set up environments for Python 2 and 3
py_vers=("2.7.13" "3.6.0")
for ver in "${py_vers[@]}"
do
    env_name="disp_dos_$ver"

    # Create environment
    pyenv virtualenv $ver $env_name

    # Install packages in environment
    source ~/.bash_profile # Reload shell vars
    pyenv activate $env_name
    pip install --upgrade pip
    pip install ipykernel
    python -mpip install matplotlib
    # ** install any other required packages here **
    pyenv deactivate

    # Add Jupyter kernel containing environment
    KERNEL_DIR="$HOME/.local/share/jupyter/kernels"
    mkdir -p "$KERNEL_DIR/$env_name"
    cat <<EOF >$KERNEL_DIR/$env_name/kernel.json
    {
     "argv": [ "$PYENV_ROOT/versions/$env_name/bin/python", "-m", "ipykernel",
               "-f", "{connection_file}"],
     "display_name": "$env_name",
     "language": "python"
    }
EOF
done
