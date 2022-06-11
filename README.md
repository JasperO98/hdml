# Install Conda
Install conda at: https://www.anaconda.com/products/distribution

# Install Dependencies
Open a terminal in directory and enter the following commands:

```
 conda config --add channels conda-forge
 conda config --add channels bioconda
 conda create -y -n my_env python=3.8.13
 conda activate my_env
 conda install -y catboost xgboost lightgbm scikit-learn scipy numpy pandas matplotlib seaborn pillow tqdm brotlipy jupyter
 conda install -y pydot graphviz
 conda install -y scikit-learn-intelex 
 pip install Flask-SQLAlchemy
 pip install plot_keras_history
```

# Tensorflow (GPU)
Install via tensorflow using the commands below.
For more detailed installation instructions go to: https://www.tensorflow.org/install/pip

```
# Check if drivers are installed
nvidia-smi
# Install
conda install -y cudatoolkit=11.2 cudnn=8.1.0
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
pip install tensorflow
# Verify install:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
