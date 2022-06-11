from os.path import isdir, join
from os import mkdir
from preprocess import PreProcessor

from keras.layers import GRU

from models.RNN import RNN
import itertools


def RNN_args():
    """ Get GNN model and hyperparameters """
    train_args = dict(layers=3, recurrent_layer=GRU, bidirect=False,
                      hidden_size=256, L2=1e-3,
                      drop=0.0, rdrop=0.0, lr=1e-5
                      )
    parameters_names = ['layers', 'recurrent_layer', 'hidden_size', 'L2']
    layers = [3, 5]
    recurrent_layers = [GRU]
    hidden_sizes = [128, 256, 512]
    L2 = [1e-7, 1e-5, 1e-3]
    parameters = itertools.product(*[layers, recurrent_layers, hidden_sizes, L2])
    return RNN, train_args, parameters, parameters_names


def tune(data, outdir, tuning_model, model_args, all_parameters, names):
    """ Tune the specific model with all the given combination of hyperparameters """
    for parameters in all_parameters:
        print(names, parameters)
        for name, param in zip(names, parameters):
            print(name, param)
            model_args[name] = param
        model = tuning_model(data=data, outdir=outdir, load_from_dir=False, **model_args)
        model.train(maxiter=10000, batch_size=128, patience=100, freq=1, verbose=1)
    return


if __name__ == '__main__':
    # Change dir to root
    pred = ['drive']
    prob = ['classification']

    # Create outdir if needed output/label
    out = join('output', '_'.join(pred), 'tuning')
    if not isdir(out):
        mkdir(out)

    # How to load model from directory
    load_from_dir = False

    # Give complete path of models to load
    if load_from_dir:
        out = join('output', 'drive', 'GRU_RNN', 'model_0')

    # Get the data for the RNN predictions (predict_ml=False)
    data = PreProcessor(infile='data/imputed_pre_and_manifest.csv', outdir='output/figures',
                        mask_value=-2.0,
                        mask_file='data/filtered_pre_and_manifest.csv',
                        timestep_size=365, max_timestep=5, filt_outliers=False, predict_ml=False,
                        labels=pred, problem=prob, mask_label=True, shift=0, scaler_type='norm')

    # Tune RNN
    rnn, train_args, parameters, parameters_names = RNN_args()
    tune(data, out, rnn, train_args, parameters, parameters_names)
