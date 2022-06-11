from os.path import isdir, join
from os import mkdir, listdir, chdir

from keras.layers import SimpleRNN, LSTM, GRU

from preprocess import PreProcessor
from plot import Plotter
from models.RNN import RNN


if __name__ == '__main__':
    # Change dir to root
    pred = ['drive']
    prob = ['classification']

    # Create outdir if needed output/label
    out = join('output', '_'.join(pred))
    if not isdir(out):
        mkdir(out)

    # How to load model from directory
    load_from_dir = False

    # Give complete path of models to load
    if load_from_dir:
        out = join('output', 'drive', 'GRU_RNN', 'model_0')

    # Load data, depends on output and model
    data = PreProcessor(infile='data/imputed_pre_and_manifest.csv', outdir='output/figures',
                        mask_value=-2.0,
                        mask_file='data/filtered_pre_and_manifest.csv',
                        timestep_size=365, max_timestep=5, filt_outliers=False, predict_ml=False,
                        labels=pred, problem=prob, mask_label=True, shift=0, scaler_type='norm')

    # Loading training arguments
    if load_from_dir:
        train_args = dict()
    else:
        train_args = dict(layers=3, recurrent_layer=GRU, bidirect=False,
                          hidden_size=256, L2=1e-7,
                          drop=0.0, rdrop=0.0, lr=1e-5
                          )

    # Load model
    model = RNN(data=data, outdir=out, load_from_dir=load_from_dir, **train_args)
    if not load_from_dir:
        model.train(maxiter=10000, batch_size=128, patience=100, freq=1, verbose=1)

    # Load plotter
    plotter = Plotter(data=data, model=model, outdir=model.outdir, save_mode=True, show_mode=False)

    plotter.forecasting_measures(subset='test', label='drive')
    # Make plots
    if pred[0] == 'drive':
        plotter.roc(label='drive', subset='test')
        plotter.confusion_matrix(subset='test', label='drive')
        for i in range(30):
            plotter.forecasting_plot(subset='test', label='drive', select='change', complete=True)
    else:
        plotter.ae_distribution('test', pred[0])
