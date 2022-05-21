# Standard
from os.path import isfile, isdir, join
from time import time
import shutil
import pickle
import numpy as np

# Inherited class
from .BaseModel import Base

# Random seeds
from os import environ
from random import seed as rseed
import tensorflow as tf

# Plotting
import matplotlib.pyplot as plt
from plot_keras_history import plot_history
from keras.utils.vis_utils import plot_model

# Neural Networks
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Masking, TimeDistributed, Dense, Input, GRU, Reshape, Bidirectional, Flatten
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import *
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import Callback, EarlyStopping


# Define seeds
seed_value = 0

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
environ['PYTHONHASHSEED'] = str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
rseed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
tf.compat.v1.set_random_seed(seed_value)


class ScaleMetrics(Callback):
    def __init__(self, target, numeric, transformer=lambda x, t: x, interval=1):
        """
        :param target: For which labels the evaluation metrics should be scaled
        :param numeric: The numeric columns in the data
        :param transformer: Function to rescale the data (self.data.inverse_transform)
        :param interval: How many epochs need to elapse before printing to screen

        This callback undo's the normalization of all evaluation metrics, so that they are more interpretable.
        """
        super(Callback, self).__init__()
        self.transformer = transformer
        self.labels = target
        self.numeric = numeric
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        """ At the end of each epoch rescale the measures using the transformer """
        if epoch % self.interval == 0:
            print("Epoch {}, Loss: {:.5f} ({:.5f}) - ".format(epoch, logs["loss"], logs["val_loss"]) +
                  ' - '.join(['{}: {:.3f} ({:.3f})'
                             .format(t + '_' + m,
                                     self.transformer(logs[t + '_' + m], t),
                                     self.transformer(logs['val_' + t + '_' + m], t),
                                     )
                              for t in self.labels for m in ['MAE', 'RMSE', 'Accuracy', 'Recall', 'Precision', 'AUC',
                                                             'TP', 'TN', 'FP', 'FN',
                                                             ]
                              if t + '_' + m in logs.keys() and 'val_' + t + '_' + m in logs.keys()
                              ])
                  )


class RNN(Base):
    def __init__(self, data, outdir, load_from_dir, shap=False, layers=3, hidden_size=512, recurrent_layer=GRU,
                 bidirect=False, L2=1e-5, drop=0.4, rdrop=0.4, lr=1e-4):
        """
        :param data: class variable from PreProcessor
        :param outdir: in which directory to save the model (output/label) or where to load the model (complete path)
        :param load_from_dir: whether to load the model or to create a new one
        :param shap: whether to create SHAP model or normal model (relates to flattening the output of the model)
        :param layers: number of dense and dropout layer that should be stacked
        :param hidden_size: the number of units in the dense layers
        :param recurrent_layer: type of RNN, either: SimpleRNN, LSTM, or GRU
        :param bidirect: whether to make the model bidirectional or not (not used)
        :param L2: the L2 regularization value
        :param drop: dropout rate
        :param rdrop: recurrrent dropout rate
        :param lr: learning rate

        Class for the RNN models (SimpleRNN, LSTM, RNN). Here the recurrent neural network is created or loaded.
        If the model is loaded the given training parameters are ignored.
        """
        super().__init__(data, outdir, load_from_dir, prefix=recurrent_layer.__name__ + '_')

        # Load arguments from file if model is loaded
        if load_from_dir:
            print('Overwriting training params')
            train_args = pickle.load(open(join(self.outdir, 'train_args.pickle'), 'rb'))
            recurrent_layer = train_args['recurrent_layer']
            bidirect = train_args['bidirect']
            L2 = train_args['L2']
            drop = train_args['drop']
            rdrop = train_args['rdrop']
            lr = train_args['lr']
            layers = train_args['layers']
            hidden_size = train_args['hidden_size']
        # Save arguments if new model is created
        else:
            self.create_outdir()
            # Save kwargs
            with open(join(self.outdir, 'train_args.pickle'), 'wb') as outfile:
                vars = [layers, hidden_size, recurrent_layer, bidirect, L2, drop, rdrop, lr]
                names = ['layers', 'hidden_size', 'recurrent_layer', 'bidirect', 'L2', 'drop', 'rdrop', 'lr']
                pickle.dump({names[i]: vars[i] for i in range(len(vars))}, outfile)

        # Initialize values
        self.recurrent_layer = recurrent_layer
        self.bidirect = bidirect
        self.L2 = L2
        self.drop = drop
        self.rdrop = rdrop
        self.lr = lr
        self.layers = layers
        self.hidden_size = hidden_size

        # Get SHAP or normal model
        if shap:
            self.model = self.shap_model(timesteps=self.data.timesteps)
        else:
            self.model = self.get_model(timesteps=None)
        self.model.summary()

        # If model is loaded from file, load the trained weights
        if self.load_from_dir:
            self.model.load_weights(join(self.outdir, 'model', 'weights.h5'))
        # Else plot the layout of the model
        else:
            plot_model(self.model, to_file=join(self.outdir, 'figures', 'model.png'), show_shapes=False,
                       show_layer_names=False)

    def train(self, maxiter, batch_size, patience, freq, verbose=1):
        """
        :param maxiter: Maximum number of iterations
        :param batch_size: Batch size to train on
        :param patience: Early stopping patience
        :param patience: Early stopping patience
        :param freq: frequency of epochs to print to command line (1=each epoch)

        Trains and saves the model and creates figures.
        """
        assert not self.load_from_dir, 're-training is not supported!'

        # Callback classes for the model
        # Early stopping prevents overfitting
        early_stop_monitor = EarlyStopping(monitor='val_loss', mode='min',
                                           patience=patience, restore_best_weights=True)
        scale_monitor = ScaleMetrics(target=self.data.labels, numeric=self.data.cols[self.data.numeric_cols],
                                     interval=freq, transformer=self.data.inverse_transform)

        # Load model and give the number of time steps it should expects, makes training faster.
        self.model = self.get_model(timesteps=self.data.timesteps)
        start = time()
        try:
            if verbose == 1:
                history = self.model.fit(self.data.x_train, self.flatten_dict(self.data.y_train),
                                         sample_weight=self.flatten_dict(self.data.train_weights),
                                         batch_size=batch_size, epochs=maxiter,
                                         validation_data=(self.data.x_test, self.flatten_dict(self.data.y_test),
                                                          self.flatten_dict(self.data.test_weights)),
                                         validation_batch_size=batch_size,
                                         verbose=0,
                                         shuffle=True,
                                         callbacks=[scale_monitor, early_stop_monitor,
                                                    # transition_metrics if 'drive' in self.labels else None
                                                    ],
                                         )
            else:
                history = self.model.fit(self.data.x_train, self.flatten_dict(self.data.y_train),
                                         sample_weight=self.flatten_dict(self.data.train_weights),
                                         batch_size=batch_size, epochs=maxiter,
                                         validation_data=(self.data.x_test, self.flatten_dict(self.data.y_test),
                                                          self.flatten_dict(self.data.test_weights)),
                                         validation_batch_size=batch_size,
                                         # validation_freq=freq,
                                         verbose=0,
                                         shuffle=True,
                                         callbacks=[early_stop_monitor],
                                         )
        except KeyboardInterrupt:
            try:
                shutil.rmtree(self.outdir)
            except OSError as e:
                print("Error: %s : %s" % (self.outdir, e.strerror))
            exit(0)

        # Training time
        elapsed = time() - start
        print(self.convert(elapsed))

        # Define stopped epoch
        best_epoch = np.argmin(history.history['val_loss'])
        print('Model epoch:', best_epoch)

        # log
        self.log(elapsed, str(len(history.history['loss'])), str(best_epoch))

        # Model has to be transformed (timesteps=None) to save the weights
        new_weights = self.model.get_weights()
        self.model = self.get_model(timesteps=None)
        self.model.set_weights(new_weights)
        self.model.save_weights(join(self.outdir, 'model', 'weights.h5'))

        # Transform history metrics
        for p, l in zip(self.data.problem, self.data.labels):
            if p == 'regression' and l in self.data.cols[self.data.numeric_cols]:
                for m in ['MAE', 'RMSE']:
                    history.history[l + '_' + m] = self.data.inverse_transform(history.history[l + '_' + m], l)
                    history.history['val_' + l + '_' + m] = self.data.inverse_transform(
                        history.history['val_' + l + '_' + m], l)

        # Plot history
        history_stats = {k: history.history[k] for k in history.history.keys()
                         if k.split('_')[-1] not in ['Accuracy', 'Recall', 'Precision']}
        plot_history(history_stats, path=join(self.outdir, 'figures'), single_graphs=True)
        for k in range(len(history_stats.keys()) // 2):
            plt.close()

        # History
        history.history['best_epoch'] = best_epoch
        with open(join(self.outdir, 'model', 'history.pickle'), 'wb') as outfile:
            pickle.dump(history.history, outfile)

        label = self.data.labels[0]
        print(label, self.evaluate(self.data.y_test[label], self.custom_predict(self.data.x_test)[0],
                                   self.data.problem[0], self.data.labels[0], self.data.test_weights[label]))

    def get_model(self, timesteps=None):
        """
        :param timesteps: Number of timesteps the model should expect (None=Flexible). Makes training faster if known.

        The input layers consist of a simple input layer and a masking layer.
        The masking layer ensures that missing visits are skipped (visits with only the masking value, self.mask)
        The number of output layers depend on the number of labels to classify, one layer for each label.
        The output layers are always TimeDistributed layers, where the activation depends on the problem type.
        :return: a RNN
        """
        print('Building model...')
        # If shap input size needs to be known
        inp = Input(shape=(timesteps, len(self.data.cols)))  # , batch_size=None if train else 1)

        mask = Masking(mask_value=self.data.mask)(inp)
        network = [mask]

        # Define RNN configuration
        rnn_config = self.recurrent_layer(self.hidden_size,
                                          dropout=self.drop, recurrent_dropout=self.rdrop,
                                          kernel_regularizer=l2(self.L2),
                                          unroll=True if timesteps is not None else False,
                                          return_sequences=True,
                                          ).get_config()
        del (rnn_config['name'])

        # Add recurrent layers to network
        for i, lay in enumerate(range(self.layers)):
            if self.bidirect:
                network.append(Bidirectional(self.recurrent_layer.from_config(rnn_config))(network[-1]))
            else:
                network.append(self.recurrent_layer.from_config(rnn_config)(network[-1]))

        # Define output layers
        out = []
        wmetrics = {}
        losses = {}
        output_config = Dense(1, activation='linear', kernel_regularizer=l2(self.L2)).get_config()
        for l in range(len(self.data.labels)):
            # Regression output
            if self.data.problem[l] == 'regression':
                wmetrics[self.data.labels[l]] = [
                    RootMeanSquaredError(
                        name=self.data.labels[l] + '_' + 'RMSE' if len(self.data.labels) == 1 else 'RMSE'),
                    MeanAbsoluteError(name=self.data.labels[l] + '_' + 'MAE' if len(self.data.labels) == 1 else 'MAE'),
                ]
                losses[self.data.labels[l]] = 'mean_squared_error'

            # Classification output
            elif self.data.problem[l] == 'classification':
                output_config['activation'] = 'sigmoid'
                wmetrics[self.data.labels[l]] = [
                    BinaryAccuracy(
                        name=self.data.labels[l] + '_' + 'Accuracy' if len(self.data.labels) == 1 else 'Accuracy'),
                    Precision(
                        name=self.data.labels[l] + '_' + 'Precision' if len(self.data.labels) == 1 else 'Precision'),
                    Recall(name=self.data.labels[l] + '_' + 'Recall' if len(self.data.labels) == 1 else 'Recall'),
                    AUC(name=self.data.labels[l] + '_' + 'AUC' if len(self.data.labels) == 1 else 'AUC'),
                ]
                losses[self.data.labels[l]] = 'binary_crossentropy'
            # Add output layer
            out_layer = TimeDistributed(Dense.from_config(output_config), name=self.data.labels[l])(network[-1])
            out.append(out_layer)

        # Output shape needs to be 2d
        model = Model(inp, out)

        model.compile(loss=losses,
                      optimizer=Adam(learning_rate=self.lr),  # default: 0.001
                      weighted_metrics=wmetrics,
                      sample_weight_mode='temporal',
                      )
        return model

    def shap_model(self, timesteps=None):
        """
        Same as get_model(), however the output is flattened, which is required by the SHAP algorithm.
        :return: a RNN
        """
        # Add to notebook
        print('Building SHAP model...')
        # Number of features depends on wether or not the label is added as an input value
        # features = len(self.cols) if self.mask_label else len(self.cols) - 1

        # If shap input size needs to be known
        inp = Input(shape=(timesteps, len(self.data.cols)))  # , batch_size=None if train else 1)

        # mask = Masking(mask_value=self.data.mask)(inp)
        # network = [mask]

        network = [inp]

        # Add recurrent layers to network
        for lay in range(self.layers):
            if self.bidirect:
                network.append(Bidirectional(
                    self.recurrent_layer(self.hidden_size,
                                         return_sequences=True,
                                         dropout=self.drop, recurrent_dropout=self.rdrop,
                                         kernel_regularizer=l2(self.L2),
                                         unroll=True if timesteps is not None else False,
                                         ))(network[-1]))
            else:
                network.append(self.recurrent_layer(self.hidden_size,
                                                    return_sequences=True,
                                                    dropout=self.drop, recurrent_dropout=self.rdrop,
                                                    kernel_regularizer=l2(self.L2),
                                                    unroll=True if timesteps is not None else False,
                                                    )(network[-1]))

        out = []
        wmetrics = {}
        losses = {}
        for l in range(len(self.data.labels)):
            if self.data.problem[l] == 'regression':
                out_layer = TimeDistributed(Dense(1, activation='linear', kernel_regularizer=l2(self.L2)))(network[-1])
                out.append(Reshape((timesteps,), name=self.data.labels[l])(out_layer))
                wmetrics[self.data.labels[l]] = [
                    RootMeanSquaredError(
                        name=self.data.labels[l] + '_' + 'RMSE' if len(self.data.labels) == 1 else 'RMSE'),
                    MeanAbsoluteError(name=self.data.labels[l] + '_' + 'MAE' if len(self.data.labels) == 1 else 'MAE'),
                ]
                losses[self.data.labels[l]] = 'mean_squared_error'
            elif self.data.problem[l] == 'classification':
                out_layer = TimeDistributed(Dense(1, activation='sigmoid', kernel_regularizer=l2(self.L2)))(network[-1])
                out.append(Reshape((timesteps,), name=self.data.labels[l])(out_layer))
                wmetrics[self.data.labels[l]] = [
                    BinaryAccuracy(
                        name=self.data.labels[l] + '_' + 'Accuracy' if len(self.data.labels) == 1 else 'Accuracy'),
                    Precision(
                        name=self.data.labels[l] + '_' + 'Precision' if len(self.data.labels) == 1 else 'Precision'),
                    Recall(name=self.data.labels[l] + '_' + 'Recall' if len(self.data.labels) == 1 else 'Recall'),
                    AUC(name=self.data.labels[l] + '_' + 'AUC' if len(self.data.labels) == 1 else 'AUC'),
                ]
                losses[self.data.labels[l]] = 'binary_crossentropy'

        # Output shape needs to be 2d
        model = Model(inp, out)

        model.compile(loss=losses,
                      optimizer=Adam(learning_rate=self.lr),  # default: 0.001
                      weighted_metrics=wmetrics,
                      sample_weight_mode='temporal',
                      )
        return model
