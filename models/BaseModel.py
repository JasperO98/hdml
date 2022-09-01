# Standard
from os import environ, mkdir, listdir
from os.path import isfile, join, dirname, basename, isdir
from time import time
import shutil
import numpy as np
import pickle

# Random seeds
from random import seed as rseed

# Metrics
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, accuracy_score
from sklearn.metrics import r2_score, mean_absolute_error, max_error, mean_squared_error

# Tensorflow
import tensorflow as tf


class Base:
    def __init__(self, data, outdir, load_from_dir, prefix=''):
        """
        :param data: class variable from PreProcessor
        :param outdir: in which directory to save the model (output/label) or where to load the model (complete path)
        :param load_from_dir: whether to load the model or to create a new one
        :param prefix: prefix for the model name

        Base class of all the models.
        """
        # CPU or GPU setup
        environ["CUDA_VISIBLE_DEVICES"] = "{0-7}"

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

        # GPU setup
        print(tf.config.experimental.list_physical_devices())
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print('Basemodel Random Seed set to', seed_value)

        # Initialize values
        if load_from_dir:
            self.check_files(outdir)
        else:
            model_dir = join(outdir, prefix + self.__class__.__name__)
            if not isdir(model_dir):
                mkdir(model_dir)
            print(model_dir)
            name = self.get_outdir_name(model_dir)
            outdir = join(model_dir, name)
            print(outdir)
        self.load_from_dir = load_from_dir
        self.outdir = outdir
        self.data = data
        self.model = None

    @staticmethod
    def check_files(outdir):
        """ When loading a model all the required files are checked"""
        assert isdir(outdir), "can't load from {} if directory does not exists".format(outdir)
        assert isfile(join(outdir, 'train_args.pickle')), "can't load {} if file does not exists". \
            format(join(outdir, 'train_args.pickle'))
        assert isdir(join(outdir, 'model')), "can't load model if {} does not exist".format(join(outdir, 'model'))
        assert isfile(join(outdir, 'model', 'weights.h5')) or isfile(join(outdir, 'model', 'trained_model.pickle')), (
            "Can't load model if weights.h5 (DNN) or trained_model.pickle (ML) does not exist."
        )

    @staticmethod
    def get_outdir_name(out):
        """ Create outdir directory """
        i = 0
        name = 'model_{}'.format(i)
        print(listdir(out))
        while name in listdir(out):
            i += 1
            name = 'model_{}'.format(i)
        return name

    def create_outdir(self):
        """
        Creates the directories in outdir where all figures, tables and the models are saved.
        """
        mkdir(join(self.outdir))
        mkdir(join(self.outdir, 'model'))
        mkdir(join(self.outdir, 'figures'))
        mkdir(join(self.outdir, 'tables'))

    def flatten_dict(self, x):
        """ Flatten the dictionary of sample weights/labels"""
        x = x.copy()
        for lab in self.data.labels:
            shape = (x[lab].shape[0], x[lab].shape[1] * x[lab].shape[2], 1)
            values = x[lab]
            x[lab] = np.empty(shape)
            x[lab] = values.reshape(shape)
        return x

    def custom_predict(self, x):
        """
        Instead of using the keras predict function use custom_predict(), this reshapes it in the correct way.
        The reshape is basically that the predictions for each label are put in a list, e.g.:
        [y_pred of label1, y_pred of label2, ...]
        Depends on how many different labels the model should predict
        """
        pred = self.model.predict(x)
        shape = (x.shape[0], -1, self.data.cons_t)
        if isinstance(pred, list):
            return [p.reshape(shape) for p in pred]
        else:
            return [pred.reshape(shape)]

    @staticmethod
    def convert(seconds):
        """ Converts execution time of model in seconds to h:m:s"""
        seconds = seconds % (24 * 3600)
        hour = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        return '{:.0f}h {:.0f}m {:.0f}s'.format(hour, minutes, seconds)

    def log(self, elapsed, total_epochs, best_epoch):
        """
        Save basic stats of trained model, this includes:
        - model
        - epochs
        - best epoch (where model stopped learning)
        - elapsed time (seconds)
        - format time (h:m:s)
        """
        # Get log file name
        log = join(dirname(self.outdir), 'log.csv')
        empty = not isfile(log)
        # Add columns to log if log file is empty
        cols = []
        for d in ['train', 'test']:
            for l, p in zip(self.data.labels, self.data.problem):
                if p == 'classification':
                    for m in ['AUC', 'Accuracy', 'F1']:
                        cols.append(d + '_' + l + '_' + m)
                elif p == 'regression':
                    for m in ['MAE', 'RMSE', 'Max AE', 'R2']:
                        cols.append(d + '_' + l + '_' + m)

       # Add to info to log
        with open(log, 'a') as outfile:
            if empty:
                outfile.write(','.join(['model', 'epochs', 'best epoch', 'elapsed time', 'format time'] + cols) + '\n')
            measures = []
            for d, x, y, w in zip(['train', 'test'],
                                  [self.data.x_train.copy(), self.data.x_test.copy()],
                                  [self.data.y_train, self.data.y_test],
                                  [self.data.train_weights, self.data.test_weights]):
                for i, prob, lab in zip(range(len(self.data.labels)), self.data.problem, self.data.labels):
                    lab_measures = self.evaluate(y[lab], self.custom_predict(x)[i], prob, lab, w[lab])
                    for lm in lab_measures:
                        measures.append(str(lm))
            outfile.write(','.join([basename(self.outdir), total_epochs, best_epoch,
                                    str(int(round(elapsed, 0))), self.convert(elapsed)] + measures) + '\n')

    def evaluate(self, y_true, y_pred, problem, label, weights):
        """
        :param y_true: the actual labels
        :param y_pred: the predicted labels
        :param problem: the problem (regression/classification)
        :param label: the target label
        :param weights: the sample weights associated with each labeled instance

        Calculates evaluation metrics.
        Classification metrics:
            - AUC, Accuracy, and F1-Score
        Regression metrics:
            - MAE, RMSE, Max AE, and R2-Score

        :return: regression/classification measures
        """
        y_pred = y_pred.copy()[weights != 0]
        y_true = y_true.copy()[weights != 0]
        weights = weights.copy()[weights != 0]
        if problem == 'classification':
            if len(y_true) == 0:
                return [0, 0, 0]
            best_t = 0.5
            auc = roc_auc_score(y_true, y_pred, sample_weight=weights)
            acc = accuracy_score(y_true, (y_pred >= best_t), sample_weight=weights)
            f1 = f1_score(y_true, (y_pred >= best_t), labels=[0, 1], pos_label=1, sample_weight=weights)
            return [auc, acc, f1]  # [auc, acc, recall, precision, f1]
        elif problem == 'regression':
            if len(y_true) == 0:
                return [0, 0, 0, 0]
            mae = self.data.inverse_transform(mean_absolute_error(y_true, y_pred, sample_weight=weights), label)
            rmse = self.data.inverse_transform(np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=weights)),
                                               label)
            max_ae = self.data.inverse_transform(max_error(y_true, y_pred), label)
            r2 = r2_score(y_true, y_pred, sample_weight=weights)
            return [mae, rmse, max_ae, r2]


class MLBase(Base):
    """
    Base class of the machine learning models. Also inherits from the Base class.
    """
    def custom_predict(self, x):
        """
        Instead of using the sklearn predict function use custom_predict(), this reshapes it in the correct way.
        It also reshapes the input data in the correct way
        The reshape is basically that the predictions for each label are put in a list, e.g.:
        [y_pred of label1, y_pred of label2, ...]
        Depends on how many different labels the model should predict
        """
        shape = (x.shape[0], -1, 1)

        pred = self.model.predict(x.reshape((x.shape[0], self.data.timesteps * len(self.data.cols))))
        if isinstance(pred, list):
            return [p.reshape(shape) for p in pred]
        else:
            return [pred.reshape(shape)]

    def train(self):
        """ Train the model """
        assert not self.load_from_dir, 're-training is not supported!'

        start = time()
        try:
            # Get x, y, and sample weight train data and fit the sklearn model
            y = self.data.y_train[self.data.labels[0]].reshape((-1, 1))
            x = self.data.x_train.reshape((-1, self.data.timesteps * len(self.data.cols)))
            self.model.fit(x, y, sample_weight=self.data.train_weights[self.data.labels[0]].reshape(-1))
        except KeyboardInterrupt:
            try:
                shutil.rmtree(self.outdir)
            except OSError as e:
                print("Error: %s : %s" % (self.outdir, e.strerror))
            exit(0)

        # Training time
        elapsed = time() - start
        print(self.convert(elapsed))
        self.log(elapsed, '-1', '-1')

        # Save model
        pickle.dump(self.model, open(join(self.outdir, 'model', 'trained_model.pickle'), 'wb'))

        label = self.data.labels[0]
        print(label, self.evaluate(self.data.y_test[label], self.custom_predict(self.data.x_test)[0],
                                   self.data.problem[0], self.data.labels[0], self.data.test_weights[label]))
