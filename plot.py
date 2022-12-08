# Standard
from os.path import isdir, join
from os import mkdir, listdir

# Seeds
from os import environ
from random import sample, seed as rseed

# Array manipulations
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Metrics
from seaborn import heatmap
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, accuracy_score, confusion_matrix, auc

# Define seeds
seed_value = 0

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
environ['PYTHONHASHSEED'] = str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
rseed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

cm = 1/2.54

class Plotter:
    def __init__(self, data, model, outdir, save_mode=True, show_mode=False):
        """
        :param data: Input data, should be from PreProcessor class
        :param model: Model, should be from one of the classes in ./models
        :param outdir: Where to output the figures
        :param save_mode: Whether to save the plots
        :param show_mode: Whether to show the plots
        """
        # Initialize values
        self.data = data
        self.model = model
        self.outdir = outdir  # join(outdir, 'figures')
        self.save_mode = save_mode
        self.show_mode = show_mode

        # Create outdir
        if not isdir(self.outdir):
            mkdir(self.outdir)

        # Calculate plotting variables if model is not None
        if self.model is not None:
            self.plot_vars = self.get_plotting_variables()
        else:
            self.plot_vars = None

    def correct_label_for_plot(self, label, problem):
        """ Checks if label and problem are correct for plotting """
        assert problem in ['regression', 'classification'], 'Unknown problem found!'
        if problem == 'regression':
            return label in self.data.cols[self.data.numeric_cols], '{} is not a regression label'.format(label)
        elif problem == 'classification':
            return label not in self.data.cols[self.data.numeric_cols], '{} is not a regression label'.format(label)

    def get_plotting_variables(self):
        """
        Get all the variables needed to make the plots.
        Here all the variables are reshaped in a way that it is the same for all models.
        :return: Nested dictionary of the sample masks, sample weights, input values,
                 predictions and true labels per subset and label.
                 {'train': {'data': x_train,
                            'original_data': original_x_train,
                            'label1': (mask, weights, y_pred, y_true),
                            'label2': (mask, weights, y_pred, y_true)
                           },
                  'test': {'data': x_test,
                            'original_data': original_x_test,
                           'label1': (mask, weights, x, y_pred, y_true),
                           'label2': (mask, weights, x, y_pred, y_true)
                          }
                 }
        """
        plot_vars = {}
        if self.data.predict_ml:
            reshape = lambda y: y.reshape((y.shape[0] // self.data.timesteps, self.data.timesteps, 1))
            select = lambda x: x[(self.data.timesteps - 1)::self.data.timesteps]
        else:
            reshape = lambda y: y
            select = lambda x: x
        for i, lab in enumerate(self.data.labels):
            predictions = [self.model.custom_predict(x=self.data.x_train),
                           self.model.custom_predict(x=self.data.x_test)]
            for d, pred in zip(['train', 'test'], predictions):
                if d == 'train':
                    plot_vars[d] = {'data': select(self.data.x_train.copy()),
                                    'original_data': self.data.original_x_train.copy(),
                                    lab: (reshape(self.data.train_masked_weights[lab].copy()),
                                          reshape(self.data.train_weights[lab].copy()),
                                          reshape(pred[i]), reshape(self.data.y_train[lab].copy())
                                          ),
                                    # 'original_' + lab: reshape(self.data.original_y_train[lab].copy())
                                    }
                    mask = plot_vars[d][lab][0]
                    plot_vars[d][lab][-1][~mask] = self.data.mask
                elif d == 'test':
                    plot_vars[d] = {'data': select(self.data.x_test.copy()),
                                    'original_data': self.data.original_x_test.copy(),
                                    lab: (reshape(self.data.test_masked_weights[lab].copy()),
                                          reshape(self.data.test_weights[lab].copy()),
                                          reshape(pred[i]), reshape(self.data.y_test[lab].copy())
                                          ),
                                    # 'original_' + lab: reshape(self.data.original_y_test[lab].copy())
                                    }
                    mask = plot_vars[d][lab][0]
                    plot_vars[d][lab][-1][~mask] = self.data.mask
        return plot_vars

    def set_model(self, model):
        """
        :param model: new model

        Sets the new model in plotter to given model and recalculates the predicted labels and plotting variables.
        """
        self.model = model
        if self.model is None:
            self.plot_vars = None
        else:
            self.plot_vars = self.get_plotting_variables()
        return

    def ae_distribution(self, subset, label):
        """
        Create the AE distribution of the subset (train or test) for the given label.
        """
        assert label in self.data.labels, '{} is not a label predicted by the model'.format(label)
        assert subset in ['train', 'test'], 'subset must equal {} or {}, not: {}'.format('train', 'test', subset)
        assert self.correct_label_for_plot(label, 'regression')
        mask, weights, y_pred, y_true = self.plot_vars[subset][label]
        ae = self.data.inverse_transform(np.abs(y_true - y_pred), label, metric=True)
        values = []
        for t in range(self.data.timesteps):
            for v in ae[:, t, :][mask[:, t, :]]:
                values.append([t, v])
        df = pd.DataFrame(values, columns=['t', 'ae'])

        fig, ax = plt.subplots(1, 1, figsize=(5.972, 8.67 * .4))

        sns.boxplot(y='t', x='ae', data=df, ax=ax, width=.2, fliersize=3, linewidth=0, orient='h')
        sns.violinplot(y='t', x='ae', data=df, dodge=False, inner='box', ax=ax, orient='h')
        if self.data.shift == 1:
            plt.ylabel('Years Ahead')
        else:
            plt.ylabel('Time Step (t)')
        plt.xlabel('Absolute Error (AE)')
        plt.yticks(range(self.data.timesteps),
                   ['{} (n={})'.format(t + self.data.shift, (df['t'] == t).sum()) for t in range(self.data.timesteps)])
        max_ae = df['ae'].max().round(0).astype(int) + 5
        plt.xticks(np.arange(0, max_ae, 5), np.arange(0, max_ae, 5), rotation=270)

        plt.tight_layout()
        if self.save_mode:
            plt.savefig(join(self.outdir, 'figures', '{}_{}_ae_distribution.png'.format(subset, label)))
            plt.savefig(join(self.outdir, 'figures', '{}_{}_ae_distribution.pdf'.format(subset, label)))
        if self.show_mode:
            plt.show()
        plt.close()

    def measures_per_status(self, subset, label):
        """
        Create evaluation plots of the different groups in the data.
        Here the groups are defined as unknown, progressive, stable, and recovered.
        Unknown is not shown.
        """
        assert label in self.data.labels, '{} is not a label predicted by the model'.format(label)
        assert subset in ['train', 'test'], 'subset must equal {} or {}, not: {}'.format('train', 'test', subset)

        cols = []
        problem = self.data.problem[self.data.labels == label]
        if problem == 'classification':
            cols = ['AUC', 'Accuracy', 'F1']
        elif problem == 'regression':
            cols = ['MAE', 'RMSE', 'Max AE', 'R2']

        # Get original data to get the first visit
        x = self.plot_vars[subset]['original_data'].copy()
        if self.data.predict_ml:
            x = x[self.data.timesteps - 1::self.data.timesteps]
        # Get first visit
        first_visit = x[:, 0, self.data.cols == label].reshape(-1)

        # Get plotting variables
        mask, weights, y_pred, y_true = self.plot_vars[subset][label]
        # Get the label at the last visit
        last_visit_idx = np.argmax(mask * np.arange(1, mask.shape[1] + 1).reshape((1, mask.shape[1], 1)), axis=1)
        last_visit = np.array([y[i] for i, y in zip(last_visit_idx, y_true)]).reshape(-1)

        # Define the unknown and stable groups
        unknown = np.logical_or((first_visit == self.data.mask), (last_visit == self.data.mask))
        stable = (last_visit == first_visit) * ~unknown

        # For motscore the recovered and progressive different for other measures (big motscore bad)
        if label == 'motscore':
            recovered = (last_visit < first_visit) * ~unknown
            progressive = (last_visit > first_visit) * ~unknown
        else:
            recovered = (last_visit > first_visit) * ~unknown
            progressive = (last_visit < first_visit) * ~unknown

        flat_rows, rows = [], []
        # Evaluation of the different groups
        for name, status in zip(['Stable', 'Recovered', 'Progressive'], [stable, recovered, progressive]):
            n = np.sum(status)
            if n != 0:
                measures = self.model.evaluate(y_true[status], y_pred[status], problem, label, weights[status])
                rows.append([name, n] + measures)
                for m, c in zip(measures, cols):
                    # flat_rows.append(['{} (n={})'.format(name, n), c, m])
                    flat_rows.append(['{}\n(n={})'.format(name, n), c, m])
        # Plotting
        df = pd.DataFrame(rows, columns=['status', 'n'] + cols)
        flat_df = pd.DataFrame(flat_rows, columns=['status', 'measure', 'value'])
        sns.catplot(data=flat_df, x='status', y='value', hue='measure', col='measure', kind='bar', sharey=False,
                    col_wrap=2)
        plt.tight_layout()
        # Save/Show
        if self.save_mode:
            df.to_csv(join(self.outdir, 'tables', '{}_{}_measures_per_status.csv'.format(subset, label)), index=False,
                      float_format='%.3f')
            plt.savefig(join(self.outdir, 'figures', '{}_{}_measures_per_status.png'.format(subset, label)))
        if self.show_mode:
            plt.show()
        plt.close()
        return flat_df

    def forecasting_measures(self, subset, label):
        """
        Calculate the measures at each timestep in subset (train or test) of the given label
        :return: dataframe of evaluation metrics, used in plot_tuning.ipynb
        """
        assert label in self.data.labels, '{} is not a label predicted by the model'.format(label)
        assert subset in ['train', 'test'], 'subset must equal {} or {}, not: {}'.format('train', 'test', subset)
        problem = self.data.problem[self.data.labels == label]
        cols = []
        if problem == 'classification':
            cols = ['AUC', 'Accuracy', 'F1']
        elif problem == 'regression':
            cols = ['MAE', 'RMSE', 'Max AE', 'R2']

        # Get plot vars
        mask, weights, y_pred, y_true = self.plot_vars[subset][label]
        values = []
        s = 0
        # Calculate measures at each time step
        for t in range(weights.shape[1]):
            if weights[:, t, s].sum() != 0:
                measures = self.model.evaluate(y_true[:, t, s], y_pred[:, t, s], problem, label, weights[:, t, s])
                # n = np.sum(~np.isnan(y_true[:, t, s].reshape(-1)))
                n = np.sum(mask[:, t, s].reshape(-1))
                for c, m in zip(cols, measures):
                    values.append([n, t, '{:d} (n={:d})'.format(t + 1, n), c, m])
        # plot
        df = pd.DataFrame(values, columns=['n', 'time step (t)', 'xlabel', 'measure', 'value'])
        # Plot
        g = sns.catplot(x="xlabel", y="value", hue="time step (t)", col='measure', data=df, kind="bar",
                        col_wrap=3 if problem == 'classification' else 2, sharey=False)
        g.tight_layout()
        if self.save_mode:
            g.savefig(join(self.outdir, 'figures', '{}_{}_forecast_measures.png'.format(subset, label)))
            # Save as csv
            flat_df = df.pivot(index=['n', 'time step (t)'], columns=['measure'], values=['value'])
            flat_df.columns = flat_df.columns.get_level_values(1)
            flat_df = flat_df.reset_index()
            flat_df.to_csv(join(self.outdir, 'tables', '{}_{}_forecast_measures.csv'.format(subset, label)),
                           index=False)
        if self.show_mode:
            plt.show()
        plt.close()
        return df
    
    def consecutive_measures(self, subset, label):
        """
        Calculate the measures at each timestep in subset (train or test) of the given label
        :return: dataframe of evaluation metrics, used in plot_tuning.ipynb
        """
        assert label in self.data.labels, '{} is not a label predicted by the model'.format(label)
        assert subset in ['train', 'test'], 'subset must equal {} or {}, not: {}'.format('train', 'test', subset)
        problem = self.data.problem[self.data.labels == label]
        cols = []
        if problem == 'classification':
            cols = ['AUC', 'Accuracy', 'F1']
        elif problem == 'regression':
            cols = ['MAE', 'RMSE', 'Max AE', 'R2']

        # Get plot vars
        mask, weights, y_pred, y_true = self.plot_vars[subset][label]
        values = []
        # Calculate measures at each time step
        for ct in range(weights.shape[2]):
            if weights[:, :, ct].sum() != 0:
                measures = self.model.evaluate(y_true[:, :, ct], y_pred[:, :, ct], problem, label, weights[:, :, ct])
                # n = np.sum(~np.isnan(y_true[:, t, s].reshape(-1)))
                n = np.sum(mask[:, :, ct].reshape(-1))
                for c, m in zip(cols, measures):
                    values.append([n, ct, '{:d} (n={:d})'.format(ct + 1, n), c, m])
        # plot
        df = pd.DataFrame(values, columns=['n', 'consecutive time step (ct)', 'xlabel', 'measure', 'value'])
        # Plot
        g = sns.catplot(x="xlabel", y="value", hue="consecutive time step (ct)", col='measure', data=df, kind="bar",
                        col_wrap=3 if problem == 'classification' else 2, sharey=False)
        g.tight_layout()
        if self.save_mode:
            g.savefig(join(self.outdir, 'figures', '{}_{}_consecutive_measures.png'.format(subset, label)))
            # Save as csv
            flat_df = df.pivot(index=['n', 'consecutive time step (ct)'], columns=['measure'], values=['value'])
            flat_df.columns = flat_df.columns.get_level_values(1)
            flat_df = flat_df.reset_index()
            flat_df.to_csv(join(self.outdir, 'tables', '{}_{}_consecutive_measures.csv'.format(subset, label)),
                           index=False)
        if self.show_mode:
            plt.show()
        plt.close()
        return df

    def select_from_sample(self, select, complete, y):
        """
        :param select: what kind of sample to pick: any, a sample with change, or a sample with no change (constant)
        :param complete: if all visits should have a label
        :param y: the labels

        Select a patient that fits the inclusion criteria of select and complete

        :return the index of the random choice of the selection criteria
        """
        assert select in ['any', 'change', 'constant'], "selection type is not in ['any', 'change', 'constant']"
        y = y[:, :, 0].copy()
        # Take any sample or a sample without missing values
        samples = np.array(y != self.data.mask).all(axis=1).astype(int)
        if not complete or np.sum(samples) == 0:
            samples = np.ones(y.shape[0])
        # Filter samples
        if select != 'any':
            # n.o. unique labels per row/observation
            unique_values = np.array([len(set(tuple(row))) for row in y])
            if select == 'change':
                samples *= (unique_values > 1)
            elif select == 'constant':
                samples *= (unique_values == 1)
            else:
                raise
        probabilities = samples / np.sum(samples)
        return np.random.choice(y.shape[0], 1, replace=False, p=probabilities)[0]

    def accuracy_overlay(self, label, subset, fig, ax):
        """ Calculate the accuracy heatmap for the driving plot"""
        mask, weights, y_pred, y_true = self.plot_vars[subset][label]

        optimal_threshold = 0.5
        y_class = y_pred >= optimal_threshold

        # Calculate accuracy for each band
        prob_min = np.arange(0.0, 1.0, 0.1).round(1)
        prob_max = np.arange(0.1, 1.1, 0.1).round(1)
        accuracies = np.empty((prob_max.shape[0]+2, 1))
        for i, p_min, p_max in zip(range(accuracies.shape[0]), prob_min, prob_max):
            p_mask = (y_pred > p_min) * (y_pred <= p_max) * mask
            acc = accuracy_score(y_class[p_mask], y_true[p_mask], sample_weight=weights[p_mask])
            accuracies[i+1] = acc
        accuracies[0] = accuracies[1]
        accuracies[-1] = accuracies[-2]

        # Plot
        hm = ax.imshow(accuracies, alpha=0.6, extent=[-0.03, self.data.timesteps + self.data.cons_t - 2 + .03, -0.1, 1.1],
                       cmap=sns.color_palette("Greens", as_cmap=True),
                       origin='lower', aspect='auto')
        fig.colorbar(hm, ax=ax, label='Accuracy')
        ax.set_yticks(np.arange(0.0, 1.1, 0.1).round(1))
        ax.set_yticklabels(np.arange(0.0, 1.1, 0.1).round(1))
        xticks = range(self.data.timesteps + self.data.cons_t - 1)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)
        ax.set_xlim(-.03, self.data.timesteps + self.data.cons_t - 2 + .03)
        ax.set_ylim(-.03, 1.03)

    def forecasting_plot(self, subset, label, select='any', complete=True):
        """
        :param subset: train or test
        :param label: name of label
        :param select: what kind of sample to pick: any, a sample with change, or a sample with no change (constant)
        :param complete: if all visits should have a label

        Create the forecasting plot for the driving capability

        :return:
        """
        # Assertions
        assert label in self.data.labels, '{} is not a label predicted by the model'.format(label)
        assert subset in ['train', 'test'], 'subset must equal {} or {}, not: {}'.format('train', 'test', subset)
        # Init variables
        label_idx = np.where(self.data.labels == label)[0][0]
        problem = self.data.problem[label_idx]

        # Get labels, need all labels so the original dataset
        if subset == 'train':
            true_labels = self.data.original_y_train[label].copy()[self.data.timesteps - 1::self.data.timesteps] # First value
            x = self.data.original_x_train.copy()[self.data.timesteps - 1::self.data.timesteps]
        else:
            true_labels = self.data.original_y_test[label].copy()[self.data.timesteps - 1::self.data.timesteps] # First value
            x = self.data.original_x_test.copy()[self.data.timesteps - 1::self.data.timesteps]

        # Get sample idx, which fits inclusion criteria
        sample_idx = self.select_from_sample(select, complete, true_labels)
        x = x[sample_idx]

        # Get the true and predicted label
        y_pred_idx = self.model.custom_predict(x.reshape((1, -1, len(self.data.cols))))[label_idx][0]
        y_true_idx = x[:, self.data.cols == label].reshape(-1).copy()
        y_true_idx[y_true_idx == self.data.mask] = np.nan
        y_true_idx = self.data.inverse_transform(y_true_idx, label, metric=False).reshape(-1)
        y_pred_idx = self.data.inverse_transform(y_pred_idx, label, metric=False)

        # Info to list and later pandas dataframe for plotting
        cols = ['time step (t)', 'type', label]
        values = []
        for t in range(self.data.timesteps + self.data.shift):
            values.append([str(t), 'Patient\'s\nDriving Status', y_true_idx[t]])
            values.append([str(t + self.data.shift), 'Model\nAssessment', y_pred_idx[t, 0]])
            # if self.data.cons_t == 2:
            #     values.append([str(t + self.data.shift + 1), 'Model Future\nPrediction', y_pred_idx[t, 1]])
        
        if self.data.cons_t == 2:
            values.append([str(self.data.timesteps - 1 + self.data.shift), 'Model Future\nPrediction', y_pred_idx[-1, 0]])
            values.append([str(self.data.timesteps + self.data.shift), 'Model Future\nPrediction', y_pred_idx[-1, 1]])
        # Plot
        fig, ax = plt.subplots(1, figsize=(17*cm, 9*cm))
        if problem == 'classification':
            # Threshold
            optimal_threshold = 0.5
            ax.axhline(y=optimal_threshold, color="black", dashes=(2, 1),
                       label='Advisory\nThreshold ({:.1f})'.format(optimal_threshold))
            # Heatmap
            self.accuracy_overlay(label, subset, fig, ax)

        df = pd.DataFrame(values, columns=cols)
        truth = df.loc[df['type'] == 'Patient\'s\nDriving Status']
        ax.plot(truth['time step (t)'], truth[label], label='Patient\'s\nDriving Status', marker='o')
                 
        pred = df.loc[df['type'] == 'Model\nAssessment']
        ax.plot(pred['time step (t)'], pred[label], label='Model\nAssessment', marker='o')
        if self.data.cons_t == 2:
            pred = df.loc[df['type'] == 'Model Future\nPrediction']
            ax.plot(pred['time step (t)'], pred[label], label='Model Future\nPrediction', color='C1', linestyle='--')

        # Legend
        ax.set_xlabel('Visits / Time Points (t)')
        ax.set_ylabel(label)
        xticklabels = ax.get_xticks().astype(str)
        xticklabels[-1] += '*'
        ax.set_xticklabels(xticklabels)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4, fontsize=8)
        plt.tight_layout()
        # Save plot
        if self.save_mode:
            name = '{}_{}_forecasting_{}'.format(subset, label, select)
            num = np.sum(['.'.join(file.split(' ')[:-1]) == name for file in listdir(join(self.outdir, 'figures'))])
            plt.savefig(join(self.outdir, 'figures', '{} ({}).pdf'.format(name, num)), format='pdf', dpi=300, bbox_inches="tight", pad_inches=0)
        if self.show_mode:
            plt.show()
        plt.close()

    def roc(self, label, subset):
        """
        :param subsets: List of subsets to create roc plot for
        :param label: What target to calculate the ROC curve on

        Calculates and plots the ROC curve, Area Under the Curve (AUC), and the
        optimal threshold for binary classification.
        """
        assert label in self.data.labels, '{} is not a label predicted by the model'.format(label)
        assert self.correct_label_for_plot(label, 'classification')

        # Get plotting variables
        mask, weights, y_pred, y_true = self.plot_vars[subset][label]

        # Get fpr and tpr at different thresholds
        fpr, tpr, thresholds = roc_curve(y_true[mask], y_pred[mask], pos_label=1, sample_weight=weights[mask])

        # Get idx where TPR=0.99 and TNR=0.99
        normal_idx = np.argmax(thresholds.round(2) == 0.5)
        negative_idx = len(fpr) - np.argmax(fpr[::-1].round(2) == 0.01) - 1
        positive_idx = np.argmax(tpr.round(2) == 0.99)

        # Calculate AUC
        area = auc(fpr, tpr)

        plt.figure(figsize=(16*cm, 9*cm))
        # Plot points of intersting thresholds
        plt.scatter(fpr[negative_idx], tpr[negative_idx], c='C0', s=20)
        plt.annotate('(T={:.2f}, FPR={:.2f}, TPR={:.2f})'.format(thresholds[negative_idx], fpr[negative_idx], tpr[negative_idx]),
                     (fpr[negative_idx], tpr[negative_idx]), (fpr[negative_idx] + 0.01, tpr[negative_idx]), fontsize=8)

        plt.scatter(fpr[positive_idx], tpr[positive_idx], c='C0', s=20)
        plt.annotate('(T={:.2f}, FPR={:.2f}, TPR={:.2f})'.format(thresholds[positive_idx], fpr[positive_idx], tpr[positive_idx]),
                     (fpr[positive_idx], tpr[positive_idx]), (fpr[positive_idx] + 0.01, tpr[positive_idx] - 0.05), fontsize=8)

        plt.scatter(fpr[normal_idx], tpr[normal_idx], c='C0', s=20)
        plt.annotate('(T={:.2f}, FPR={:.2f}, TPR={:.2f})'.format(thresholds[normal_idx], fpr[normal_idx], tpr[normal_idx]),
                     (fpr[normal_idx], tpr[normal_idx]), (fpr[normal_idx] + 0.01, tpr[normal_idx] - 0.05), fontsize=8)

        plt.plot(fpr, tpr, c='C0', label='ROC curve on {} set (AUC: {:.3f})'.format(subset, area))

        plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), linestyle='--', c='C2',
                 label='Random model')
        plt.legend(fontsize=8)
        plt.xlabel('False Positive Rate (Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.tight_layout()
        if self.save_mode:
            plt.savefig(join(self.outdir, 'figures', label + '_roc_curve.pdf'), dpi=300, bbox_inches="tight", pad_inches=0)
        if self.show_mode:
            plt.show()
        plt.close()

    def confusion_matrix(self, subset, label):
        """
        :param subset: whether to use the train or test data
        :param label: the target label

        Calculates the confusion matrix for target label (t).
        The classification threshold is defined using the roc_curve() function from sklearn
        """
        assert label in self.data.labels, '{} is not a label predicted by the model'.format(label)
        assert subset in ['train', 'test'], 'subset must equal {} or {}, not: {}'.format('train', 'test', subset)
        assert self.correct_label_for_plot(label, 'classification')
        mask, weights, y_pred, y_true = self.plot_vars[subset][label]
        confusion_labels = ['TP', 'FN', 'FP', 'TN']

        y_pred = y_pred[mask]
        y_true = y_true[mask]

        optimal_threshold = 0.5

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred >= optimal_threshold, labels=[0, 1]).ravel()
        matrix = np.array([tp, fn, fp, tn]).reshape(2, 2)
        perc_matrix = matrix / np.sum(matrix, axis=1).reshape(-1, 1) * 100
        labels = ['{}\n{:.2f}%\n({:d})'.format(lab, p, m) for lab, p, m in zip(confusion_labels, perc_matrix.reshape(-1),
                                                                              matrix.reshape(-1))]
        fig, ax = plt.subplots(1, figsize=(8.5*cm, 8.5*cm))
        heatmap(pd.DataFrame(perc_matrix, index=['Positive', 'Negative'], columns=['Positive', 'Negative']),
                fmt='', vmin=0, vmax=100, annot=np.array(labels).reshape((2, 2)),
                square=True, ax=ax, cbar_kws={"shrink": .7},
                cmap=sns.color_palette("crest", as_cmap=True)
                )
        plt.ylabel('True Value')
        plt.xlabel('Predicted Value')
        plt.tight_layout()
        if self.save_mode:
            plt.savefig(join(self.outdir, 'figures', '{}_confusion_matrix.pdf'.format(label)),
                        bbox_inches='tight', pad_inches=0, dpi=300)
        if self.show_mode:
            plt.show()
        plt.close()
