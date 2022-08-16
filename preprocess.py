from os.path import join
from pandas.api.types import is_numeric_dtype

# Seeds
from os import environ
from random import seed as rseed

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns


class PreProcessor:
    def __init__(self, infile, mask_file, outdir, mask_value, timestep_size=365, max_timestep=5, filt_outliers=False,
                 labels=None, problem=None, mask_label=False, predict_ml=False, shift=1, scaler_type='norm'):
        """
        :param infile: What data to pre-process, generally data/imputed/PDS5_IRP.csv
        :param mask_file: What data to use as a masking file for the labels, generally data/pre-processed/PDS5_IRP.csv
                          This file is used to mask labels which are imputed, since we don't want to train on imputed
                          data.
        :param outdir: Where to save the figures, generally results/figures
        :param mask_value: What value to use as a mask default=-2, should be outside value range, around -1 to +1.
        :param timestep_size: How many days should be between each visits (default=365 days)
        :param max_timestep: Maximum number of visits allowed (default=5)
        :param filt_outliers: Whether to filter outliers. Outliers are defined in filter_outliers(). (default=False)
        :param labels: What label(s) to predict, e.g. cUHDRS, motscore, etc. (list)
        :param problem: Whether to handle the prediction problem as a classification or regression problem. (list)
        :param mask_label: Whether to mask the label(s) in the input data. This is done for drive.
        :param predict_ml: Whether to predict as a RNN model or not. When True the samples are duplicated to allow
                           longitudinal predictions.
        :param shift: Whether to predict the current (0) or next visit (1).
        :param scaler_type: standardization (stand) or normalization (norm). (default=norm)
        """
        # Assertions
        assert len(labels) == len(problem), 'number of labels and problems are not the same length!'
        assert scaler_type in ['norm', 'normalization', 'stand', 'standardization'], 'scaler type unknown'
        assert shift in [0, 1], 'shift needs to be either 0 or 1 not {}'.format(shift)

        # Define seeds
        seed_value = 0
        # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
        environ['PYTHONHASHSEED'] = str(seed_value)
        # 2. Set `python` built-in pseudo-random generator at a fixed value
        rseed(seed_value)
        # 3. Set `numpy` pseudo-random generator at a fixed value
        np.random.seed(seed_value)
        print('Preprocess Random Seed set to', seed_value)

        # Initialize values
        self.outdir = outdir
        self.mask = mask_value
        self.shift = shift
        self.mask_label = mask_label
        self.labels = np.array(labels)
        self.problem = np.array(problem)
        self.predict_ml = predict_ml

        # Use appropriate scales
        self.scaler_type = scaler_type
        if self.scaler_type == 'norm' or self.scaler_type == 'normalization':
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
        elif self.scaler_type == 'stand' or self.scaler_type == 'standardization':
            self.scaler = StandardScaler()

        # Load and reshape both input and mask data
        self.X, self.cols, self.numeric_cols, self.constant_cols = self.proces_csv(infile, timestep_size, max_timestep,
                                                                                   filt_outliers)
        X_mask, _, _, _ = self.proces_csv(mask_file, timestep_size, max_timestep, filt_outliers)

        # Calculate number of time steps
        self.timesteps = self.X.shape[1] - self.shift

        print(self.timesteps)
        print('Data:', self.X.shape)
        print('Targets:', ', '.join(['{} ({})'.format(t, p[0]) for t, p in zip(self.labels, self.problem)]))

        # Split, Scale and Label data
        train, test, train_mask, test_mask = self.split(self.X, X_mask)

        # Mask samples
        print(train.shape, train_mask.shape)
        print(test.shape, test_mask.shape)
        if predict_ml:
            train = self.mask_samples(train, fill=True)
            test = self.mask_samples(test, fill=True)

            train_mask = self.mask_samples(train_mask, fill=False)
            test_mask = self.mask_samples(test_mask, fill=False)

        # Scale data
        train, test = self.scale(train, test)
        train_mask, test_mask = self.scale(train_mask, test_mask)
        print('=====================================================')
        print(train.shape, train_mask.shape)
        print(test.shape, test_mask.shape)
        print('=====================================================')

        # Label data
        self.x_train, self.y_train, y_train_mask, self.original_x_train, self.original_y_train = self.label_data(train, train_mask)
        self.x_test, self.y_test, y_test_mask, self.original_x_test, self.original_y_test = self.label_data(test, test_mask)

        # Mask labels in input data if needed, see mask_label
        if self.mask_label:
            for label in self.labels:
                self.x_train[:, :, self.cols == label] = self.mask
                self.x_test[:, :, self.cols == label] = self.mask

        # Get mask for train and test labels, i.e. weight = 0 is label is missing (mask) else weight = 1
        self.train_masked_weights = self.calc_masked_weights(y_train_mask)
        self.test_masked_weights = self.calc_masked_weights(y_test_mask)

        # Add class weight if problem is a classification problem
        self.train_weights = self.add_class_weights(y_train_mask, self.train_masked_weights)
        self.test_weights = self.add_class_weights(y_test_mask, self.test_masked_weights)

        # Info / Plots
        self.train_test_stats()
        self.plot_visit_dist()

        # Print Scaling info
        for t in self.labels:
            if t in self.cols[self.numeric_cols]:
                if self.scaler_type in ['norm', 'normalization']:
                    print('{}: x - {} / {}'.format(t, self.scaler.min_[self.cols[self.numeric_cols] == t][0],
                                                   self.scaler.scale_[self.cols[self.numeric_cols] == t][0]))
                elif self.scaler_type in ['stand', 'standardization']:
                    print('{}: x - {} / {}'.format(t, self.scaler.mean_[self.cols[self.numeric_cols] == t][0],
                                                   self.scaler.scale_[self.cols[self.numeric_cols] == t][0]))

    @staticmethod
    def filter_outliers(df_in, x):
        """ Filter outliers or not, generally not done. Here outliers are defined as values oustide u+o or u-o"""
        # removes values x times the std
        for c in df_in.columns:
            if is_numeric_dtype(df_in[c]) and c not in ['visdy', 'dpd', 'dpdy']:
                # Set to nan
                df_in.loc[~((df_in[c] - df_in[c].mean()).abs() <= (x * df_in[c].std())), c] = np.nan
        return df_in

    @staticmethod
    def check(df_in, cols):
        """ Check if columns are in the dataframe or if they are dummy columns"""
        new_cols = []
        for c in cols:
            new = list(df_in.columns[df_in.columns.str.startswith(pat=c + '_')].values)
            if len(new) == 0:
                new_cols.append(c)
            else:
                new_cols += new
        return new_cols

    @staticmethod
    def reshape_data(df_in, const_in, timestep_size=(365 / 12), max_t=np.inf):
        """
        df_in: Input dataframe
        const_in: Constant columns (unused)
        timestep_size: How many days between each time step
        max_t: Maximum number of timesteps allowed in data

        Reshapes the input data to fit the temporal distance between the time steps (timestep_size = in years).
        It also cut-offs visits which go beyond the maximum time steps allowed (max_t).
        First the code calculates what the maximum number of visits it should expect (t).
        Next it makes an numpy array filled with NaN of shape (#patients * t, #features).
        Then it calculates the appropriate index (t) for each patient using the visdy variable.
        The nan array is then filled in with all the available visit data, all missing visits remain NaN.
        Lastly, the dataset is clipped using the max_t, if needed.
        """
        # Set base to 0
        df_in['subjid_i'] = pd.factorize(df_in['subjid'])[0]
        df_in['timestep'] = np.nan

        # Define shapes
        s = df_in['subjid_i'].nunique()  # Subjects/Rows

        # Maximum Timesteps/Observations found in data( max( (max(day) - min(day)) / timestep_size ) )
        # Used to define how big the reshaped data will get, visit wise
        t = np.ceil(np.max(df_in.groupby('subjid')['visdy'].last().values -
                           df_in.groupby('subjid')['visdy'].first().values) / timestep_size).astype(int)

        # Give each subject unique ID
        df_in = df_in.drop(['subjid'], axis=1)  # , 'studyid', 'seq', 'visstat', 'visit'], axis=1)
        cols = df_in.columns
        f = df_in.shape[1]  # N.o. Features/Columns

        # Make dataset (subject * timesteps, features), all values are first set to nan
        dataset = np.full((s * t, f), np.nan, dtype=float)

        # Calculate appropiate indexes of each visit for each patient (visdy - first visdy)
        s_i = df_in['subjid_i'].values
        # First visit day
        base = df_in.groupby('subjid_i')['visdy'].transform('first').values
        # Index of each visit based on first visit (visdy)
        t_i = np.round((df_in['visdy'] - base) / timestep_size, 0).astype(int)

        # Fill in timestep data, sometimes there is no data so it remains nan
        dataset[s_i * t + t_i, :] = df_in.values

        # Fill timestep column
        dataset[:, np.where(cols == 'timestep')[0][0]] = np.resize(np.arange(0, t), t * s).astype(int)

        # Fill in subject id
        subject_ids = np.concatenate([np.repeat(i, t) for i in range(s)])  # repeat subjid (i), t times for each subject
        dataset[:, np.where(cols == 'subjid_i')[0][0]] = subject_ids

        # Reshape data
        dataset = dataset.reshape((s, t, f))

        # Clip dataset to maximum time step if needed
        if t > max_t:
            print('#time steps exceeds max_t setting #time step to:', max_t)
            t = max_t
            dataset = dataset[:, :t, :]

        print(dataset.shape)
        return dataset, cols

    @staticmethod
    def replace_dummies(df_in):
        """ Use dummy encoding on features with missing values"""
        df_in = df_in.copy()
        df_imp = pd.read_csv('data/imputed_pre_and_manifest.csv')

        if df_imp.isnull().any().any():
            missing_cols = df_in.loc[:, df_imp.isnull().any()].columns
            dummies = pd.get_dummies(df_in.loc[:, missing_cols].astype('category'))
            df_in = df_in.drop(missing_cols, axis=1).join(dummies)
        return df_in

    @staticmethod
    def calc_cogscore(df_in):
        # Features to combine using PCA
        cog_cols = ['sdmt1', 'verfct5', 'scnt1', 'swrt1', 'sit1', 'trla1', 'trlb1', 'verflt05']
        # Which rows will be used to do PCA on, non can be missing
        missing_rows = df_in[cog_cols].isnull().any(1).values
        # New feature array
        new_f = np.full(missing_rows.shape, np.nan)
        # Do Standard Scaling and PCA on non-missing rows and selected columns
        pca = PCA(n_components=len(cog_cols), random_state=0)
        # Save Principal components
        pcs = pca.fit_transform(StandardScaler().fit_transform(df_in.loc[~missing_rows, cog_cols].values))
        # Only take the first Principal component (71% of explained variance)
        new_f[~missing_rows] = pcs[:, 0]
        # Return feature
        return new_f

    def proces_csv(self, csv, size, thresh, outliers):
        """
        :param csv: Input file (imputed or pre-imputed dataset)
        :param size: Preferred number of days between visits
        :param thresh: Max number of visits
        :param outliers: Whether to filter outliers using filter_outliers()
        :return: reshaped data, the columns of the data, mask for numerical columns, mask for categorical columns
        """
        print('Processing:', csv.split('/')[-1],
              'time between visits: {:.3f} days, max visits: {}'.format(size, thresh))
        # Load data
        df = pd.read_csv(csv)
        # Create dummies
        df = self.replace_dummies(df)
        # Recalculate packy, hxpacky, and cUHDRS
        # df['packy'] = (df['tobcpd'] / 20) * df['tobyos']
        # df['hxpacky'] = (df['hxtobcpd'] / 20) * df['hxtobyos']
        df['dbscore'] = df['pbas11sv'] * df['pbas11fr']
        df['manifest'] = (df['hdcat'] == 3).astype(int)
        df['cUHDRS'] = (((df['tfcscore'] - 10.4) / 1.9 ) - ((df['motscore'] - 29.7) / 14.9) + ((df['sdmt1'] - 28.4) / 11.3) +
                        ((df['swrt1'] + 66.1) / 20.1)) + 10
        df['cogscore2'] = self.calc_cogscore(df)
        # Drop unknowns
        # df = df.drop(['hxalcab_9999.0', 'hxtobab_9999.0', 'hxdrugab_9999.0'], axis='columns')

        # Define longitudinal columns
        long_cols = self.check(df, ['age', 'hddiagn', 'parentagesx', 'ccmtrage', 'sxsubj', 'sxfam', 'rtrddur', 'cccogage',
                     # 'ccdepage', 'ccirbage', 'ccvabage', 'ccaptage', 'ccpobage', 'ccpsyage',
                     'bmi', 'alcunits', 'tobcpd', 'tobyos', 'packy', 'cafab', 'cafpd', 'drugab',
                     # var items II
                     'manifest',
                     # 'maristat', 'res', 'jobclas', 'emplnrsn', # ===>MISSING<===
                     # 'isced', 'rdcwk', 'rdcwkd', 'rdcwkhw', # ===>MISSING<===
                     'jobpaid', 'emplnrd', 'ssdb', 'rtrnwk',
                     'capscore',
                     # motscore
                     'motscore', 'diagconf',
                     'ocularh', 'ocularv', 'sacinith', 'sacinitv', 'sacvelh', 'sacvelv', 'dysarth', 'tongue', 'fingtapr',
                     'fingtapl', 'prosupr', 'prosupl', 'luria', 'rigarmr', 'rigarml', 'brady', 'dysttrnk', 'dystrue',
                     'dystlue', 'dystrle', 'dystlle', 'chorface', 'chorbol', 'chortrnk', 'chorrue', 'chorlue', 'chorrle',
                     'chorlle', 'gait', 'tandem', 'retropls',
                     # fascore
                     'fascore', 'indepscl',
                     'drive',
                     'emplusl', 'emplany', 'volunt', 'fafinan', 'grocery', 'cash', 'supchild', 'housewrk',
                     'laundry', 'prepmeal', 'telephon', 'ownmeds', 'feedself', 'dress', 'bathe', 'pubtrans', 'walknbr',
                     'walkfall', 'walkhelp', 'comb', 'trnchair', 'bed', 'toilet', 'carehome',
                     # tfc score
                     'tfcscore',
                     'occupatn', 'finances', 'chores', 'adl', 'carelevl',
                     # Cognitive score
                     # 'cogscore', 'verfct5', 'trla1', 'trlb1',  # Some
                     # 'cogscore1', 'verfct5', 'trla1', 'trlb1',  # Some
                     'cogscore2', 'sdmt1', 'verfct5', 'scnt1', 'swrt1', 'sit1', 'trla1', 'trlb1', 'verflt05',  # All
                     'tug1', 'scst1',  # physiotherapy
                     'mmsetotal',  # Mini mental state
                     'depscore', 'irascore', 'psyscore', 'aptscore', 'exfscore', 'dbscore',  # pba
                     'pf', 'rp', 'bp', 'gh', 'vt', 'sf', 're', 'mh', 'pcs', 'mcs',  # SF-12
                     'anxscore', 'hads_depscore', 'irrscore', 'outscore', 'inwscore',  # HADS-SIS
                     # 'wpaiscr1', 'wpaiscr2', 'wpaiscr3', 'wpaiscr4',  # WPAI‐SHP # ===>MISSING<===
                     'cUHDRS',
                     ])
        # Select constant columns
        const_cols = self.check(df, ['sex', 'handed', # race and handed
                                     'hxsid', 'caghigh', 'caglow', 'momhd', 'dadhd', 'fhx', 'sxraterm',
                                     'ccmtr', 'sxsubjm', 'sxfamm', 'ccdep', 'ccirb', 'ccvab', 'ccapt', 'ccpob', 'ccpsy',
                                     'cccog',
                                     'hxalcab',
                                     'hxtobab', 'hxtobcpd', 'hxtobyos', 'hxpacky',
                                     'hxdrugab',
                                     ]
                                )
        # Drive remove the fascore
        df = df.loc[:, ['subjid', 'visdy'] + const_cols + long_cols]
        if 'drive' in self.labels:
            df = df.drop(['fascore'], axis='columns')
            long_cols.remove('fascore')

        # Filter outliers if needed
        if outliers:
            print('Before:', df.shape, df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
            df = self.filter_outliers(df.copy(), 1)
            print('After:', df.shape, df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)

        # Define numeric columns
        num = df.drop(['subjid', 'visdy'], axis='columns').nunique().values != 2
        # Make a mask of constant columns
        const = np.concatenate([np.full(len(const_cols), True, dtype=bool), np.full(len(long_cols), False, dtype=bool)])

        # Reshape
        dataset, cols = self.reshape_data(df, const_in=const_cols,
                                          timestep_size=size, max_t=thresh)
        # Drop useless variables
        sel = pd.DataFrame(dataset.reshape(-1, len(cols)), columns=cols).drop(['subjid_i', 'visdy', 'timestep'], axis=1)
        # Save reshaped data
        outfile = 'data/reshaped/{}{}_{:.0f}m_{:.0f}y.npz'.format('filt_' if outliers else '',
                                                                  csv.split('/')[-1].split('.')[0],
                                                                  size / (365 / 12),
                                                                  (size * dataset.shape[1]) / 365)
        print('Out:', outfile)
        data = sel.values.reshape((dataset.shape[0], -1, sel.shape[1]))
        np.savez(outfile,
                 data=data, subjects=dataset.shape[0], columns=const_cols + long_cols, constant_cols=const,
                 numeric_cols=num)
        return data, np.array(const_cols + long_cols), num, const.astype(np.bool)

    def calc_last_visit(self, x):
        """ Calculate the last available visit for all patients"""
        if np.isnan(x).any():
            return np.argmax(np.arange(1, x.shape[1] + 1) * ~(np.isnan(x).all(2)), axis=1)
        else:
            return np.argmax(np.arange(1, x.shape[1] + 1) * ~(np.array(x == self.mask).all(2)), axis=1)

    def is_missing(self, v):
        """ Check if a visit is missing """
        assert v.shape == (len(self.cols), ), 'Visit does not match shape ({}, )'.format(len(self.cols))
        if np.isnan(v).any():
            return np.isnan(v).all()
        else:
            return np.array(v == self.mask).all()

    def mask_samples(self, x, fill=False):
        """
        :param x: input for model (train or test)
        :param fill: whether to fill missing visits with the latest non-missing visit values
        Transforms each observation in multiple unique masked copies to allow a prediction for each time step.
        This makes it possible for ML techniques to train on each time step per observation separately like a RNN.
        However, the ML technique sees these as different samples not like the RNN.

        Example (Shift=0):
        xi = [1, 1, 1, 1,   masks = [[T, T, T, T,   [T, T, T, T,   [T, T, T, T,
              2, 2, 2, 2,             F, F, F, F,    T, T, T, T,    T, T, T, T,
              3, 3, 3, 3]             F, F, F, F]    F, F, F, F]    T, T, T, T]]

        Example (Shift=1):
        xi = [1, 1, 1, 1,   masks = [[T, T, T, T,   [T, T, T, T,
              2, 2, 2, 2,             T, T, T, T,    T, T, T, T,
              3, 3, 3, 3]             F, F, F, F]    T, T, T, T]]

        :return: Product of observations and masks displayed above. (Only unique sample per observations are kept)
        """
        x = x.copy()
        total_t = x.shape[1]
        placeholders = np.empty((total_t - self.shift, total_t, len(self.cols)), dtype=np.bool)
        for i, t in enumerate(range(self.shift + 1, total_t + 1)):
            placeholder_i = np.ones((total_t, len(self.cols)), dtype=np.bool)
            placeholder_i[:t] = True
            placeholder_i[t:] = False
            placeholders[i] = placeholder_i

        masked_x = []

        for i in range(x.shape[0]):
            # Forward fill
            if fill:
                last_visit = x[i, 0].copy()
                assert not self.is_missing(last_visit), "First visit is completely masked can't forward fill"
                for t in range(1, x.shape[1]):
                    if self.is_missing(x[i, t]):
                        x[i, t] = last_visit
                    else:
                        last_visit = x[i, t].copy()
            # Make masked samples
            for p in placeholders:
                xi = x[i].copy()
                xi[~p] = np.nan
                masked_x.append(xi)
        # masked_x = [x[i].copy() * placeholders for i in range(x.shape[0])]
        return np.array(masked_x).reshape((-1, total_t, len(self.cols)))

    def split(self, x, x_mask):
        """
        Split the data into a train and test set.
        The data is divided into 80/20 for each maximum number of visits available.
        :return:
        """
        max_visits = self.calc_last_visit(x)
        train, test, train_mask, test_mask = [], [], [], []
        for t in range(x.shape[1]):
            # Check number of subjects for time step t
            mask = np.array(max_visits == t)
            no_samples = np.sum(mask)

            # No (useful) samples can be added
            if t == 0 and self.shift == 1:
                print(no_samples, 'subjects have a maximum of 1 time step! Skipping samples...')
            elif no_samples == 0:
                print('There are {} subjects with a maximum of {} time steps! Skipping time step...'.format(no_samples,
                                                                                                            t))
            else:
                train_idx = np.random.choice([True, False], size=no_samples, p=[0.8, 0.2])
                train.append(x[mask][train_idx])
                train_mask.append(x_mask[mask][train_idx])
                test.append(x[mask][~train_idx])
                test_mask.append(x_mask[mask][~train_idx])

        # Concatenate train and test observations
        return [np.concatenate(arr, axis=0) for arr in [train, test, train_mask, test_mask]]

    def scale(self, train, test):
        """
        :param train: train data
        :param test: test data

        Scales the train and test dataset to the (-1, 1) using the MinMaxScaler.
        :return: Scaled train and test
        """
        # stack x_train and x_test and make 2D
        x = np.vstack([train, test]).reshape(-1, len(self.cols))

        # Normalize / Standardize (fit only on train)
        num = train[:, :, self.numeric_cols].reshape((-1, np.sum(self.numeric_cols)))
        self.scaler.fit(num)

        # Transform x_train and x_test
        x[:, self.numeric_cols] = self.scaler.transform(x[:, self.numeric_cols])

        mask_outside_range = (self.mask < np.nanmin(x.reshape(-1))).all() or \
                             (self.mask > np.nanmax(x.reshape(-1))).all()
        print(self.mask, np.nanmin(x.reshape(-1)), np.nanmax(x.reshape(-1)), mask_outside_range)
        assert mask_outside_range, 'Mask value(' + str(self.mask) + ') is already an existing value in the dataset!'

        x[np.isnan(x)] = self.mask
        x = x.reshape(train.shape[0] + test.shape[0], train.shape[1], len(self.cols))

        # Unstack
        train = x[:train.shape[0]]
        test = x[-test.shape[0]:]
        return train, test

    def label_data(self, x, x_mask):
        """
        :param x: train or test data (imputed data)
        :param x_mask: train or test data (not imputed)

        Labels the input data x and x_mask. Here the labels of x_mask are used to calculate the sample weights.
        The labels originate from the input data, i.e. x and x_mask.
        How the labels are extracted from the input data depends on the model, see mask_samples()

        :return:
        x: Shifted data if needed (not needed for drive).
        y: dictionary of the labels
        y: dictionary of the unimputed labels
        original_x: unshifted data
        original_y: unshifted labels
        """
        # Label (shift labels 0 or 1 time steps)
        y = {}
        y_mask = {}
        original_y = {}
        original_x = x.copy()
        last_visit = self.calc_last_visit(x)

        for lab in self.labels:
            if self.predict_ml:
                # The label is at the last visit of the input, see mask_samples()
                y[lab] = x[np.arange(x.shape[0]), last_visit, self.cols == lab].reshape((x.shape[0], -1, 1)).copy()
                y_mask[lab] = x_mask[np.arange(x.shape[0]), last_visit, self.cols == lab].reshape((x.shape[0], -1, 1)).copy()
                # Simply take all labels from each time steps
                original_y[lab] = x[:, :, self.cols == lab].reshape((x.shape[0], -1, 1)).copy()
            else:
                # The labels are taken from each visit
                y[lab] = x[:, self.shift:, self.cols == lab].reshape((x.shape[0], -1, 1)).copy()
                y_mask[lab] = x_mask[:, self.shift:, self.cols == lab].reshape((x.shape[0], -1, 1)).copy()
                # Simply take all labels from each time steps
                original_y[lab] = x[:, :, self.cols == lab].reshape((x.shape[0], -1, 1)).copy()

        if self.shift == 1:
            if self.predict_ml:
                x[np.arange(x.shape[0]), last_visit] = self.mask
            x = x[:, :-1]  # Train on the first t-shift time steps

        return x, y, y_mask, original_x, original_y

    def calc_masked_weights(self, y):
        """
        :param y: the observation's labels
        :return: weight for each sample and time step (not per label)
        Here weights are simply 1 or 0 depending on the label being equal to the mask (missing) value
        """
        masked_weights = {}
        for i, l in enumerate(self.labels):
            masked_weights[l] = np.array(y[l] != self.mask)
        return masked_weights

    def add_class_weights(self, y, masked_weights):
        """
        Adds class weights to the sample weights if the labels problem is a classification problem.
        """
        # Include class label weights
        y = y.copy()
        masked_weights = {key: masked_weights[key].copy().astype(np.float64) for key in masked_weights.keys()}
        for p, l in zip(self.problem, self.labels):
            yl = y[l]
            if p == 'classification' and l != 'drive':
                # Calculate frequency of each label that is not equal to a mask.
                x, count = np.unique(yl[yl != self.mask].reshape(-1), return_counts=True)
                # Calculate weight based on frequency
                weights = 2 * (1 - count / np.sum(count))
                # Assign class weight to each label
                class_matrix = yl[:, :, self.labels == l].copy()
                for i, xi in enumerate(x):
                    class_matrix[class_matrix == xi] = weights[i]
                masked_weights[l] *= class_matrix
                print('Added class weights for label', l)
            else:
                print('Skipped adding class weights for label', l)
        return masked_weights

    def inverse_transform(self, x, t, metric=True):
        """
        :param x: float of array of floats to transform
        :param t: which label to transform
        :param metric: whether the input variable (x) is a metric, i.e. error or not.

        Transform a value back to the min max range (-1, 1) using the MinMaxScaler
        :return: Inverse Transformed value
        """
        if t in self.cols[self.numeric_cols]:
            if isinstance(x, np.ndarray) or isinstance(x, list):
                x = np.array(x)
                x = x.copy()
            # Get appropriate scaler and idx
            scaler = self.scaler
            idx = np.argmax(self.cols[self.numeric_cols] == t)

            # Use appropriate transformation
            if self.scaler_type in ['norm', 'normalization']:
                if not metric:
                    x -= scaler.min_[idx]
                x /= scaler.scale_[idx]
            elif self.scaler_type in ['stand', 'standardization']:
                x *= scaler.scale_[idx]
                if not metric:
                    x += scaler.mean_[idx]
        return x

    def transform(self, x, t, metric=True):
        """
        :param x: float of array of floats to transform
        :param t: which label to transform
        :param islabel: whether to force the transformer to use the transformer of the label or of the input data
        :param metric: whether the input variable (x) is a metric, i.e. error or not.

        Transform a value back to its original range, using the MinMaxScaler
        :return: Transformed value
        """
        if t in self.cols[self.numeric_cols]:
            if not isinstance(x, np.ndarray):
                x = np.array(x)
            x = x.copy()
            # Get appropriate scaler and idx
            scaler = self.scaler
            idx = np.argmax(self.cols[self.numeric_cols] == t)

            # Use appropriate transformation
            if self.scaler_type in ['norm', 'normalization']:
                x *= scaler.scale_[idx]
                if not metric:
                    x += scaler.min_[idx]
            elif self.scaler_type in ['stand', 'standardization']:
                if not metric:
                    x -= scaler.mean_[idx]
                x /= scaler.scale_[idx]
        return x

    def plot_visit_dist(self):
        """
        Plots the number of visits with at least one label for each time step.
        """
        # Plot number of available labels at each time step
        w = .3
        subsets = ['Train', 'Test']
        v_per_t = lambda x: np.sum(np.array(x != self.mask).any(2), axis=0)
        visits = [v_per_t(self.y_train[self.labels[0]]).reshape(-1), v_per_t(self.y_test[self.labels[0]]).reshape(-1)]
        x = np.arange(0, visits[0].shape[0], 1) + 1
        for i in range(2):
            plt.bar(x+w*i, visits[i], width=w, label=subsets[i])
        plt.xticks(x + w/2, x.astype(int))
        plt.xlabel('Time Step (t)')
        plt.ylabel('#Visits with at least one label')
        plt.legend()
        plt.tight_layout()
        plt.savefig(join(self.outdir, 'visits.png'))
        plt.close()

    def train_test_stats(self):
        """
        Prints the shape of x_train, y_yrain, x_test and y_test.
        Also creates a figure for each target to show the label distribution
        :return:
        """
        w = .3
        for p, l in zip(self.problem, self.labels):
            train_perc_labelled = (np.sum(np.array(self.y_train[l] != self.mask).reshape(-1))
                                   / self.y_train[l].reshape(-1).shape[0]) * 100
            test_perc_labelled = (np.sum(np.array(self.y_test[l] != self.mask).reshape(-1))
                                  / self.y_test[l].reshape(-1).shape[0]) * 100
            train_perc_masked = self.train_masked_weights[l].sum() / \
                                self.train_masked_weights[l].reshape(-1).shape[0] * 100
            test_perc_masked = self.test_masked_weights[l].sum() / \
                               self.test_masked_weights[l].reshape(-1).shape[0] * 100

            print('{}:\nTrain X: {} Train y: {} Train weights: {} ({:.2f}% labelled; {:.2f}% unmasked) \n'
                  'Test X : {} Test y : {} Test weights: {} ({:.2f}% labelled; {:.2f}% unmasked)'
                  .format(l, self.x_train.shape, self.y_train[l].shape, self.train_weights[l].shape,
                          train_perc_labelled, train_perc_masked,
                          self.x_test.shape, self.y_test[l].shape, self.test_weights[l].shape,
                          test_perc_labelled, test_perc_masked,
                          )
                  )
            values = []
            for t in range(self.y_train[l].shape[1]):
                train_values = self.y_train[l][:, t][self.y_train[l][:, t] != self.mask]
                for v in train_values:
                    values.append(['train', t, v])
                test_values = self.y_test[l][:, t][self.y_test[l][:, t] != self.mask]
                for v in test_values:
                    values.append(['test', t, v])
            df = pd.DataFrame(values, columns=['Dataset', 'Time Point (t)', l])
            if p == 'classification':
                # Group by value counts
                df = pd.DataFrame(df.groupby(['Dataset', 'Time Point (t)'])['drive'].value_counts())
                df.columns = ['count']
                df = df.reset_index()
                df['drive'] = df['drive'].astype(int)
                sns.catplot(data=df, x='Time Point (t)', y='count', hue='drive', col='Dataset', kind='bar',
                            col_order=['train', 'test'], height=8.67 * .4, aspect=(5.972 * .8) / (8.67 * .4))
            elif p == 'regression':
                plt.figure(figsize=(5.972 * .8, 8.67 * .4))
                df[l] = self.inverse_transform(df[l].values, l, metric=False)
                sns.boxplot(data=df, x='Time Point (t)', y=l, hue='Dataset')
                plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower center", ncol=2)
                plt.ylabel(l)
            plt.tight_layout()
            plt.savefig(join(self.outdir, l+'_labels.pdf'))
            plt.close()
