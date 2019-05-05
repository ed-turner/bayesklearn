from abc import ABCMeta, abstractmethod

# necessary data management packages
import numpy as np
import pandas as pd

# main model solver package
import pystan as stan

# metric
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


# base model
class SSModelBase(metaclass=ABCMeta):
    """

    """
    # this gets the model code
    def __init__(self, numeric_cols, target_col, cat_cols):
        """

        :param numeric_cols:
        :param target_col:
        :param cat_cols:
        """
        self.num_cols = numeric_cols
        self.y = target_col
        self.cat_cols_ = cat_cols

        self.model=None

    @abstractmethod
    def _assemble_model_code(self, cat_cols):
        pass

    # this is our fit method
    def _fit(self, df):
        """

        :param df:
        :return:
        """
        # this gets a list of the categorial columns with unique values
        if len(self.cat_cols_) > 0:
            cat_cols = list(df.loc[:, self.cat_cols_].nunique().astype(str).to_dict().items())
        else:
            cat_cols = []

        model_code = self._assemble_model_code(cat_cols=cat_cols)

        # this gets the data into the dictionary format for stan
        data = {'N': df.shape[0], 'p': len(self.num_cols), 'x': df.loc[:, self.num_cols].values, 'y': df[self.y].values}

        if len(cat_cols) == 0:
            data['y_mean'] = df[self.y].mean()
            data['y_std'] = df[self.y].std()

        else:
            for col in self.cat_cols_:
                data['y_mean_' + col] = df.groupby(col)[self.y].mean()
                data['y_std_' + col] = df.groupby(col)[self.y].std()
                data[col] = df[col].values

        data['nu'] = df.shape[0] - 1

        sm = stan.StanModel(
            model_code=model_code)

        # this is our fit
        self.model = sm.sampling(data=data, n_jobs=-1)

    @abstractmethod
    def fit(self, df):
        pass

    # predict
    def _predict(self, test_df):
        """

        :param test_df:
        :return:
        """
        parameter_values = self.model.extract()

        # we take the mean of the last 1000 betas
        beta = parameter_values.get('beta')[-1000:].mean(axis=0)

        res = test_df.loc[:, self.num_cols].values.dot(beta).reshape(test_df.shape[0], )

        # we average the last 1000 shifts
        if len(self.cat_cols_) > 0:
            for cat_col in self.cat_cols_:
                res += parameter_values.get('shift_' + cat_col)[-1000:].mean(axis=0)[test_df.loc[:, cat_col].values - 1]

        else:
            res += parameter_values.get('shift')[-1000:].mean()

        return res

    @abstractmethod
    def score(self, x_val, y_val):
        pass


class AssembleRegressionModelCode:

    @staticmethod
    def __assemble_data(cat_cols):
        """

        :param cat_cols:
        :return:
        """
        # we get our input sizes
        feature_sizes = 'int<lower = 1> N; int<lower = 1> p; '

        # we assemble our vector variables
        features = feature_sizes + 'matrix[N, p] x; '

        features += 'vector[N] y; real<lower=0> nu; '

        # we return our vector variables if we do not have any category columns
        if len(cat_cols) == 0:
            features += 'real y_mean; real y_std; '
            return features

        # else, we return the effects for each categorial feature
        else:
            categorials = ['int ' + cat_col[0] + '[N]' for cat_col in cat_cols]
            categorials += ['vector[' + cat_col[1] + '] y_mean_' + cat_col[0] for cat_col in cat_cols]
            categorials += ['vector[' + cat_col[1] + '] y_std_' + cat_col[0] for cat_col in cat_cols]

            return features + '; '.join(categorials) + '; '

    @staticmethod
    def __assemble_parameters(cat_cols):
        """

        :param cat_cols:
        :return:
        """

        # if there were not any categorial faetures, we just return the beta parameter and the sigma parameter
        if len(cat_cols) == 0:
            params = 'vector[p] beta; real<lower = 0> sigma; real shift; real mu_b; real<lower=0> sigma_b; real mu_s; ' \
                     'real<lower=0> sigma_s; '

            return params

        # these are for the shifts.. we may set this as a separate variable until we figure out the beta problem
        means = ['real mu_' + cat_col[0] for cat_col in cat_cols] + ['real mu_b']
        sigmas = ['real<lower=0> sigma_' + cat_col[0] for cat_col in cat_cols] + ['real<lower=0> sigma_b',
                                                                                  'real<lower=0> sigma']

        # we assume the weights are all the same for now
        betas = ['vector[p] beta']
        shifts = ['vector[' + cat_col[1] + '] shift_' + cat_col[0] for cat_col in cat_cols]

        params = '; '.join(means + sigmas + betas + shifts) + '; '

        return params

    # this assembles the model depending on whether we are performing regression
    @staticmethod
    def __assemble_model_code(cat_cols):
        """

        :param cat_cols:
        :return:
        """
        if len(cat_cols) == 0:
            weight_dis = 'mu_b ~ normal(0, 100); sigma_b ~ normal(0, 20); beta ~ normal(mu_b, sigma_b); '
            shifts_dis = 'mu_s ~ normal(y_mean, y_std); sigma_s ~ normal(0,2); shift ~ normal(mu_s, sigma_s); '

            model = 'sigma ~ normal(0, 20); y ~ student_t(nu, x * beta + shift, sigma ); '

        else:

            # this is a helper function
            def setup_shift_dist(col):
                mu_dist = 'mu_' + col + ' ~ normal(y_mean_' + col + ', y_std_' + col + '); '
                sigma_dist = 'sigma_' + col + ' ~ normal(0, 2); '
                return mu_dist + sigma_dist + 'shift_' + col + ' ~ normal(mu_' + col + ', sigma_' + col + '); '

            shifts_dis = ' '.join([setup_shift_dist(cat_col[0]) for cat_col in cat_cols])
            weight_dis = 'mu_b ~ normal(0, 100); sigma_b ~ normal(0, 20); beta ~ normal(mu_b, sigma_b); '

            shifts = '+ '.join(['shift_' + cat_col[0] + '[' + cat_col[0] + ']' for cat_col in cat_cols])
            weighted = 'x*beta'

            model = 'sigma ~ normal(0, 20); y ~ student_t(nu, ' + weighted + ' + ' + shifts + ', sigma); '

        return weight_dis + shifts_dis + model

    def _assemble_model_code(self, cat_cols):
        """

        :param cat_cols:
        :return:
        """
        data = self.__assemble_data(cat_cols)

        params = self.__assemble_parameters(cat_cols)

        model = self.__assemble_model_code(cat_cols)

        return 'data {' + data + '} parameters {' + params + '} model {' + model + '}'


class AssembleClassificationModelCode:

    @staticmethod
    def __assemble_data(cat_cols):
        """

        :param cat_cols:
        :return:
        """
        # we get our input sizes
        feature_sizes = 'int<lower = 1> N; int<lower = 1> p; '

        # we assemble our vector variables
        features = feature_sizes + 'matrix[N, p] x; '

        features += 'int<lower=0, upper=1> y[N]; real<lower=0> nu;'

        # we return our vector variables if we do not have any category columns
        if len(cat_cols) == 0:
            features += 'real y_mean; real y_std; '
            return features

        # else, we return the effects for each categorial feature
        else:
            categorials = ['int ' + cat_col[0] + '[N]' for cat_col in cat_cols]
            categorials += ['vector[' + cat_col[1] + '] y_mean_' + cat_col[0] for cat_col in cat_cols]
            categorials += ['vector[' + cat_col[1] + '] y_std_' + cat_col[0] for cat_col in cat_cols]

            return features + '; '.join(categorials) + '; '

    @staticmethod
    def __assemble_parameters(cat_cols):
        """

        :param cat_cols:
        :return:
        """

        # if there were not any categorial faetures, we just return the beta parameter and the sigma parameter
        if len(cat_cols) == 0:
            params = 'vector[p] beta; real<lower = 0> sigma; real shift; real mu_b; real<lower=0> sigma_b; real mu_s; ' \
                     'real<lower=0> sigma_s; '

            params += 'vector[N] eta; '

            return params

        # these are for the shifts.. we may set this as a separate variable until we figure out the beta problem
        means = ['real mu_' + cat_col[0] for cat_col in cat_cols] + ['real mu_b']
        sigmas = ['real<lower=0> sigma_' + cat_col[0] for cat_col in cat_cols] + ['real<lower=0> sigma_b',
                                                                                  'real<lower=0> sigma']

        # we assume the weights are all the same for now
        betas = ['vector[p] beta']
        shifts = ['vector[' + cat_col[1] + '] shift_' + cat_col[0] for cat_col in cat_cols]

        params = '; '.join(means + sigmas + betas + shifts) + '; '

        params += 'vector[N] eta; '

        return params

    # this assembles the model depending on whether we are performing regression
    @staticmethod
    def __assemble_model_code(cat_cols):
        """

        :param cat_cols:
        :return:
        """
        if len(cat_cols) == 0:
            weight_dis = 'mu_b ~ normal(0, 100); sigma_b ~ normal(0, 5); beta ~ normal(mu_b, sigma_b); '
            shifts_dis = 'mu_s ~ normal(y_mean, y_std); sigma_s ~ normal(0,2); shift ~ normal(mu_s, sigma_s); '

            model = 'sigma ~ normal(0, 20); eta ~ normal( x * beta + shift, sigma ); y ~ bernoulli_logit( eta ); '

        else:
            # this is a helper function
            def setup_shift_dist(col):
                mu_dist = 'mu_' + col + ' ~ normal(y_mean_' + col + ', y_std_' + col + '); '
                sigma_dist = 'sigma_' + col + ' ~ normal(0, 2); '
                return mu_dist + sigma_dist + 'shift_' + col + ' ~ normal(mu_' + col + ', sigma_' + col + '); '

            shifts_dis = ' '.join([setup_shift_dist(cat_col[0]) for cat_col in cat_cols])
            weight_dis = 'mu_b ~ normal(0, 100); sigma_b ~ normal(0, 20); beta ~ normal(mu_b, sigma_b); '

            shifts = '+ '.join(['shift_' + cat_col[0] + '[' + cat_col[0] + ']' for cat_col in cat_cols])
            eqn = 'x*beta + ' + shifts

            model = 'sigma ~ normal(0, 20); eta ~ student_t(nu,' + eqn + ', sigma); y ~ bernoulli_logit(eta); '

        return weight_dis + shifts_dis + model

    def _assemble_model_code(self, cat_cols):
        """

        :param cat_cols:
        :return:
        """
        data = self.__assemble_data(cat_cols)

        params = self.__assemble_parameters(cat_cols)

        model = self.__assemble_model_code(cat_cols)

        return 'data {' + data + '} parameters {' + params + '} model {' + model + '}'


# regressor using bayesian gauassian linear regression
class BGLRegressor(AssembleRegressionModelCode, SSModelBase):
    """

    """
    def fit(self, df):
        self._fit(df)

    def predict(self, test_df):
        """

        :param test_df:
        :return:
        """
        return self._predict(test_df)

    def score(self, x_val, y_val):
        """

        :param x_val:
        :param y_val:
        :return:
        """
        if self.model is None:
            raise RuntimeError

        y_pred = self.predict(x_val)

        return r2_score(y_val, y_pred)


# classification class
class BGLClassifier(AssembleClassificationModelCode, SSModelBase):
    """

    """
    def fit(self, df):
        """

        :param df:
        :return:
        """
        self._fit(df)

    # this is the predict probability method
    # this predicts the probability of getting 0
    def predict_proba(self, test_df):
        """

        :param test_df:
        :return:
        """
        if not 'model' in self.__dict__.keys():
            print('The estimator was not fitted')
            return None

        return self._predict(test_df)

    # this makes hard predictions
    def predict(self, test_df):
        """

        :param test_df:
        :return:
        """
        raw_probs = (1.0 + np.exp(self.predict_proba(test_df))) ** -1.0

        raw_prob_1 = 1.0 - raw_probs

        labels = np.ones(test_df.shape[0], ).astype(int)

        labels[raw_prob_1 < 0.5] = 0

        return labels

    # this is the mean accuracy
    def score(self, x_val, y_val):
        """

        :param x_val:
        :param y_val:
        :return:
        """
        return np.mean(self.predict(x_val) == y_val)


# regressor for bayesian ensemble model
class BGRFRegressor(AssembleRegressionModelCode, SSModelBase):
    """

    """
    def __init__(self, train_cols, target_col, n_estimators, max_depth):
        """

        :param train_cols:
        :param target_col:
        :param n_estimators:
        :param max_depth:
        """
        self.n_est = n_estimators
        self.m_depth = max_depth
        self.x = train_cols
        super(BGRFRegressor, self).__init__(['estimator_' + str(i) for i in range(1, n_estimators + 1)], target_col, [])
        self.tree_model = None

    # fits the model
    def fit(self, df):
        """

        :param df:
        :return:
        """
        tree_rf = RandomForestRegressor(max_depth=self.m_depth, n_estimators=self.n_est, n_jobs=-1)
        tree_rf.fit(df[self.x], df[self.y])

        self.tree_model = tree_rf

        cols = ['estimator_' + str(i) for i in range(1, self.n_est + 1)]

        df_tree = pd.DataFrame(np.column_stack([est.predict(df[self.x]) for est in tree_rf.estimators_]), columns=cols)

        df_tree[self.y] = df[self.y]

        self._fit(df_tree)

    # prediction method
    def predict(self, test_df):
        """

        :param test_df:
        :return:
        """
        cols = ['estimator_' + str(i) for i in range(1, self.n_est + 1)]

        test_tree = pd.DataFrame(np.column_stack([est.predict(test_df[self.x]) for est in self.tree_model.estimators_]),
                                 columns=cols)

        return self._predict(test_tree)

    def score(self, x_val, y_val):
        """

        :param x_val:
        :param y_val:
        :return:
        """
        if self.model is None:
            raise RuntimeError

        y_pred = self.predict(x_val)

        return r2_score(y_val, y_pred)


# classifier using logit function
class BGRFClassifier(AssembleClassificationModelCode, SSModelBase):
    """

    """
    # initializes
    def __init__(self, train_cols, target_col, n_estimators, max_depth):
        """

        :param train_cols:
        :param target_col:
        :param n_estimators:
        :param max_depth:
        """
        self.n_est = n_estimators
        self.m_depth = max_depth
        self.x = train_cols
        super(BGRFClassifier, self).__init__(['estimator_' + str(i) for i in range(1, n_estimators + 1)], target_col,
                                             [])
        self.tree_model = None

    # fit method
    def fit(self, df):
        """

        :param df:
        :return:
        """
        tree_rf = RandomForestClassifier(max_depth=self.m_depth, n_estimators=self.n_est, n_jobs=-1)
        tree_rf.fit(df[self.x], df[self.y])

        self.tree_model = tree_rf

        cols = ['estimator_' + str(i) for i in range(1, self.n_est + 1)]

        df_tree = pd.DataFrame(np.column_stack([est.predict(df[self.x]) for est in tree_rf.estimators_]), columns=cols)

        df_tree[self.y] = df[self.y]

        self._fit(df_tree)

    # predict proba
    def predict_proba(self, test_df):
        """

        :param test_df:
        :return:
        """
        cols = ['estimator_' + str(i) for i in range(1, self.n_est + 1)]

        test_tree = pd.DataFrame(np.column_stack([est.predict(test_df[self.x]) for est in self.tree_model.estimators_]),
                                 columns=cols)

        return self._predict(test_tree)

    # this makes hard predictions
    def predict(self, test_df):
        """

        :param test_df:
        :return:
        """
        raw_probs = (1.0 + np.exp(self.predict_proba(test_df))) ** -1.0

        raw_prob_1 = 1.0 - raw_probs

        labels = np.ones(test_df.shape[0], ).astype(int)

        labels[raw_prob_1 < 0.5] = 0

        return labels

    # this is the mean accuracy
    def score(self, x_val, y_val):
        """

        :param x_val:
        :param y_val:
        :return:
        """
        return np.mean(self.predict(x_val) == y_val)
