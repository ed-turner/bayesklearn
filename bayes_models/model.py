import logging
from abc import ABCMeta, abstractmethod

# necessary data management packages
import numpy as np
import pandas as pd

# main model solver package
import pystan as stan

# metric
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def _assert_is_list(x):
    """
    This is just a helper to make sure each of the attributes is a list
    :param x:
    :return:
    """
    assert isinstance(x, list)


def _assert_in_dataframe(df, x):
    """
    This method asserts that the values of x is in the dataframe columns object

    :param df: The input dataframe
    :type df: pandas.DataFrame
    :param x: The iterable
    :type x: collections.iterable
    :return:
    """

    assert np.all([_x in df.columns for _x in x])


def _assert_is_dataframe(df):
    """
    This method only asserts that the input is a pandas.DataFrame

    :param df: The input
    :type df: Any
    :return:
    """
    assert isinstance(df, pd.DataFrame)


class _ModelDecorators:
    @classmethod
    def log_all_errors(cls, logger):
        def run_function(decorated):
            def tmp_funct(*args, **kwargs):
                try:
                    return decorated(*args, **kwargs)
                except Exception as e:
                    logger.error(e, exc_info=True)
                    raise e

            return tmp_funct
        return run_function


# base model
class _SSModelBase(metaclass=ABCMeta):
    """
    This is a base class to help organize the common behaviors the child classes should inherit.  Since each class
    will assemble it's own model code, we will delegate that to it's own separate class.  We will just handle the
    model behaviors
    """

    # this is the class logger
    _logger = logging.getLogger(__name__)
    format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    _logger.addHandler(logging.StreamHandler().setFormatter(format))

    # this gets the model code
    def __init__(self, numeric_cols, target_col, cat_cols):
        """
        This is just the initializer for the class.  We assume the input to the fit method is a pandas.DataFrame.

        So we are assuming each of the list of columns passed are columns in the pandas.DataFrame passed

        :param numeric_cols: The pure numeric columns for our modelling tasks
        :type numeric_cols: list
        :param target_col: The column we want to predict
        :type target_col: str
        :param cat_cols: The pure categorical columns for our modelling tasks
        :type cat_cols: list
        """

        _assert_is_list(numeric_cols)

        self.num_cols = numeric_cols
        self.y = target_col

        _assert_is_list(cat_cols)
        self.cat_cols_ = cat_cols

        self.model = None

    @_ModelDecorators.log_all_errors(logger=_logger)
    @abstractmethod
    def _assemble_model_code(self, cat_cols):
        """
        This is to be overrided in a child class.

        :param cat_cols: The pure categorical columns for our modelling tasks
        :return:
        """
        pass

    # this is our fit method
    @_ModelDecorators.log_all_errors(logger=_logger)
    def _fit(self, df):
        """
        This is a protected method and should not be overloaded for whathever reason.

        :param df: Our dataset with our features and predictive columns
        :type df: pandas.DataFrame
        :return: self
        """

        _assert_is_dataframe(df)

        # we assert the columns are in the dataset
        _assert_in_dataframe(df, self.num_cols)
        _assert_in_dataframe(df, self.cat_cols_)
        assert self.y in df.columns

        # this gets a list of the categorial columns with unique values
        if len(self.cat_cols_) > 0:
            cat_cols = list(df.loc[:, self.cat_cols_].nunique().astype(str).to_dict().items())
        else:
            cat_cols = []

        self._logger.info("Assembling the model code")
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
        self._logger.info("Fitting the model")
        self.model = sm.sampling(data=data, n_jobs=-1)

    @abstractmethod
    def fit(self, df):
        pass

    # predict
    @_ModelDecorators.log_all_errors(logger=_logger)
    def _predict(self, test_df):
        """
        This predicts the raw values of our test dataframe

        :param test_df: The dataset we will predict on.
        :type test_df: pandas.DataFrame
        :return: res
        """
        _assert_is_dataframe(test_df)

        _assert_in_dataframe(test_df, self.num_cols)
        _assert_in_dataframe(test_df, self.cat_cols_)

        self._logger.info("Extracting our values")
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


class _AssembleRegressionModelCode:
    """
    This class defines how we are going to handle regression-based tasks.
    """
    @staticmethod
    def __assemble_data(cat_cols):
        """
        For regression based tasks, we assemble our model code for the Stan package.

        :param cat_cols: The pure categorical columns for our modelling tasks
        :type cat_cols: list
        :return:
        """

        _assert_is_list(cat_cols)

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

        :param cat_cols: The pure categorical columns for our modelling tasks
        :type cat_cols: list
        :return:
        """
        _assert_is_list(cat_cols)

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

        :param cat_cols: The pure categorical columns for our modelling tasks
        :type cat_cols: list
        :return:
        """

        _assert_is_list(cat_cols)

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

        :param cat_cols: The pure categorical columns for our modelling tasks
        :type cat_cols: list
        :return:
        """

        _assert_is_list(cat_cols)

        data = self.__assemble_data(cat_cols)

        params = self.__assemble_parameters(cat_cols)

        model = self.__assemble_model_code(cat_cols)

        return 'data {' + data + '} parameters {' + params + '} model {' + model + '}'


class _AssembleClassificationModelCode:
    """
        This class defines how we are going to handle classification-based tasks.
    """

    @staticmethod
    def __assemble_data(cat_cols):
        """

        :param cat_cols: The pure categorical columns for our modelling tasks
        :type cat_cols: list
        :return:
        """

        _assert_is_list(cat_cols)

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

        :param cat_cols: The pure categorical columns for our modelling tasks
        :type cat_cols: list
        :return:
        """

        _assert_is_list(cat_cols)

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

        :param cat_cols: The pure categorical columns for our modelling tasks
        :type cat_cols: list
        :return:
        """

        _assert_is_list(cat_cols)

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

        :param cat_cols: The pure categorical columns for our modelling tasks
        :type cat_cols: list
        :return:
        """

        _assert_is_list(cat_cols)

        data = self.__assemble_data(cat_cols)

        params = self.__assemble_parameters(cat_cols)

        model = self.__assemble_model_code(cat_cols)

        return 'data {' + data + '} parameters {' + params + '} model {' + model + '}'


# regressor using bayesian gauassian linear regression
class BGLRegressor(_AssembleRegressionModelCode, _SSModelBase):
    """

    This class is responsible for performing Bayesian-Gaussian Linear Regression, which treats categorical variables
    correctly.

    """
    def fit(self, df):
        """
        This method is responsible for fitting the model onto the data

        :param df: Our dataset with our features and predictive columns
        :type df: pandas.DataFrame
        :return: self
        """
        return self._fit(df)

    def predict(self, test_df):
        """
        This predicts the raw values of our test dataframe

        :param test_df: The dataset we will predict on.
        :type test_df: pandas.DataFrame
        :return: res
        """
        return self._predict(test_df)

    def score(self, x_val, y_val):
        """
        The score returned is the R2 Score between the predicted and the actual.

        :param x_val: The validation dataset
        :type x_val: pandas.DataFrame
        :param y_val: The validation actual values
        :type y_val: numpy.array or pandas.DataFrame
        :return: R2 Score of Predictions
        """
        if self.model is None:
            raise RuntimeError

        y_pred = self.predict(x_val)

        return r2_score(y_val, y_pred)


# classification class
class BGLClassifier(_AssembleClassificationModelCode, _SSModelBase):
    """

    This class is responsible for performing Bayesian-Gaussian Logistic Regression, which treats categorical variables
    correctly.

    """
    def fit(self, df):
        """
        This method is responsible for fitting the model onto the data

        :param df: Our dataset with our features and predictive columns
        :type df: pandas.DataFrame
        :return: self
        """
        return self._fit(df)

    # this is the predict probability method
    # this predicts the probability of getting 0
    def predict_proba(self, test_df):
        """
        This predicts the raw probabilities of our test dataframe

        :param test_df: The dataset we will predict on.
        :type test_df: pandas.DataFrame
        :return: res
        """
        if not 'model' in self.__dict__.keys():
            print('The estimator was not fitted')
            return None

        return self._predict(test_df)

    # this makes hard predictions
    def predict(self, test_df):
        """
        This predicts the binary values of our test dataframe

        :param test_df: The dataset we will predict on.
        :type test_df: pandas.DataFrame
        :return: res
        """
        raw_probs = (1.0 + np.exp(self.predict_proba(test_df))) ** -1.0

        raw_prob_1 = 1.0 - raw_probs

        labels = np.ones(test_df.shape[0], ).astype(int)

        labels[raw_prob_1 < 0.5] = 0

        return labels

    # this is the mean accuracy
    def score(self, x_val, y_val):
        """
        This returns the accuracy of the predictions

        :param x_val: The validation dataset
        :type x_val: pandas.DataFrame
        :param y_val: The validation actual values
        :type y_val: numpy.array or pandas.DataFrame
        :return:
        """
        return np.mean(self.predict(x_val) == y_val)


# regressor for bayesian ensemble model
class BGRFRegressor(_AssembleRegressionModelCode, _SSModelBase):
    """

    This class is responsible for performing Bayesian-Gaussian Linear Regression on the predictions of the Decision Trees
    from the fitted sklearn.ensemble.RandomForestRegressor model.  This helps us optimally derive the best weighting
    based on the predictions of the individual Decision Trees.

    """
    def __init__(self, train_cols, target_col, n_estimators, max_depth):
        """
        :param train_cols: The columns in the dataframe to be passed in the fit method
        :type train_cols: list
        :param target_col: The column to be predicted
        :type target_col: str
        :param n_estimators: The number of trees to use
        :type n_estimators: int
        :param max_depth: The maximum depth our trees to grow
        :type max_depth: int
        """
        assert isinstance(n_estimators, int)
        self.n_est = n_estimators

        assert isinstance(max_depth, int)

        self.m_depth = max_depth

        _assert_is_list(train_cols)
        self.x = train_cols

        super(BGRFRegressor, self).__init__(['estimator_' + str(i) for i in range(1, n_estimators + 1)], target_col, [])
        self.tree_model = None

    # fits the model
    def fit(self, df):
        """
        First, we will fit the RandomForest on our dataset.
        Then we will use those predictions as features for our Bayesian Model

        :param df: Our dataset with our features and predictive columns
        :type df: pandas.DataFrame
        :return: self
        """

        tree_rf = RandomForestRegressor(max_depth=self.m_depth, n_estimators=self.n_est, n_jobs=-1)
        tree_rf.fit(df[self.x], df[self.y])

        self.tree_model = tree_rf

        cols = ['estimator_' + str(i) for i in range(1, self.n_est + 1)]

        df_tree = pd.DataFrame(np.column_stack([est.predict(df[self.x]) for est in tree_rf.estimators_]), columns=cols)

        df_tree[self.y] = df[self.y]

        return self._fit(df_tree)

    # prediction method
    def predict(self, test_df):
        """
        This predicts the raw values of our test dataframe

        :param test_df: The dataset we will predict on.
        :type test_df: pandas.DataFrame
        :return: res
        """
        cols = ['estimator_' + str(i) for i in range(1, self.n_est + 1)]

        test_tree = pd.DataFrame(np.column_stack([est.predict(test_df[self.x]) for est in self.tree_model.estimators_]),
                                 columns=cols)

        return self._predict(test_tree)

    def score(self, x_val, y_val):
        """
        The score returned is the R2 Score between the predicted and the actual.

        :param x_val: The validation dataset
        :type x_val: pandas.DataFrame
        :param y_val: The validation actual values
        :type y_val: numpy.array or pandas.DataFrame
        :return:
        """
        if self.model is None:
            raise RuntimeError

        y_pred = self.predict(x_val)

        return r2_score(y_val, y_pred)


# classifier using logit function
class BGRFClassifier(_AssembleClassificationModelCode, _SSModelBase):
    """

    This class is responsible for performing Bayesian-Gaussian Logistic Regression on the predictions of the Decision Trees
    from the fitted sklearn.ensemble.RandomForestClassifier model.  This helps us optimally derive the best weighting
    based on the predictions of the individual Decision Trees.

    """
    # initializes
    def __init__(self, train_cols, target_col, n_estimators, max_depth):
        """
        :param train_cols: The columns in the dataframe to be passed in the fit method
        :type train_cols: list
        :param target_col: The column to be predicted
        :type target_col: str
        :param n_estimators: The number of trees to use
        :type n_estimators: int
        :param max_depth: The maximum depth our trees to grow
        :type max_depth: int
        """
        assert isinstance(n_estimators, int)
        self.n_est = n_estimators

        assert isinstance(max_depth, int)

        self.m_depth = max_depth

        _assert_is_list(train_cols)
        self.x = train_cols

        super(BGRFClassifier, self).__init__(['estimator_' + str(i) for i in range(1, n_estimators + 1)], target_col,
                                             [])
        self.tree_model = None

    # fit method
    def fit(self, df):
        """
        First, we will fit the RandomForest on our dataset.
        Then we will use those predictions as features for our Bayesian Model

        :param df: Our dataset with our features and predictive columns
        :type df: pandas.DataFrame
        :return: self
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
        This predicts the raw values of our test dataframe

        :param test_df: The dataset we will predict on.
        :type test_df: pandas.DataFrame
        :return: res
        """
        cols = ['estimator_' + str(i) for i in range(1, self.n_est + 1)]

        test_tree = pd.DataFrame(np.column_stack([est.predict(test_df[self.x]) for est in self.tree_model.estimators_]),
                                 columns=cols)

        return self._predict(test_tree)

    # this makes hard predictions
    def predict(self, test_df):
        """
        This predicts the binary values of our test dataframe

        :param test_df: The dataset we will predict on.
        :type test_df: pandas.DataFrame
        :return: res
        """
        raw_probs = (1.0 + np.exp(self.predict_proba(test_df))) ** -1.0

        raw_prob_1 = 1.0 - raw_probs

        labels = np.ones(test_df.shape[0], ).astype(int)

        labels[raw_prob_1 < 0.5] = 0

        return labels

    # this is the mean accuracy
    def score(self, x_val, y_val):
        """
        This returns the accuracy of the predictions

        :param x_val: The validation dataset
        :type x_val: pandas.DataFrame
        :param y_val: The validation actual values
        :type y_val: numpy.array or pandas.DataFrame
        :return:
        """
        return np.mean(self.predict(x_val) == y_val)
