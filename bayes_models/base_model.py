# necessary data management packages
import numpy as np
import pandas as pd

# main model solver package
import pystan as stan

# metric
from sklearn.metrics import r2_score

# modelcode
from .assemble_model_code import assemble_model_code


# base model
class SSModelBase:

    # this gets the model code
    def __init__(self, numeric_cols, target_col, cat_cols, is_reg):
        self.num_cols = numeric_cols
        self.y = target_col
        self.cat_cols_ = cat_cols
        self.is_reg = is_reg

    # this is our fit method
    def fit(self, df):

        # this gets a list of the categorial columns with unique values
        if len(self.cat_cols_) > 0:
            cat_cols = list(df.loc[:, self.cat_cols_].nunique().astype(str).to_dict().items())
        else:
            cat_cols = []

        model_code = assemble_model_code(cat_cols=cat_cols, is_regression=self.is_reg)

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

    # predict
    def _predict(self, test_df):

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


# regression class
class SSRegressor(SSModelBase):

    def __init__(self, numeric_cols, target_col, cat_cols):
        super(SSRegressor, self).__init__(numeric_cols, target_col, cat_cols, True)

    def predict(self, test_df):
        return self._predict(test_df)

    def score(self, x_val, y_val):
        if not 'model' in self.__dict__.keys():
            print('The estimator was not fitted')
            return None

        y_pred = self.predict(x_val)

        return r2_score(y_val, y_pred)


# classification class
class SSClassifier(SSModelBase):

    def __init__(self, numeric_cols, target_col, cat_cols):
        super(SSClassifier, self).__init__(numeric_cols, target_col, cat_cols, False)

    # this is the predict probability method
    # this predicts the probability of getting 0
    def predict_proba(self, test_df):
        if not 'model' in self.__dict__.keys():
            print('The estimator was not fitted')
            return None

        return super(SSClassifier, self)._predict(test_df)

    # this makes hard predictions
    def predict(self, test_df):
        raw_probs = (1.0 + np.exp(self.predict_proba(test_df))) ** -1.0

        raw_prob_1 = 1.0 - raw_probs

        labels = np.ones(test_df.shape[0], ).astype(int)

        labels[raw_prob_1 < 0.5] = 0

        return labels

    # this is the mean accuracy
    def score(self, x_val, y_val):
        return np.mean(self.predict(x_val) == y_val)
