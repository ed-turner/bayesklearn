# this is a module for bayesian decision tree learning

# base package
from .base_model import SSRegressor, SSClassifier

# important python packages that needs to be installed
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


# regressor for bayesian ensemble model
class BGRFRegressor(SSRegressor):

    def __init__(self, train_cols, target_col, n_estimators, max_depth):
        self.n_est = n_estimators
        self.m_depth = max_depth
        self.x = train_cols
        super(BGRFRegressor, self).__init__(['estimator_' + str(i) for i in range(1, n_estimators + 1)], target_col, [])

    # fits the model
    def fit(self, df):
        tree_rf = RandomForestRegressor(max_depth=self.m_depth, n_estimators=self.n_est, n_jobs=-1)
        tree_rf.fit(df[self.x], df[self.y])

        self.tree_model = tree_rf

        cols = ['estimator_' + str(i) for i in range(1, self.n_est + 1)]

        df_tree = pd.DataFrame(np.column_stack([est.predict(df[self.x]) for est in tree_rf.estimators_]), columns=cols)

        df_tree[self.y] = df[self.y]

        super(BGRFRegressor, self).fit(df_tree)

    # prediction method
    def predict(self, test_df):
        cols = ['estimator_' + str(i) for i in range(1, self.n_est + 1)]

        test_tree = pd.DataFrame(np.column_stack([est.predict(test_df[self.x]) for est in self.tree_model.estimators_]),
                                 columns=cols)

        return super(BGRFRegressor, self).predict(test_tree)


# classifier using logit function
class BGRFClassifier(SSClassifier):

    # initializes
    def __init__(self, train_cols, target_col, n_estimators, max_depth):
        self.n_est = n_estimators
        self.m_depth = max_depth
        self.x = train_cols
        super(BGRFClassifier, self).__init__(['estimator_' + str(i) for i in range(1, n_estimators + 1)], target_col,
                                             [])

    # fit method
    def fit(self, df):
        tree_rf = RandomForestClassifier(max_depth=self.m_depth, n_estimators=self.n_est, n_jobs=-1)
        tree_rf.fit(df[self.x], df[self.y])

        self.tree_model = tree_rf

        cols = ['estimator_' + str(i) for i in range(1, self.n_est + 1)]

        df_tree = pd.DataFrame(np.column_stack([est.predict(df[self.x]) for est in tree_rf.estimators_]), columns=cols)

        df_tree[self.y] = df[self.y]

        super(BGRFClassifier, self).fit(df_tree)

    # predict proba 
    def predict_proba(self, test_df):
        cols = ['estimator_' + str(i) for i in range(1, self.n_est + 1)]

        test_tree = pd.DataFrame(np.column_stack([est.predict(test_df[self.x]) for est in self.tree_model.estimators_]),
                                 columns=cols)

        return super(BGRFClassifier, self).predict_proba(test_tree)
