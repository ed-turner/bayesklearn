Introduction
------------

The goal of this repository is to show how to integrate bayesian statistics within the
machine learning, while also allowing giving a sklearn-type API for the pystan package, which
is the python package to perform the bayesian inference in the backend.

Generally speaking, given the machine learning task and base algorithm, there are assumptions
about the distribution of the data.  However, there are not any assumptions made on the
learned parameters of the algorithm.  For example, the coefficients of the Linear Regression
model is taken as a uniform distribution in most highly distributed software packages. Given
the data, and perhaps the collinear nature of the features in the dataset, we may need to
give some regularization to the coefficients to prevent the coefficients to have large values.
Although this is the best thing to do in that particular case, it may be advantageous
to enforce a distribution of the coefficients, thereby regularizing the coefficients. The
bayesian methodology also allows categorical feature weights and intercepts.

Installation
--------------

One method of installing the python package, whether in a virtual environment
or your own local machine, is to git clone the repo, change the directory
to the python-package directory, and run `python setup.py install`.

Tutorial
--------

To use this model, simply follow this short example

.. code-block:: python

  from bayes_models.model import BGLRegressor

  model_params = {
  "numeric_cols": ["x"],
  "target_col": "y",
  "cat_cols": ["cat_col_1", "cat_col_2"]
                 }

  model = BGLRegressor(**model_params)

  model.fit(train_df)

  test_labels = model.predict(
                          test_df
                          )

As a note, it is suggested that all missing values are taken cared off before
using the model.

Documentation
-------------

For code documentations, please go `here <https://ed-turner.github.io/bayesklearn/>`_.

Or have a look at the code `repository <https://github.com/ed-turner/bayesklearn>`_.
