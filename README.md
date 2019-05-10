# Welcome to Bayesian Sci-kit-Learn 

A package to create models using a bayesian approach. 

## Introduction

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

## Implementation

TODO

## Tutorial

TODO

## Documentation

For code documentations, please go `here <https://ed-turner.github.io/bayesklearn/>`_.

Or have a look at the code `repository <https://github.com/ed-turner/bayesklearn>`_.

