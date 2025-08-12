# UncertaintyProject

This repository contains code for uncertainty approximation with different approaches (Bayesian Neural Nets, Monte Carlo Dropout, Quantile Regression) using neural networks.

We used public data from the University of Nicosia, though the focus of the data was on the US. It can be found [here](https://www.kaggle.com/competitions/m5-forecasting-uncertainty).

Our final results can be found [here](https://github.com/DerrickhSun/CS578-Uncertainty-Project/blob/main/CS_578_Project.pdf).

## Getting Started

Due to the size of the data set, the data cannot be kept in the repository. Please download the tables from the url (https://www.kaggle.com/competitions/m5-forecasting-uncertainty) then move it to the data folder.

Afterwards, run the jupyter notebook preprocessing.ipynb to process the data. This will clean, merge, and filter the data.

## Training

### Bayesian Neural Network
The Bayesian Neural Network leverages ELBO loss, using negative log likelihood to approximate both the mean data and the standard deviation. To run it:
```bash
python src/bayesian/bnn.py
```
This will both train and evaluate the model, displaying the evaluation metrics with plots.

### Monte Carlo Dropout
Monte Carlo Dropout introduces randomness into the neural network itself, deliberately allowing the neural network to make sub-optimal predictions to account for uncertainty. It can be run through the jupyter notebook src/mc_dropout/MCDropout.ipynb.

### Quantile Regression
Quantile Regression is an approach that asks the neural network to predict not only the mean, but varying percentiles. To run it:
```bash
python src/quantile_regression/quantile_regression.py
```
This will train a large number of neural networks, one for every 5 percentile, then save the models as "nn_p[number]". It will also evaluate it and output the loss as "loss_array_[number].npy".
