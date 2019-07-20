# Uncertainty Quantification in Deep Learning

[![Python 3.6+](https://img.shields.io/badge/Platform-Python%203.6-blue.svg)](https://www.python.org/)
[![PyTorch 1.1.0](https://img.shields.io/badge/Implementation-Pytorch-brightgreen.svg)](https://pytorch.org/)

This repo contains a literature survey and benchmark implementation for various approaches for predictive uncertainty estimation for deep learning models.  

## Literature survey

#### Basic background for uncertainty estimation 

- B. Efron. "Jackknife‐after‐bootstrap standard errors and influence functions." Journal of the Royal Statistical Society: Series B (Methodological), 1992. [[Link]](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1992.tb01866.x)

- J. Robins and A. Van Der Vaart. "Adaptive nonparametric confidence sets." The Annals of Statistics, 2006. [[Link]](https://projecteuclid.org/download/pdfview_1/euclid.aos/1146576262)

#### Predictive uncertainty for general machine learning models

- S. Wager, T. Hastie, and B. Efron. "Confidence intervals for random forests: The jackknife and the infinitesimal jackknife." The Journal of Machine Learning Research, 2014. [[Link]](http://jmlr.org/papers/volume15/wager14a/wager14a.pdf)

- L. Mentch and G. Hooker. "Quantifying uncertainty in random forests via confidence intervals and hypothesis tests." The Journal of Machine Learning Research, 2016. [[Link]](http://jmlr.org/papers/volume17/14-168/14-168.pdf)

#### Predictive uncertainty for deep learning

- B. Lakshminarayanan, A. Pritzel, and C. Blundell. "Simple and scalable predictive uncertainty estimation using deep ensembles." NeurIPS, 2017. [[Link]](http://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles.pdf)

- Y. Gal and Z. Ghahramani. "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." ICML, 2016. [[Link]](https://arxiv.org/pdf/1506.02142.pdf)

- V. Kuleshov, N. Fenner, and S. Ermon. "Accurate Uncertainties for Deep Learning Using Calibrated Regression." ICML, 2018. [[Link]](http://proceedings.mlr.press/v80/kuleshov18a/kuleshov18a.pdf)

- J. Hernández-Lobato and R. Adams. "Probabilistic backpropagation for scalable learning of bayesian neural networks." ICML, 2015. [[Link]](http://proceedings.mlr.press/v37/hernandez-lobatoc15.pdf)

- S. Liang, Y. Li, and R. Srikant. "Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks." ICLR, 2018. [[Link]](https://openreview.net/forum?id=H1VGkIxRZ)

- K. Lee, H. Lee, K. Lee, and J. Shin. "Training Confidence-calibrated classifiers for detecting out-of-distribution samples." ICLR, 2018. [[Link]](https://openreview.net/forum?id=ryiAv2xAZ)


## Implemented baselines

