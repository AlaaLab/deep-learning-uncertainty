# Uncertainty Quantification in Deep Learning

[![Python 3.6+](https://img.shields.io/badge/Platform-Python%203.6-blue.svg)](https://www.python.org/)
[![PyTorch 1.1.0](https://img.shields.io/badge/Implementation-Pytorch-brightgreen.svg)](https://pytorch.org/)

This repo contains a literature survey and benchmark implementation for various approaches for predictive uncertainty estimation for deep learning models.  

## Literature survey

#### Basic background for uncertainty estimation 

- B. Efron and R. Tibshirani. "Bootstrap methods for standard errors, confidence intervals, and other measures of statistical accuracy." Statistical science, 1986. [[Link]](https://www.jstor.org/stable/pdf/2245500.pdf)

- R. Barber, E. J. Candes, A. Ramdas, and R. J. Tibshirani. "Predictive inference with the jackknife+." arXiv, 2019. [[Link]](https://arxiv.org/abs/1905.02928)

- B. Efron. "Jackknife‐after‐bootstrap standard errors and influence functions." Journal of the Royal Statistical Society: Series B (Methodological), 1992. [[Link]](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1992.tb01866.x)

- J. Robins and A. Van Der Vaart. "Adaptive nonparametric confidence sets." The Annals of Statistics, 2006. [[Link]](https://projecteuclid.org/download/pdfview_1/euclid.aos/1146576262)

- V. Vovk, et al., "Cross-conformal predictive distributions." JMLR, 2018. [[Link]](http://proceedings.mlr.press/v91/vovk18a/vovk18a.pdf) 

- M. H Quenouille., "Approximate tests of correlation in time-series." Journal of the Royal Statistical Society, 1949. [[Link]](https://www.jstor.org/stable/2983696?seq=1#metadata_info_tab_contents) 

- M. H Quenouille. "Notes on bias in estimation." Biometrika, 1956. [[Link]](https://www.jstor.org/stable/2332914?seq=1#metadata_info_tab_contents) 

- J. Tukey. "Bias and confidence in not quite large samples." Ann. Math. Statist, 1958. 

- R. G. Miller. "The jackknife–a review." Biometrika, 1974. [[Link]](https://www.jstor.org/stable/2334280?seq=1#metadata_info_tab_contents) 

- B. Efron. "Bootstrap methods: Another look at the jackknife." Ann. Statist., 1979. [[Link]](https://projecteuclid.org/euclid.aos/1176344552) 

- R. A Stine. "Bootstrap prediction intervals for regression." Journal of the American Statistical Association, 1985. [[Link]](https://amstat.tandfonline.com/doi/abs/10.1080/01621459.1985.10478220) 

- R. F. Barber, E. J. Candes, A. Ramdas, and R. J. Tibshirani. "Conformal prediction under covariate shift." arXiv preprint arXiv:1904.06019, 2019. [[Link]](https://arxiv.org/pdf/1904.06019.pdf) 

- R. F. Barber, E. J. Candes, A. Ramdas, and R. J. Tibshirani. "The limits of distribution-free conditional predictive inference." arXiv preprint arXiv:1903.04684, 2019b. [[Link]](https://arxiv.org/pdf/1903.04684.pdf) 

- J. Lei, M. G'Sell, A. Rinaldo, R. J. Tibshirani, and L. Wasserman. "Distribution-free predictive inference for regression." Journal of the American Statistical Association, 2018. [[Link]](https://www.tandfonline.com/doi/pdf/10.1080/01621459.2017.1307116) 

- R. Giordano, M. I. Jordan, and T. Broderick. "A Higher-Order Swiss Army Infinitesimal Jackknife." arXiv, 2019. [[Link]](https://arxiv.org/pdf/1907.12116.pdf) 

- P. W. Koh, K. Ang, H. H. K. Teo, and P. Liang. "On the Accuracy of Influence Functions for Measuring Group Effects." arXiv, 2019. [[Link]](https://arxiv.org/pdf/1905.13289.pdf)  

- D. H. Wolpert. "Stacked generalization." Neural networks, 1992. [[Link]](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.133.8090&rep=rep1&type=pdf)  

- R. D. Cook, and S. Weisberg. "Residuals and influence in regression." New York: Chapman and Hall, 1982. [[Link]](https://conservancy.umn.edu/handle/11299/37076)  

- R. Giordano, W. Stephenson, R. Liu, M. I. Jordan, and T. Broderick. "A Swiss Army Infinitesimal Jackknife." arXiv preprint arXiv:1806.00550, 2018. [[Link]](https://arxiv.org/pdf/1806.00550.pdf)  

- P. W. Koh, and P. Liang. "Understanding black-box predictions via influence functions." ICML, 2017. [[Link]](https://dl.acm.org/citation.cfm?id=3305576) 

- S. Wager and S. Athey. "Estimation and inference of heterogeneous treatment effects using random forests." Journal of the American Statistical Association, 2018. [[Link]](https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1319839) 


#### Predictive uncertainty for general machine learning models

- S. Wager, T. Hastie, and B. Efron. "Confidence intervals for random forests: The jackknife and the infinitesimal jackknife." The Journal of Machine Learning Research, 2014. [[Link]](http://jmlr.org/papers/volume15/wager14a/wager14a.pdf)

- L. Mentch and G. Hooker. "Quantifying uncertainty in random forests via confidence intervals and hypothesis tests." The Journal of Machine Learning Research, 2016. [[Link]](http://jmlr.org/papers/volume17/14-168/14-168.pdf)

#### Predictive uncertainty for deep learning

- J. A. Leonard, M. A. Kramer, and L. H. Ungar. "A neural network architecture that computes its own reliability." Computers & chemical engineering, 1992. [[Link]](https://www.sciencedirect.com/science/article/pii/0098135492800358)

- C. Blundell, J. Cornebise, K. Kavukcuoglu, and D. Wierstra. "Weight uncertainty in neural networks." ICML, 2015. [[Link]](https://arxiv.org/pdf/1505.05424.pdf) 

- B. Lakshminarayanan, A. Pritzel, and C. Blundell. "Simple and scalable predictive uncertainty estimation using deep ensembles." NeurIPS, 2017. [[Link]](http://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles.pdf)

- Y. Gal and Z. Ghahramani. "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." ICML, 2016. [[Link]](https://arxiv.org/pdf/1506.02142.pdf)

- V. Kuleshov, N. Fenner, and S. Ermon. "Accurate Uncertainties for Deep Learning Using Calibrated Regression." ICML, 2018. [[Link]](http://proceedings.mlr.press/v80/kuleshov18a/kuleshov18a.pdf)

- J. Hernández-Lobato and R. Adams. "Probabilistic backpropagation for scalable learning of bayesian neural networks." ICML, 2015. [[Link]](http://proceedings.mlr.press/v37/hernandez-lobatoc15.pdf)

- S. Liang, Y. Li, and R. Srikant. "Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks." ICLR, 2018. [[Link]](https://openreview.net/forum?id=H1VGkIxRZ)

- K. Lee, H. Lee, K. Lee, and J. Shin. "Training Confidence-calibrated classifiers for detecting out-of-distribution samples." ICLR, 2018. [[Link]](https://openreview.net/forum?id=ryiAv2xAZ)

- P. Schulam and S. Saria "Can You Trust This Prediction? Auditing Pointwise Reliability After Learning." AISTATS, 2019. [[Link]](http://proceedings.mlr.press/v89/schulam19a/schulam19a.pdf) 

- A. Malinin and M. Gales. "Predictive uncertainty estimation via prior networks." NeurIPS, 2018. [[Link]](http://papers.nips.cc/paper/7936-predictive-uncertainty-estimation-via-prior-networks.pdf) 

- D. Hendrycks, M. Mazeika, and T. G. Dietterich. "Deep anomaly detection with outlier exposure." arXiv preprint arXiv:1812.04606, 2018. [[Link]](https://arxiv.org/pdf/1812.04606.pdf)

- D. Madras, J. Atwood, A. D'Amour, "Detecting Extrapolation with Influence Functions." ICML Workshop on Uncertainty and Robustness in Deep Learning, 2019. [[Link]](http://www.gatsby.ucl.ac.uk/~balaji/udl2019/accepted-papers/UDL2019-paper-05.pdf)    

## Implemented baselines

- Bayesian neural networks (Pyro).
- Monte Carlo dropout (PyTorch).
- Bayes by backprop (PyTorch).
- Probabilistic backprop.
- Naive Jackknife, Jackknife-minmax, and Jackknife+ (PyTorch).
- Cross conformal and split conformal learning (PyTorch).
- Deep ensembles (PyTorch).
- Resampling uncertainty estimation (PyTorch).
