# Uncertainty Quantification in Deep Learning

[![Python 3.6+](https://img.shields.io/badge/Platform-Python%203.6-blue.svg)](https://www.python.org/)
[![PyTorch 1.1.0](https://img.shields.io/badge/Implementation-Pytorch-brightgreen.svg)](https://pytorch.org/)

This repo contains literature survey and implementation of baselines for predictive uncertainty estimation in deep learning.  

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

- J. F. Lawless, and M. Fredette. "Frequentist prediction intervals and predictive distributions." Biometrika, 2005. [[Link]](https://ideas.repec.org/a/oup/biomet/v92y2005i3p529-542.html) 

- F. R. Hampel, E. M. Ronchetti, P. J. Rousseeuw, and W. A. Stahel. "Robust statistics: the approach based on influence functions." John Wiley and Sons, 2011. [[Link]](https://www.wiley.com/en-us/Robust+Statistics%3A+The+Approach+Based+on+Influence+Functions-p-9781118150689)

- P. J. Huber and E. M. Ronchetti. "Robust Statistics." John Wiley and Sons, 1981.

- Y. Romano, R. F. Barber, C. Sabatti, E. J. Candès. "With Malice Towards None: Assessing Uncertainty via Equalized Coverage." arXiv, 2019. [[Link]](https://arxiv.org/pdf/1908.05428.pdf)

- H. R. Kunsch. "The Jackknife and the Bootstrap for General Stationary Observations." The annals of Statistics, 1989. [[Link]](https://www.jstor.org/stable/pdf/2241719.pdf)


#### Predictive uncertainty for general machine learning models

- S. Wager, T. Hastie, and B. Efron. "Confidence intervals for random forests: The jackknife and the infinitesimal jackknife." The Journal of Machine Learning Research, 2014. [[Link]](http://jmlr.org/papers/volume15/wager14a/wager14a.pdf)

- L. Mentch and G. Hooker. "Quantifying uncertainty in random forests via confidence intervals and hypothesis tests." The Journal of Machine Learning Research, 2016. [[Link]](http://jmlr.org/papers/volume17/14-168/14-168.pdf)

- J. Platt. "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods." Advances in large margin classifiers, 1999. [[Link]](https://www.researchgate.net/profile/John_Platt/publication/2594015_Probabilistic_Outputs_for_Support_Vector_Machines_and_Comparisons_to_Regularized_Likelihood_Methods/links/004635154cff5262d6000000.pdf)

- A. Abadie, S. Athey, G. Imbens. "Sampling-based vs. design-based uncertainty in regression analysis." arXiv preprint (arXiv:1706.01778), 2017. [[Link]](https://arxiv.org/pdf/1706.01778.pdf) 

- T. Duan, A. Avati, D. Y. Ding, S. Basu, Andrew Y. Ng, and A. Schuler. "NGBoost: Natural Gradient Boosting for Probabilistic Prediction." arXiv preprint, 2019. [[Link]](https://arxiv.org/pdf/1910.03225.pdf) 

- V. Franc, and D. Prusa. "On Discriminative Learning of Prediction Uncertainty." ICML, 2019. [[Link]](http://proceedings.mlr.press/v97/franc19a/franc19a.pdf)

- Y. Romano, M. Sesia, and E. J. Candès. "Classification with Valid and Adaptive Coverage." arXiv preprint, 2020. [[Link]](https://arxiv.org/pdf/2006.02544.pdf) 


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

- A-A. Papadopoulos, M. R. Rajati, N. Shaikh, and J. Wang. "Outlier exposure with confidence control for out-of-distribution detection." arXiv preprint arXiv:1906.03509, 2019. [[Link]](https://arxiv.org/pdf/1906.03509.pdf)

- D. Madras, J. Atwood, A. D'Amour, "Detecting Extrapolation with Influence Functions." ICML Workshop on Uncertainty and Robustness in Deep Learning, 2019. [[Link]](http://www.gatsby.ucl.ac.uk/~balaji/udl2019/accepted-papers/UDL2019-paper-05.pdf)  

- M. Sensoy, L. Kaplan, and M. Kandemir. "Evidential deep learning to quantify classification uncertainty." NeurIPS, 2018. [[Link]](https://papers.nips.cc/paper/7580-evidential-deep-learning-to-quantify-classification-uncertainty.pdf)

- W. Maddox, T. Garipov, P. Izmailov, D. Vetrov, and A. G. Wilson. "A simple baseline for bayesian uncertainty in deep learning." arXiv preprint arXiv:1902.02476, 2019. [[Link]](https://arxiv.org/pdf/1902.02476.pdf)

- Y. Ovadia, et al. "Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift." arXiv preprint arXiv:1906.02530, 2019. [[Link]](https://arxiv.org/pdf/1906.02530.pdf)

- D. Hendrycks, et al. "Using Self-Supervised Learning Can Improve Model Robustness and Uncertainty." arXiv preprint arXiv:1906.12340, 2019. [[Link]](https://arxiv.org/pdf/1906.12340.pdf)

- A. Kumar, P. Liang, T. Ma. "Verified Uncertainty Calibration." arXiv preprint, 2019. [[Link]](https://arxiv.org/abs/1909.10155) 

- I. Osband, C. Blundell, A. Pritzel, and B. Van Roy. "Deep Exploration via Bootstrapped DQN." NeurIPS, 2016. [[Link]](https://papers.nips.cc/paper/6501-deep-exploration-via-bootstrapped-dqn.pdf) 

- I. Osband. "Risk versus Uncertainty in Deep Learning: Bayes, Bootstrap and the Dangers of Dropout." NeurIPS Workshop, 2016. [[Link]](http://bayesiandeeplearning.org/2016/papers/BDL_4.pdf) 

- J. Postels et al. "Sampling-free Epistemic Uncertainty Estimation Using Approximated Variance Propagation." ICCV, 2019. [[Link]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Postels_Sampling-Free_Epistemic_Uncertainty_Estimation_Using_Approximated_Variance_Propagation_ICCV_2019_paper.pdf)  

- A. Kendall and Y. Gal. "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" NeurIPS, 2017. [[Link]](https://arxiv.org/pdf/1703.04977.pdf) 

- N. Tagasovska and D. Lopez-Paz. "Single-Model Uncertainties for Deep Learning." NeurIPS, 2019. [[Link]](https://papers.nips.cc/paper/8870-single-model-uncertainties-for-deep-learning.pdf) 

- A. Der Kiureghian and O. Ditlevsen. "Aleatory or Epistemic? Does it Matter?." Structural Safety, 2009. [[Link]](https://www.sciencedirect.com/science/article/pii/S0167473008000556) 

- D. Hafner, D. Tran, A. Irpan, T. Lillicrap, and J. Davidson. "Reliable uncertainty estimates in deep neural networks using noise contrastive priors." arXiv, 2018. [[Link]](https://arxiv.org/pdf/1807.09289.pdf)

- S. Depeweg, J. M. Hernández-Lobato, F. Doshi-Velez, and S. Udluft. "Decomposition of uncertainty in Bayesian deep learning for efficient and risk-sensitive learning." ICML, 2018. [[Link]](http://publications.eng.cam.ac.uk/945907/)

- L. Smith and Y. Gal, "Understanding Measures of Uncertainty for Adversarial Example Detection." UAI, 2018. [[Link]](https://arxiv.org/pdf/1803.08533.pdf)

- L. Zhu and N. Laptev. "Deep and Confident Prediction for Time series at Uber." IEEE International Conference on Data Mining Workshops, 2017. [[Link]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8215650)

- M. W. Dusenberry, G. Jerfel, Y. Wen, Yi-an Ma, J. Snoek, K. Heller, B. Lakshminarayanan, D. Tran. "Efficient and Scalable Bayesian Neural Nets with Rank-1 Factors." arXiv, 2020. [[Link]](https://arxiv.org/abs/2005.07186)

- J. van Amersfoort, L. Smith, Y. W. Teh, and Y. Gal. "Uncertainty Estimation Using a Single Deep Deterministic Neural Network." ICML, 2020. [[Link]](https://arxiv.org/abs/2003.02037)

- E. Begoli, T. Bhattacharya and D. Kusnezov. "The need for uncertainty quantification in machine-assisted medical decision making." Nature Machine Intelligence, 2019. [[Link]](https://www.nature.com/articles/s42256-018-0004-1)  

- T. S. Salem, H. Langseth, and H. Ramampiaro. "Prediction Intervals: Split Normal Mixture from Quality-Driven Deep Ensembles." UAI, 2020. [[Link]](http://proceedings.mlr.press/v124/saleh-salem20a/saleh-salem20a.pdf)  

- K. Posch, and J. Pilz, "Correlated Parameters to Accurately Measure Uncertainty in Deep Neural Networks." IEEE Transactions on Neural Networks and Learning Systems, 2020. [[Link]](https://ieeexplore.ieee.org/document/9070148)

- B. Kompa, J. Snoek, and A. Beam. "Empirical Frequentist Coverage of Deep Learning Uncertainty Quantification Procedures." arXiv, 2020. [[Link]](https://arxiv.org/pdf/2010.03039.pdf)


#### Predictive uncertainty in sequential models

- R. Wen, K. Torkkola, B. Narayanaswamy, and D. Madeka. "A Multi-horizon Quantile Recurrent Forecaster." arXiv, 2017. 

- D. T. Mirikitani and N. Nikolaev. "Recursive bayesian recurrent neural networks for time-series modeling." IEEE Transactions on Neural Networks, 2009. [[Link]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5361332)

- M. Fortunato, C. Blundell and O. Vinyals. "Bayesian Recurrent Neural Networks." arXiv, 2019. [[Link]](https://arxiv.org/pdf/1704.02798.pdf)

- P. L. McDermott, C. K. Wikle. "Bayesian Recurrent Neural Network Models for Forecasting and Quantifying Uncertainty in Spatial-Temporal Data." Entropy, 2019. [[Link]](https://www.mdpi.com/1099-4300/21/2/184)

- Y. Gal, Z. Ghahramani. "A theoretically grounded application of dropout in recurrent neural networks." NeurIPS, 2016. [[Link]](https://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks.pdf) 

