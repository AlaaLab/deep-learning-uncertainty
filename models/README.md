# baselines

For all synthetic exp: compare coverage and efficiency vs number of samples

Metric: error rate withing subgroups

(TODO: implement error rate metric per subgroup)
(TODO: APIs for all baselines - Run experiment(baseline, data) API)
(TODO: Run experiments for all baselines in each setting)

- Standard CP [Done]
- Quantile regression [Done]
- Local CP (CP within reported subgroups) [Done]
- Conformalized quantile regression: https://github.com/yromano/cqr
- Conditional histograms: https://github.com/msesia/chr
- LCP (https://arxiv.org/abs/2106.08460) 
- Baselines from: https://github.com/msesia/cqr-comparison 
- Locally adaptive conformal prediction: https://github.com/yromano/cqr 
- Orthogonal quantile regression https://github.com/Shai128/oqr

Metric: efficiency within subgroups

TCP-RIF vs TCP-baselines

Data sets: Synthetic / UCI / Breast cancer
