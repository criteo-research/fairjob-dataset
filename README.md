# CRITEO FAIRNESS IN JOB ADS DATASET

TODO:
- Make script/explain how to download data
- Add code for plots and tables

## Summary

This dataset is released by Criteo to foster research and innovation on Fairness in Advertising and AI systems in general. 
See also [Criteo pledge for Fairness in Advertising](https://fr.linkedin.com/posts/diarmuid-gill_advertisingfairness-activity-6945003669964660736-_7Mu).

The dataset is intended to learn click predictions models and evaluate by how much their predictions are biased between different gender groups. 

## License

The data is released under the [CC-BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/) 4.0 license. 
You are free to Share and Adapt this data provided that you respect the Attribution, NonCommercial and ShareAlike conditions. 
Please read carefully the full license before using. 

## Data description
- 410,653 rows
  - each row represents a job ad
- features
  - `cat0` to `cat13`: categorical features of the ad, the publisher and user (categorical) - meaning undisclosed 
  - `num14` to `num48`: numerical features of the ad, the publisher and user - meaning undisclosed
- labels
  - `click`: was the ad clicked (binary)
  - `protected_attribute`: proxy for user gender (binary) - see below for more thorough explanation
  - `senior`: was the job offer for a senior position (binary)

### Data statistics

| dimension           | average |
|---------------------|---------|
| click               | 0.077   |
| protected attribute | 0.500   |
| senior              | 0.704   |

### Protected attribute

As Criteo does not have access to user demographics we report a proxy of gender as protected attribute. 
This proxy is reported as binary for simplicity yet we acknowledge gender is not necessarily binary.

The value of the proxy is computed as the majority of gender attributes of products seen in the user timeline.
Product having a gender attribute are typically fashion and clothing. 
We acknowledge that this proxy does not necessarily represent how users relate to a given gender yet we believe it to be a realistic approximation for research purposes.

We encourage research in Fairness defined with respect to other attributes as well.

## Metrics

We strongly recommend to measure prediction quality using Negative Log-likelihood (lower is better).

We recommend to measure Fairness of ads by Demographic Parity conditioned on Senior job offers:

$$ E[f(x) | protected\_attribute=1, senior=1] - E[f(x) | protected\_attribute=0, senior=1] $$

This corresponds to the average difference in predictions for senior job opportunities between the two gender groups (lower is better).
Intuitively, when this metric is low it means we are not biased towards presenting more senior job opportunities (e.g. Director of XXX) to one gender vs the other.

## Code structure and examples

The file `functions.py` implements all the functions and classes used in the experiments.
In order to run an example of model fit to FairJob data you can do:

```
python example_fit.py --dummy=1 --name=EXAMPLE
```

You can check all available options via:

```
python example_fit.py --help
```

If you want to run the experiment of logistic regression based on different randomization of train-test split of the data, you can do:

```
python example_simulations_LR.py --lr_fair=1 --fair_frac=1.0 --name=EXAMPLE
```

You can also in this case check all options available for `example_simulations_LR.py` with the flag `--help`.

## Paper results
In order to reproduce the results reported in the paper, please refer to the executions listed in `paper_results.sh` and to the notebook `dataset_analysis.ipynb` for the post-processing of the results.

## Citation

If you use the dataset in your research please cite it using the following Bibtex excerpt:

```
@misc{criteo_fairness_dataset
author = {CRITEO},
title = {CRITEO FAIRNESS IN JOB ADS DATASET},
year = {2024},
howpublished= {\url{http://XXX}}
}
```