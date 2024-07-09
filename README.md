<!-- Copyright 2024 Mariia Vladimirova, Federico Pavone, Eustache Diemert

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. -->

# FairJob: A Real-World Dataset for Fairness in Online Systems

Official repository of the paper *FairJob: A Real-World Dataset for Fairness in Online Systems* (Mariia Vladimirova, Federico Pavone, Eustache Diemert) available on ArXiv (https://arxiv.org/abs/2407.03059).

## Data
The dataset FairJob and its detailed description is available at https://huggingface.co/datasets/criteo/FairJob.

Download the data in the subfolder `\data` before running the code in the repository.

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

### Paper results
In order to reproduce the results reported in the paper, please refer to the executions listed in `paper_results.sh` and to the notebook `dataset_analysis.ipynb` for the post-processing of the results.

## A Note on License
This code is open-source. We share most of it under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

## Citation
If you use the dataset in your research please cite it using the following Bibtex excerpt:

```
@article{vladimirova2024fairjob,
  title={{FairJob: A Real-World Dataset for Fairness in Online Systems}},
  author={Vladimirova, Mariia and Pavone, Federico and Diemert, Eustache},
  journal={arXiv preprint arXiv:2407.03059},
  year={2024}
}
```