# Private post-GAN Boosting

This is the code repository accompanying the paper **Private post-GAN Boosting** by [Marcel Neunhoeffer](https://marcel-neunhoeffer.com), [Zhiwei Steven Wu](https://zstevenwu.com/) and [Cynthia Dwork](https://www.seas.harvard.edu/faculty?search=%22Cynthia%20Dwork%22#content).

If you have any questions feel free to email [Marcel Neunhoeffer](mailto:mneunhoe@mail.uni-mannheim.de).


*Abstract*:

Differentially private GANs have proven to be a promising approach for generating realistic synthetic data without compromising the privacy of individuals. However, due to the privacy-protective noise introduced in the  training, the convergence of GANs becomes even more elusive, which often leads to poor utility in the output generator at the end of training. We propose Private **post-GAN boosting (Private PGB)**, a differentially private method that combines samples produced by the sequence of generators obtained during GAN training to create a high-quality synthetic dataset. Our method leverages the Private Multiplicative Weights method (Hardt and Rothblum, 2010) and the discriminator rejection sampling technique (Azadi et al., 2019) for reweighting generated samples, to obtain high quality synthetic data even in cases where GAN training does not converge.  We evaluate Private PGB on a Gaussian mixture dataset and two US Census datasets, and demonstrate that Private PGB improves upon the standard private GAN approach across a collection of quality measures. Finally, we provide a non-private variant of PGB that improves the data quality of standard GAN training.


## Code for the post-GAN Boosting algorithm

`post_GAN_functions.R` and `post_GAN_functions.py` contain the implementation of our proposed approach.

## Hyperparameters

All of our hyperparameter choices are documented in the code and in the README files for each experiment.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine to replicate the experiments.

### Prerequisites

Data pre-processing and post-processing are done in R (and python via R). The GAN training is done directly in python. Therefore, an installation of [R](https://cran.r-project.org/src/base/R-3/) and python is needed. Optionally [RStudio](https://rstudio.com/products/rstudio/) can be used to run through the experiments interactively.

### Computing Infrastructure

The computing infrastructure we used is the following:

```
- OS:	Ubuntu 18.04.4 LTS

- CPU: Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz

- GPU: GeForce RTX 2070
		- CUDA: 10.0.130

		- CUDNN: 7.6.5
```

### R Environment

We used the following setup of R and RStudio:

```
R version 3.6.2 (2019-12-12)
Platform: x86_64-pc-linux-gnu (64-bit)
Running under: Ubuntu 18.04.4 LTS

Matrix products: default
BLAS:   /usr/lib/x86_64-linux-gnu/blas/libblas.so.3.7.1
LAPACK: /usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3.7.1

RStudio Server 1.3.776
```

All necessary R libraries will be installed and loaded upon running the `00-setup.R` in an experiment.

### Python Environment

This GAN code was written and tested using `python 3.7.7`. All packages in the environment are documented in `requirements.txt`. You can create a new python environment with `python 3.7.7`, activate the environment and run `pip install -r requirements.txt`. This will get you ready to go.

#### Installing tensorflow privacy
If installing `tensorflow privacy` does not work with `pip install -r requirements.txt`, follow the instructions [here](https://github.com/tensorflow/privacy) to install it manually.

## Running the Experiments

All experiments (toy-example, census-1940, census-2010 and titanic) follow the same structure.
The two most important folders for each experiment are `scripts/` and `raw-data/`. 

- `scripts/` contains all the code needed to run the experiments. It contains:
	- a script to setup the R environment,
	- a script to pre-process the raw data sets,
	- a script to run the GAN,
	- a script to post-process the GAN output,
	- and a script to summarize the results (e.g. in figures or tables as in the paper).
- `raw-data/` holds the raw unprocessed data sets for the respective experiment.

All other folders will be filled when executing the scripts. 

1. Set your working directory (for R and python respectively) to the folder of the experiment you want to run.
	- If you work with RStudio this will be automatically taken care of if you open the `.Rproj` file for the experiment.
2. Follow the order of the script files, and you should be able to replicate our results. 