# Private post-GAN Boosting

## Titanic

This folder contains all data and code to replicate the findings and figures presented in Part 4.3 of Private post-GAN Boosting

## Contents

- figures: 		.png files of figures presented in paper
- gan-input:  .csv files for the GAN input
- models:     folder to store the intermediate weights of GAN training
- raw-data: 	original raw data sets (or instructions on how/where to get them)
- scripts: 		R and python code to replicate the analysis
- synthetic-output: folder to store the synthetic data generated by the four approaches in the paper
- titanic.Rproj: Rproject for the toy-example

## Hyperparameters

### GAN Training
- Number of hidden layers: 3
	- Units per layer: (256 - 128 - 128)
- minibatch size: 64
- Discriminator learning rate
	- dp Discriminator: 0.009
		- l2_norm_clip: 10
		- noise_multiplier: 0.01
	- non-dp Discriminator: 0.001
- Generator learning rate: 0.001
- epochs: 200
- Dimension of Z: 64
- temperature for gumbel-softmax trick: 0.00001

### post-GAN Boosting
- number of models (Generators and Discriminators): 100
- save models after every 5th update steps
- number of samples per model: 5,000
- number of steps T: 400
- epsilon for post-GAN-Boosting (MW_epsilon): 0.453