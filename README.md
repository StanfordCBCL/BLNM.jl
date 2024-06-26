# BLNM.jl

*Julia implementation of Branched Latent Neural Maps (BLNMs), a computational tool for generic functional mapping of physical processes to build accurate and efficient surrogate models.*

## Mathematical details

BLNMs structurally disentangle inputs with different intrinsic roles, such as time and model parameters, by means of feedforward partially-connected neural networks. These partial connections can be propagated from the first hidden layer throughout the outputs according to the chosen disentanglement level. Furthermore, BLNMs may be endowed with latent variables in the output space, which enhance the learned dynamics of the neural map. A BLNM is defined by a simple, lightweight architecture, easy and fast to train, while effectively reproducing physical processes with sharp gradients and fast dynamics in complex solution manifolds. It breaks the curse of dimensionality by requiring small training datasets and do not degrade in accuracy when tested on a different discretization than the one used for training.
For the case of inputs given by time and model parameters coming, for instance, from a differential equation, BLNMs can be defined as follows:

$\mathbf{z}(t) = \mathcal{B L \kern0.05em N \kern-0.05em M} \left(t, \boldsymbol{\theta}; \mathbf{w} \right) \text{ for } t \in [0, T].$

This partially-connected neural network is represented by weights and biases $\mathbf{w} \in \mathbb{R}^{N_\mathrm{w}}$, and introduces a map $\mathcal{B L \kern0.05em N \kern-0.05em M} \colon \mathbb{R}^{1 + N_\mathcal{P}} \to \mathbb{R}^{N_\mathrm{z}}$ from time $t$ and model parameters $\boldsymbol{\theta} \in \boldsymbol{\Theta} \subset \mathbb{R}^{N_\mathcal{P}}$ to a state vector $\mathbf{z}(t) = [\mathbf{z}_ \mathrm{physical}(t), \mathbf{z}_ \mathrm{latent}(t)]^T$.
The state vector $\mathrm{z}(t) \in \mathbb{R}^{N_\mathrm{z}}$ contains $\mathbf{z}_ \mathrm{physical}(t)$ physical fields of interest, as well as interpretable $\mathbf{z}_\mathrm{latent}(t)$ latent temporal variables without a direct physical representation, that enhance the learned dynamics of the BLNM.
During the optimization process of the neural network tunable parameters, the Mean Square Error (MSE) between the BLNM outputs and observations, both in non-dimensional form, is minimized.
Time and model parameters are also normalized during the training phase of the BLNM.

This package can be seamlessly extended to include more than two branches involving different sets of inputs, such as space, time, model-specific parameters and geometrical features.

## Getting started

Make sure to activate the environment contained in `Project.toml` with `julia 1.8.5` installed.

The repository currently contains three different test cases:
* Create a geometry-specific surrogate model to reproduce in silico 12-lead electrocardiograms (ECGs) while spanning 7 cell-to-organ level model parameters (`train_HLHS_ECGs.jl`, `train_hyperopt_HLHS_ECGs.jl`, `test_HLHS_ECGs.jl`)
* Digital twinning of cardiac electrophysiology to match clinical 12-lead ECGs (`digital_twin_HLHS_ECGs.jl`)
* Create a cell-based surrogate model that simulates the action potential of the [Tomek Rodriguez-O'Hara Rudy (ToR-ORd) ionic model](https://elifesciences.org/articles/48890) at limit cycle while spanning 14 relevant cellular conductances (`train_ToR-ORd.jl`, `test_ToR-ORd.jl`)

For the sake of illustration on the first test case, a BLNM with user-defined parameters can be trained by running `train_HLHS_ECGs.jl` within the Julia REPL.
Once the BLNM is trained, the mean square error on the training and testing datasets can be computed and some plots can be shown by calling `test_HLHS_ECGs.jl`.
Running `import Pkg; Pkg.instantiate()` is usually necessary to properly initialize the environment within the Julia REPL.
MPI-based hyperparameters tuning with K-fold cross-validation should be run in a bash terminal by typing:
```julia
mpirun -n K julia --project=. train_hyperopt_HLHS_ECGs.jl
```
where K represents the number of K-folds.
Note that a suitable MPI implementation should be installed in order to run the last command.

`BLNM.jl` defines the actual BLNMs implementation, `InOut.jl` contains input-output features and `Utils.jl` introduces some generic utilities.

## References
```bibtex
@article{Salvador2024BLNM,
  title={Branched Latent Neural Maps},
  author={Salvador, M. and Marsden, A. L.},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={418},
  pages={116499},
  year={2024}
}
```
```bibtex
@article{Salvador2024DT,
  title={Digital twinning of cardiac electrophysiology for congenital heart disease},
  author={Salvador, M. and Kong, F. and Peirlinck, M. and Parker, D. and Chubb, H. and Dubin, A. and Marsden, A. L.},
  journal={Journal of the Royal Society Interface},
  year={2024}
}
```
## License
`BLNM.jl` is released under the MIT license.
