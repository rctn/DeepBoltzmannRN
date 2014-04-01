DeepBoltzmannRN
===============

Deep Boltzmann Machines in R^N dimensions

Readings

1. Restricted Boltzmann Machines
http://www.deeplearning.net/tutorial/rbm.html
2. Training DBMs using CD
http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS09_SalakhutdinovH.pdf
3. "Efficient" Training DBMs
http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2010_SalakhutdinovL10.pdf
4. Training DBNs
http://www.mitpressjournals.org/doi/abs/10.1162/neco.2006.18.7.1527#.UzsF8PhLt-g
5. Minimum Probability Flow
http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.107.220601

Training Ideas

* Pretrain model as described in 2 using stacked RBMs (with MPF?)
* Train model using methods described in 2 or 3.
* Train models using MPF -- how to deal with latent variables?