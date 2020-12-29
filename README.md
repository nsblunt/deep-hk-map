# Deep-HK-Map

This is a research code to investigate the use of deep learning to map between
various properties of lattice systems. In particular, the code can learn the
mapping from the local density to the wave function and other ground-state
properties. This mapping from the density to the wave function is the map known
to exist from the Hohenberg-Kohn theorem.

This code is in development. It currently uses the one-dimensional Hubbard
model and spinless Hubbard model. The spinless system has been studied using
similar approaches in the following papers:

Javier Robledo Moreno, Giuseppe Carleo, Antoine Georges,
Physical Review Letters 125, 076402 (2020)

M. Michael Denner, Mark H. Fischer, Titus Neupert,
Physical Review Research 2, 033388 (2020)

The code can generate its own training data, which is obtained by applying
random potentials to a given lattice model.

## Usage

The code uses PyTorch and a small number of other standard libraries. The
startup and input system uses Abseil. Flags can be specified on the command
line or in a config file, for example:

```
python3 deep_hk/main.py --flagfile=input.cfg

```
Example config files are given in the examples directory, and the available
flags can be viewed with
```
python3 deep_hk/main.py --help
```
Alternatively, a training script can be written directly, with an example
given in the examples/hubbard/ directoy.

## Jupyter notebook example

Also provided is a short tutorial as a Jupyter notebook, available in the
examples directory.
