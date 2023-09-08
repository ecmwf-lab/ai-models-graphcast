# ai-models-graphcast

`ai-models-graphcast` is an [ai-models](https://github.com/ecmwf-lab/ai-models) plugin to run Google Deepmind's [GraphCast](https://github.com/deepmind/graphcast).


## Installation

To install the package, run:

```bash
pip install ai-models-graphcast
```

This will install the package and most of its dependencies.

Then to install graphcast dependencies (and Jax on GPU):



### Graphcast and Jax

Graphcast depends on Jax, which needs special installation instructions for your specific hardware.

Please see the [installation guide](https://github.com/google/jax#installation) to follow the correct instructions.

We have prepared two `requirements.txt` you can use. A CPU and a GPU version:

For the preferred GPU usage:
```
pip install -r requirements-gpu.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

For the slower CPU usage:
```
pip install -r requirements.txt
```
