# online_adaptation

This is a template for a Python Machine Learning project with the following features:

* [Weights and Biases](wandb.ai) support, for experiment tracking and visualization
* [Hydra](https://hydra.cc/) support, for configuration management
* [Pytorch Lightning](https://www.pytorchlightning.ai/) support, for training and logging

In addition, it contains all the good features from the original version of this repository (and is a proper Python package):

* Installable via `pip install`. Anyone can point directly to this Github repository and install your project, either as a regular dependency or as an editable one.
* Uses the new [PEP 518, officially-recommended pyproject.toml](https://pip.pypa.io/en/stable/reference/build-system/pyproject-toml/) structure for defining project structure and dependencies (instead of requirements.txt)
* Nice, static documentation website support, using mkdocs-material. Structure can be found in `docs/`
* `black` support by default, which is an opinionated code formatting tool
* `pytest` support, which will automatically run tests found in the `tests/` directory
* `mypy` support, for optional typechecking of type hints
* `pre-commit` support, which runs various formatting modifiers on commit to clean up your dirty dirty code automatically.
* Github Actions support, which runs the following:
    * On a Pull Request: install dependencies, run style checks, run Python tests
    * After merge: same a Pull Request, but also deploy the docs site to the projects Github Pages URL!!!!

All that needs doing is replacing all occurances of `online_adaptation` and `online-adaptation` with the name of your package(including the folder `src/online_adaptation`), the rest should work out of the box!

## Installation

Starting anew:

```bash
# Create a virtualenvironment.
pyenv virtualenv 3.8.18 oa

# Activate it.
pyenv activate oa

```
First, we'll need to install platform-specific dependencies for Pytorch. See [here](https://pytorch.org/get-started/locally/) for more details. For example, if we want to use CUDA 11.8 with Pytorch 2.



```bash

# These are required for Flowbot3d.

pip install pytorch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117/

pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

pip install "flowbot3d @ git+https://github.com/r-pad/flowbot3d.git"

```

Then, we can install the package itself:

```bash

pip install -e ".[develop,notebook]"

```

Then we install pre-commit hooks:

```bash

pre-commit install

```

## Docker

To build the docker image, run:

```bash
docker build -t <my_dockerhub_username>/online-adaptation .
```

To run the training script locally, run:

```bash
WANDB_API_KEY=<API_KEY>
# Optional: mount current directory to run / test new code.
# Mount data directory to access data.
docker run \
    -v $(pwd)/data:/opt/baeisner/data \
    -v $(pwd)/logs:/opt/baeisner/logs \
    --gpus all \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e WANDB_DOCKER_IMAGE=online-adaptation \
    online-adaptation python scripts/train.py \
        dataset.data_dir=/root/data \
        log_dir=/root/logs
```

To push this:

```bash
docker push <my_dockerhub_username>/online-adaptation:latest
```

## Running on Clusters

* [Autobot](autobot.md)

## Training

To train a model, run:

```bash

# Flowbot
python scripts/train.py
```

## Evaluation

To evaluate a model, run:

```bash
# Flowbot
python scripts/eval.py wandb.group=Null checkpoint.run_id=<run-id>
```
