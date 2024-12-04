# wge

**Authors:** Evan Zheran Liu\*, Kelvin Guu\*, Panupong (Ice) Pasupat\*, Tianlin Shi, Percy Liang (\* equal contribution) 

Source code accompanying our ICLR 2018 paper:  
[**Reinforcement Learning on Web Interfaces using Workflow-Guided Exploration**](https://arxiv.org/abs/1802.08802)  

Reproducible experiments using this code are located on [our Codalab
worksheet](https://worksheets.codalab.org/worksheets/0x0f25031bd42f4aabbc17625fe1484066/).

## Purpose

The goal of this project is to train machine learning models (agents) to do
things in a browser that can be specified in natural language, e.g. "Book a
flight from San Francisco to New York for Dec 23rd."

## Setup

### General setup
- Python 
  - Create a conda environment with Python v3.8
  ```shell
  conda create -n wge python=3.8
  ```
  - Activate the conda environment
  ```shell
  conda activate wge
  ```
- Python dependencies
  ```
  pip install -r requirements.txt
  ```
  - If this gives you problems, try again and add pip's ```--ignore-installed```
  flag.

- Node and npm
  - Make sure Node and npm are installed. If they 
  are, ```node -v``` and ```npm -v``` should print version numbers.
  - To install Node and npm in macOS use
    ```shell
    brew install node
    ```
    - If brew is not installed, please follow the installation instructions on [brew website](https://brew.sh/).
  - To install Node and npm on Linux machines use
    ```shell
    sudo apt install nodejs
    sudo apt install npm
    ```
  - On Windows, Node and npm can be installed via the prebuilt installer from the [nodejs website](https://nodejs.org/en/download/prebuilt-installer)
- PyTorch
  - Install [PyTorch v1.13.1](https://pytorch.org/get-started/previous-versions/#v1131). Depending on OS type, use any of the following commands.
    - macOS
    ```shell
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch
    ```
    - Linux and Windows
    ```
    # CUDA 11.6
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
    # CUDA 11.7
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
    # CPU Only
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch
    ```

- Selenium
  - Install selenium from `conda-forge`
  ```
  conda install conda-forge::selenium
  ```
  - Alternatively, you can install selenium using `pip`
  ```
  pip install selenium
  ```

### Data directory setup

- This code depends on the environmental variable ```$RL_DATA``` being set,
  pointing to a configured data directory.

- Create a data directory ```mkdir -p /path/to/data``` and set ```export
  $RL_DATA=/path/to/data```. In order for the code to run, ```$RL_DATA```
  will need to be set to point at this directory.

- Next, set up the data directory:
  ```
  cd $RL_DATA
  # Download glove from https://nlp.stanford.edu/data/glove.6B.zip and place
  # in current directory however you want
  # Suggested: wget https://nlp.stanford.edu/data/glove.6B.zip
  # If wget is not installed, you can install it on macOS using brew install wget.
  unzip -d glove glove.6B.zip
  ```

### Demonstration directory setup
First, set the environment variable `$REPO_DIR` to point to the root of this Git repository.
```
# Where $REPO_DIR is the path to the root of this Git repository.
git clone https://github.com/stanfordnlp/miniwob-plusplus-demos.git $REPO_DIR/third-party/miniwob-demos
export RL_DEMO_DIR=$REPO_DIR/third-party/miniwob-demos/
```

### MiniWoB setup

- There are 2 ways to access MiniWoB tasks:
  1. **Run a simple server:** go to `miniwob-sandbox/html/` and run the supplied `http-serve`.
    - The tasks should now be accessible at `http://localhost:8080/miniwob/`
    - To use a different port (say 8765), run `http-serve 8765`, and then
    export the following to the `MINIWOB_BASE_URL` environment variable:
    ```
    export MINIWOB_BASE_URL='http://localhost:8765/'
    ```
  2. **Use the `file://` protocol:**
    Open `miniwob-sandbox/html/` in the browser,
    and then export the URL to the `MINIWOB_BASE_URL` environment variable:
    ```
    export MINIWOB_BASE_URL='file:///path/to/miniwob-sandbox/html/'
    ```
- Once you've followed one of the steps above, test `MiniWoBEnvironment` by running
  ```
  pytest wge/tests/miniwob/test_environment.py -s
  ```

### MiniWoB versions of FormWoB

Follow the "Run a simple server" instruction in the MiniWoB setup section above.

## Launching an Experiment

To train a model on a task, run:
```
python main.py configs/default-base.txt --task click-tab-2
```
- This executes the main entrypoint script, `main.py`. In particular, we pass it a base HOCON format config file and the task click-tab-2.
- Additional configs can be merged in by passing them as commandline arguments
  from configs/config-mixins
- Make sure that the following environmental variables are set:
  `MINIWOB_BASE_URL`, `RL_DEMO_DIR`, `REPO_DIR`.
- You may also want to set the `PYTHONPATH` to the same place as `REPO_DIR` to
  make imports work out properly
- You can also run this via docker by first running `python run_docker.py` to
  launch Docker and then running the above command. Unfortunately, you will
not be able to see the model train in the Docker container.
- The different tasks can be found in the subdirectories of
  third-party/miniwob-sandbox/html

If the script is working, you should see several Chrome windows pop up 
(operated by Selenium) and a training progress bar in the terminal.

## Experiment management

All training runs are managed by the `MiniWoBTrainingRuns` object. For example,
to get training run #141, do this:
```python
runs = MiniWoBTrainingRuns()
run = runs[141]  # a MiniWoBTrainingRun object
```

A `TrainingRun` is responsible for constructing a model, training it, saving it
and reloading it (see superclasses `gtd.ml.TrainingRun` and
`gtd.ml.TorchTrainingRun` for details.)

The most important methods on `MiniWobTrainingRun` are:
- `__init__`: the policy, the environment, demonstrations, etc, are all loaded
here.
- `train`: actual training of the policy happens here

## Model architecture

During training, there are several key systems involved:
- the environment
- policies
  - the model policy
  - the exploration policy
- episode generators
  - basic episode generator
  - best first episode generator
- the replay buffer

### Environment

All environments implement the `Environment` interface. A policy interacts
with the environment by calling the environment's `step` method and passing in
actions.

Note that an environment object is _batched_. It actually represents a batch
of environments, each running in parallel (so that we can train faster).

We mostly use `MiniWoBEnvironment` and `FormWoBEnvironment`.

### Policies

See the `Policy` interface. The most important methods are `act`,
`update_from_episodes` and `update_from_replay_buffer`.

Note that all of these methods are also batched (i.e. they operate on multiple
episodes in parallel)

The model policy is the main one that we are trying to train. See
`MiniWoBPolicy` as an example.

### Episode generators

See the `EpisodeGenerator` interface. An `EpisodeGenerator` runs a
`Policy` on an `Environment` to produce an `Episode`.

### Replay buffer

See the `ReplayBuffer` interface. A `ReplayBuffer` stores episodes produced
by the exploration policy. The final model policy is trained off episodes
sampled from the replay buffer.

## Configuration

All configs are in the `configs` folder. They are specified in HOCON format.
The arguments to `main.py` should be a list of paths to config files.
`main.py` then merges these config files according to the
[rules explained here](https://github.com/typesafehub/config/blob/master/HOCON.md#include-semantics-merging).
