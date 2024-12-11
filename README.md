# wge

**Authors:** Evan Zheran Liu\*, Kelvin Guu\*, Panupong (Ice) Pasupat\*, Tianlin Shi, Percy Liang (\* equal contribution) 

Source code accompanying our ICLR 2018 paper:  
[**Reinforcement Learning on Web Interfaces using Workflow-Guided Exploration**](https://arxiv.org/abs/1802.08802)  


## Purpose

The goal of this project is to train machine learning models (agents) to do
things in a browser that can be specified in natural language, e.g. "Book a
flight from San Francisco to New York for Dec 23rd."

## Setup

### General setup
- Python 
  - Install  virtualenv
  ```shell
  pip3 install virtualenv
  ```
  - Create a virtual environment named p5
  ```shell
  cd path/to/wge
  python3 -m venv p5
  ```
  - Activate the virtualenv
  ```shell  
  source p5/bin/activate
  ```
- Python dependencies
  ```
  pip install -r requirements.txt
  ```
  - If this gives you problems, try again and add pip's ```--ignore-installed```
  flag.

### Setup for recording
- Go to `miniwob-sandbox` folder and run the recording script from the terminal or command line. 

  ```
  cd miniwob-sandbox
  python record.py
  ```
- If everything works out, you should see the message: `Listening on http://localhost:8032/`.

#### Record your demonstration
- Open your browser and enter `http://localhost:8032/` in the address bar. You'll see an "Error: 404 Not Found" message, indicating the server is working correctly.
- Open the task environment:
  - Press Cmd+O (Mac) or Ctrl+O (Windows)
  - Navigate to `miniwob-sandbox/html/miniwob`
  - Select an environment file (choose from):
    - `click-checkboxes-soft.html`
    - `email-inbox-forward-nl.html`
    - `social-media.html`
- To begin recording, append `?record=true` to the URL in your address bar. For example:
`file:///path/to/wge/miniwob-sandbox/html/miniwob/social-media.html?record=true`
- Record 10 demonstrations for each environment (30 total recordings).

#### Viewing Recordings
- Ensure the recording server is still running.
- Open the viewer:
  - Press Cmd+O (Mac) or Ctrl+O (Windows)
  - Navigate to `miniwob-sandbox/viewer`
  - Select `viewer.html`
  - The address should look like: `file:///path/to/wge/miniwob-sandbox/viewer/viewer.html`
  - Your recordings will appear in the left panel.

### Data directory setup
- Download glove from https://nlp.stanford.edu/data/glove.6B.zip and place it in the `wge/data` directory after extraction
- Next update the following environment variables
  ```shell
  export REPO_DIR=/path/to/wge/
  export RL_DATA=/path/to/wge/data/
  export RL_DEMO_DIR=/path/to/miniwob-plusplus-master/miniwob/scripts/out/
  export MINIWOB_BASE_URL='http://localhost:8080/' 
  ```

### Demonstration directory setup
First, set the environment variable `$REPO_DIR` to point to the root of this Git repository. 
Next, run the following command to download MiniWoB++ Demonstrations to `./third-party/miniwob-demos`:
```
# Where $REPO_DIR is the path to the root of this Git repository.
git clone https://github.com/stanfordnlp/miniwob-plusplus-demos.git $REPO_DIR/third-party/miniwob-demos
export RL_DEMO_DIR=$REPO_DIR/third-party/miniwob-demos/

# To set envarionment variables in windows powershell use:
$env:REPO_DIR = "\path\to\wge"
$env:RL_DEMO_DIR = "\path\to\wge\third-party\miniwob-demos\"
```

### MiniWoB setup

- There are 2 ways to access MiniWoB tasks:
  1. **Run a simple server:** go to `miniwob-sandbox/html/` and run the supplied `http-serve`.
     - The tasks should now be accessible at `http://localhost:8080/miniwob/`
     - To use a different port (say 8765), run `http-serve 8765`, and then
     export the following to the `MINIWOB_BASE_URL` environment variable:
     ```
       export MINIWOB_BASE_URL='http://localhost:8765/'
     
       # To set envarionment variables in windows powershell use:
       $env:MINIWOB_BASE_URL = "http://localhost:8765/"
     ```
     
     **N.B. Windows user please use `http-serve.bat` file instead of `http-serve` to run the server.**
  
- Once you've followed one of the steps above, test `MiniWoBEnvironment` by running
  ```
  pytest wge/tests/miniwob/test_environment.py -s
  ```


## Launching an Experiment
Each time you open a new terminal to run an experiment, set these environment variables:

```shell
export RL_DATA=/path/to/data
export REPO_DIR=/path/to/wge/repository
export RL_DEMO_DIR=$REPO_DIR/third-party/miniwob-demos
export MINIWOB_BASE_URL='http://localhost:8080/' # replace with your port number
```
Also, the MiniWoB server must be running on the port specified in the `$MINIWOB_BASE_URL` environment variable. 
If the server isn't running, follow the "Run a simple server" instruction in the MiniWoB setup section above to 
start it.

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
