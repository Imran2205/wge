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

#### View your Recordings
- Ensure the recording server is still running.
- Open the viewer:
  - Press Cmd+O (Mac) or Ctrl+O (Windows)
  - Navigate to `miniwob-sandbox/viewer`
  - Select `viewer.html`
  - The address should look like: `file:///path/to/wge/miniwob-sandbox/viewer/viewer.html`
  - Your recordings will appear in the left panel.


### Setup for `data` directory 
- Download `glove` from `https://nlp.stanford.edu/data/glove.6B.zip` and place it in the `wge/data` directory after extraction


### Setup for model training
- Run miniwob server: go to `path/to/wge/miniwob-sandbox/html/` and run the supplied `http-serve`.
     - For Mac, use
     ```
       cd path/to/wge/miniwob-sandbox/html/
       ./http-serve
     ```
    - For Windows, use
    ```
      cd path\to\wge\miniwob-sandbox\html\
      .\http-serve.bat
    ```
- The server should be running at `http://localhost:8080/`
- Next, set the environment variables:
  - For Mac, each time you open a new terminal to run an experiment, set these environment variables:
  ```shell
    export REPO_DIR=/path/to/wge/
    export RL_DATA=/path/to/wge/data/
    export RL_DEMO_DIR=/path/to/wge/path/to/miniwob-sandbox/out/
    export MINIWOB_BASE_URL='http://localhost:8080/' 
  ```
  - For Windows, each time you open a new command line to run an experiment, set these environment variables:
  ```shell
    $env:REPO_DIR="\path\to\wge\"
    $env:RL_DATA="\path\to\wge\data\"
    $env:RL_DEMO_DIR="\path\to\wge\path\to\miniwob-sandbox\out\"
    $env:MINIWOB_BASE_URL="http:\\localhost:8080\" 
  ```  
  
- Once you've followed the above steps, test `MiniWoBEnvironment` by running
  ```
  cd /path/to/wge/
  pytest wge/tests/miniwob/test_environment.py -s
  ```

## Train a model
- To train a model on a task, say 'email-inbox-forward-nl', run:
```
python main.py configs/default-base.txt --task email-inbox-forward-nl
```
- Change the task name (the last parameter) to train for other tasks.


## Experiment management
- All training runs are managed by the `MiniWoBTrainingRuns` object.
- The most important methods on `MiniWobTrainingRun` are:
  - `__init__`: the policy, the environment, demonstrations, etc, are all loaded here.
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
