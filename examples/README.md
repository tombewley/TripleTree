## Example Scripts

Example scripts to reproduce some of the results from our original paper.

### In This Folder

- `data/`: Training datasets from the 2D road environment (4x dynamic programming policies) and LunarLanderContinuous-v2 (1x SAC deep RL agent).
- `trees/`: A pre-grown TripleTree for each of the datasets.
- {`1_grow.py`, `2_visualise.py`, `3_explain_counterfactual.py`, `4_generate_paths.py`}: Example scripts.

### Usage

- Run `1_grow.py` to grow a TripleTree model from scratch.
- Run `2_visualise.py` to try out the hyperrectangle visualisation method. For LunarLander, this creates projected visualisations onto 2 of the 8 features.
- Run `3_explain_counterfactual.py` to generate counterfactual explanations for randomly-sampled data points and a manually-specified foil condition.
- Run `4_generate_trajectories.py` to generate simulated trajectories between source and destination regions of the state space.



