import sys
sys.path.append('..')
import tripletree as tt
import pandas as pd
from joblib import dump

DATASET = 'Road_1,5_0_1'
# DATASET = 'LunarLander_SAC'

MAX_NUM_LEAVES = 200
# MAX_NUM_LEAVES = 450
THETA = [0.2, 0.6, 0.2] 
# THETA = [1, 1, 1]
GAMMA = 0.99

# ---------------------------------------------------

# Load data.
df = pd.read_csv(f'data/{DATASET}.csv', index_col=0)
if 'Road' in DATASET:
    feature_names = list(df.columns[:2])
    action_names = 'a'
elif 'LunarLander' in DATASET:
    feature_names = list(df.columns[:8])
    action_names = ['main_engine','lr_engine']
o = df[feature_names].values
a = df[action_names].values
r = df['r'].values
p = df['prev'].values
n = df['next'].values

# Initialise tree.
dt = tt.model(classifier=False, action_names=action_names, feature_names=feature_names, 
              scale_actions_by=[1,1], gamma=GAMMA)

# Grow best-first.
dt.grow(o, a, r, p, n, impurity_weights=THETA, max_num_leaves=MAX_NUM_LEAVES)
dt.compute_paths_matrix(conditional=True) # Required for path generation.

# Evaluate predictive losses and save tree.
action_loss, value_loss, derivative_loss = dt.score(dt.o, a=dt.a, g=dt.g, d_norm=dt.d_norm)
print(f'Losses: action = {action_loss}, value = {value_loss}, derivatives = {derivative_loss}')
dump(dt, f'trees/{DATASET}_{THETA}_{dt.num_leaves}leaves.joblib')