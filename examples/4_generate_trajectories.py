import numpy as np
from joblib import load
import matplotlib.pyplot as plt

TREE = 'Road_1,5_1,5_1_[0.2, 0.6, 0.2]_200leaves'
# TREE = 'LunarLander_SAC_[1, 1, 1]_450leaves'

"""
The source and destination specification is similar to how we specify counterfactuals, 
except now we're specifying limits over features rather than attributes.
"""
SOURCE = {'p':[-np.inf,1.5],'v':[-np.inf,0]} # For road.
DEST = {'p':[2.5,np.inf]} # For road.
# SOURCE = {'ang':[-np.inf,-0.5]} # For LunarLander.
# DEST = {'ang':[-0.2,0.2],'vel_ang':[-0.2,0.2]} # For LunarLander.

"""
Usually want to visualise over the features specified in the source and destination,
though this may not always be the case.
"""
VIS_FEATURES = ['p','v'] # For road.
# VIS_FEATURES = ['ang','vel_ang'] # For LunarLander.

NUM_trajectories_TO_SHOW = 1
SHOW_DERIVATIVES = True

# ---------------------------------------------------

# Load tree.
dt = load(f'trees/{TREE}.joblib')

# Find trajectories between the source and destination.
trajectories = dt.path_between(source_features=SOURCE, dest_features=DEST)
_, ax = dt.visualise(features=VIS_FEATURES)

# Plot trajectories in order of probability.
# *NOTE* The derivative alignment process is currently rather slow.
for i in range(NUM_trajectories_TO_SHOW):
    if i >= len(trajectories): break
    alpha = trajectories[i][1] / trajectories[0][1]
    dt.plot_transitions_2D(trajectories[i][0], features=['ang','vel_ang'], ax=ax,
                        align=True,
                        alignment_iterations=1e5,
                        lr=1e-2,
                        alpha=alpha 
                        )

# Optionally plot the leaf derivative vectors for comparison.
if SHOW_DERIVATIVES: dt.plot_leaf_derivatives_2D(features=VIS_FEATURES, ax=ax, lengthscale=7, alpha=0.1)

plt.show()