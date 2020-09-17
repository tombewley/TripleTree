from joblib import load
import matplotlib.pyplot as plt

TREE = 'Road_1,5_0_1_[0.2, 0.6, 0.2]_200leaves'
# TREE = 'LunarLander_SAC_[1, 1, 1]_450leaves'

"""
Features for the Road environment are just position p and velocity v.

Features for LunarLander are:
    - pos_x: horizontal coordinate.
    - pos_y: vertical coordinate.
    - vel_x: horizontal velocity.
    - vel_y: vertical velocity.
    - ang: angular tilt.
    - vel_ang: angular velocity.
    - left_contact: 1 if left leg has contact, else 0.
    - right_contact: 1 if right leg has contact, else 0.

Choose 2 of these to visualise over.
"""
VIS_FEATURES = ['p','v'] # For road.
# VIS_FEATURES = ['pos_x','pos_y'] # for LunarLander.

"""
Visualisable attributes are:
    - action (*NOTE* Need to specify "action_0" or "action_1" for LunarLander).
    - value.
    - derivatives (*NOTE* Generates a quiver plot). 
    - action/value/derivative_impurity.
    - sample_density.

"""
VIS_ATTRIBUTE = 'action' # For Road.
# VIS_ATTRIBUTE = 'action_0' # For LunarLander.

# ---------------------------------------------------

# Load tree.
dt = load(f'trees/{TREE}.joblib')

# Visualise.
if VIS_ATTRIBUTE == 'derivatives':
    _, ax = dt.visualise(features=VIS_FEATURES, attributes=[None])
    dt.plot_leaf_derivatives_2D(features=VIS_FEATURES, ax=ax, lengthscale=7)
else:
    dt.visualise(features=VIS_FEATURES, attributes=VIS_ATTRIBUTE)
plt.show()