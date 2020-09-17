import numpy as np
from joblib import load
import matplotlib.pyplot as plt

TREE = 'Road_1,5_0_1_[0.2, 0.6, 0.2]_200leaves'
# TREE = 'LunarLander_SAC_[1, 1, 1]_450leaves'

"""
The foil specification dictates which attribute to use in the counterfactual, 
and the range of values to look for in foil leaves.
Read as {"attribute": ["lower limit", "upper limit"]}
"""
# FOIL_SPEC = {'action':[-np.inf,0.5]} # For road (actions are either 0 or 1).
FOIL_SPEC = {'value':[-np.inf,0.5]} # For road.
# FOIL_SPEC = {'action_0':[0.9,np.inf]} # For LunarLander.
# FOIL_SPEC = {'action_1':[-np.inf,-0.9]} # For LunarLander.
# FOIL_SPEC = {'value':[-np.inf,0]} # For LunarLander.

# ---------------------------------------------------

# Load tree.
dt = load(f'trees/{TREE}.joblib')

# Pick a random sample from the training dataset and predict its action.
o = dt.o[np.random.randint(len(dt.o))]
attr = list(FOIL_SPEC.keys())[0]
print()
print(f"Factual state = {o}")
print(f"Factual prediction = {dt.predict(o, attributes=attr.split('_')[0])}")
print()

# Find the minimal foil.
_, o_foil, L0, L2 = dt.counterfactual(o, foil=FOIL_SPEC)
if L0 == None: raise Exception("No leaves satisfy foil condition! Try modifying it.")
print(f"Foil state = {o_foil}")
print(f"Foil prediction = {dt.predict(o_foil, attributes=attr.split('_')[0])}")
print()

print(f"L0 norm = {int(L0)}, L2 norm = {L2}")

if L0 in (1,2): 
    # Display the counterfactual on a slice visualisation.
    dt.display_counterfactual(o, o_foil, attributes=attr)
    plt.show()
elif L0 == 0:
    print("Sample already satisfies foil condition! Try running again.")
else:
    print("> 2 features changed so cannot visualise on a plane! Try running again.")