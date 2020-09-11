"""
TripleTree is a much-extended variant of the CART algorithm, 
specialised for the task of jointly modelling the policy, value function
and temporal dynamics of black box agents in Markov decision processes (MDPs). 

Introduced in "TripleTree: A Versatile Interpretable Representation of Black Box Agents and their Environments".

Code by Tom Bewley, University of Bristol.
tom.bewley@bristol.ac.uk
http://tombewley.com
"""

from .utils import *
import numpy as np
import math
from itertools import chain, combinations
import pandas as pd
pd.options.mode.chained_assignment = None
from scipy.spatial import minkowski_distance
from tqdm import tqdm
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class TripleTree:
    def __init__(self,
                 classifier,                     # Whether actions are discrete or continuous.
                 action_names,                   # Assign alphanumeric names to action dimensions (or just one name if univariate).
                 feature_names = None,           # Assign alphanumeric names to features.
                 scale_features_by = 'range',    # Method used to determine feature scaling, or pre-computed vector of scales.
                 scale_derivatives_by = 'std',   # Method used to determine derivative scaling. NOTE: 'same' = use same scales as features.
                 scale_actions_by = 'range',     # Method used to determine scaling (multivariate regression).
                 pairwise_action_loss = {},      # (For classification) badness of each pairwise action error. If unspecified, assume uniform.
                 gamma = 1                       # Discount factor to use when considering reward.
                 ):   
        self.classifier = classifier
        if type(action_names) == str: self.action_names = [action_names]
        else: self.action_names = action_names
        self.action_dim = len(self.action_names)
        if feature_names: self.feature_names = feature_names
        else:             self.feature_names = np.arange(self.num_features).astype(str) # Use numerical indices if not provided.
        self.num_features = len(feature_names)
        self.pairwise_action_loss = pairwise_action_loss
        self.gamma = gamma

        # Initial setup for feature scaling.
        if type(scale_features_by) != str:
            assert len(scale_features_by) == self.num_features
            self.scale_features_by = 'given'
            self.feature_scales = np.array(scale_features_by) 
        else: 
            self.scale_features_by = scale_features_by
            self.feature_scales = []
        self.scale_derivatives_by = scale_derivatives_by
        self.derivative_scales = []
        if not self.classifier:
            if type(scale_actions_by) != str: # This is only used for multivariate regression.
                self.scale_actions_by = 'given'
                self.action_scales = np.array(scale_actions_by) 
            else: 
                self.scale_actions_by = scale_actions_by
                self.action_scales = []


# ===================================================================================================================
# COMPLETE GROWTH ALGORITHMS.

    """
    TODO: min_samples_split and min_weight_fraction_split not used!
    TODO: Combine grow() methods to prevent duplication.
    TODO: Other algorithms: 
        - Could also define 'best' as highest impurity gain, but this requires us to try splitting every node first!
        - Best first with active sampling (a la TREPAN).
        - Restructuring. 
    """
    
    def grow(self, 
             o, a, r=[], p=[], n=[], w=[],           # Dataset.
             split_by =                  'weighted', # Attribute to split by: action, value or both.
             impurity_weights =          [1.,1.,1.], # Weight of each impurity component (if by='weighted').
             max_num_leaves =            np.inf,     #
             min_samples_split =         2,          #
             min_weight_fraction_split = 0,          # Min weight fraction at a node to consider splitting.
             min_samples_leaf =          1,          # Min samples at a leaf to accept split.
             min_split_quality =         0,          # Min relative impurity gain to accept split.
             stochastic_splits =         False,      # Whether to samples splits proportional to impurity gain. Otherwise deterministic argmax.
             ):
        """
        Accept a complete dataset and grow a complete tree best-first.
        This is done by selecting the leaf with highest impurity_sum.
        """
        assert split_by in ('action','value','derivative','pick','weighted')
        if split_by in ('value','pick','weighted'): assert r != [], 'Need reward information to split by value.'
        if split_by in ('derivative','pick','weighted'): assert n != [], 'Need successor information to split by derivatives.'
        self.split_by = split_by
        self.impurity_weights = np.array(impurity_weights).astype(float)  
        self.max_num_leaves = max_num_leaves
        self.min_samples_split = min_samples_split 
        self.min_weight_fraction_split = min_weight_fraction_split
        self.min_samples_leaf = min_samples_leaf
        self.stochastic_splits = stochastic_splits
        self.min_split_quality = min_split_quality
        self.load_data(o, a, r, p, n, w)
        self.seed()
        # The root impurities are used to normalise those at each leaf.
        self.root_impurities = np.array([self.tree.action_impurity, self.tree.value_impurity, self.tree.derivative_impurity])
        self.root_impurity_sums = self.get_node_impurity_sums(self.tree)
        self.leaf_impurity_sums = [self.root_impurity_sums]
        self.untried_leaf_nints = [1]
        with tqdm(desc='Best-first growth', total=self.max_num_leaves, initial=1) as pbar:
            pbar.update(1)
            while self.num_leaves < self.max_num_leaves and len(self.untried_leaf_nints) > 0:
                self.split_next_best(pbar)
        # List all the leaf integers.
        self.leaf_nints = self.get_leaf_nints()
        # Compute leaf transition probabilities, both marginal and conditional.
        self.compute_all_leaf_transition_probs()
                

    def grow_depth_first(self, 
                         o, a, r=[], p=[], n=[], w=[],           # Dataset.
                         split_by =                  'weighted', # Attribute to split by: action, value or both.
                         impurity_weights =          [1.,1.,1.], # Weight of each impurity component (if by='weighted').
                         max_depth =                 np.inf,     # Depth at which to stop splitting.  
                         min_samples_split =         2,          # Min samples at a node to consider splitting. 
                         min_weight_fraction_split = 0,          # Min weight fraction at a node to consider splitting.
                         min_samples_leaf =          1,          # Min samples at a leaf to accept split.
                         min_split_quality =         0,          # Min relative impurity gain to accept split.
                         stochastic_splits =         False,      # Whether to samples splits proportional to impurity gain. Otherwise deterministic argmax.
                         ):
        """
        Accept a complete dataset and grow a complete tree depth-first as in CART.
        """
        assert split_by in ('action','value','derivative','pick','weighted')
        if split_by in ('value','pick','weighted'): assert r != [], 'Need reward information to split by value.'
        if split_by in ('derivative','pick','weighted'): assert n != [], 'Need successor information to split by derivatives.'
        self.split_by = split_by
        self.impurity_weights = np.array(impurity_weights).astype(float)   
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split 
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_split = min_weight_fraction_split
        self.stochastic_splits = stochastic_splits
        self.min_split_quality = min_split_quality
        self.load_data(o, a, r, p, n, w)
        self.seed()
        def recurse(node, depth):
            if depth < self.max_depth and self.split(node):
                recurse(node.left, depth+1)
                recurse(node.right, depth+1)
        print('Growing depth first...')
        recurse(self.tree, 0)
        # List all the leaf integers.
        self.leaf_nints = self.get_leaf_nints()
        # Compute leaf transition probabilities, both marginal and conditional.
        self.compute_all_leaf_transition_probs()


# ===================================================================================================================
# METHODS FOR GROWTH.


    def load_data(self, o, a, r=[], p=[], n=[], w=[], append=False):
        """
        Complete training data has all of the following components, but only the first two are essential.
        *o = observations.
        *a = actions.
        r = rewards.
        p = index of preceding sample (-1 = start of episode).
        n = index of successor sample (-1 = end of episode).
        w = sample weight.
        """
        if append: raise Exception('Dataset appending not yet implemented.')

        # Store basic values.
        self.o, self.r, self.p, self.n = o, r, p, n
        self.num_samples, n_f = self.o.shape
        assert n_f == self.num_features, 'Observation size does not match feature_names.'
        assert self.num_samples == len(a) == len(r) == len(p) == len(n), 'All inputs must be the same length.'
        assert min(p) == min(n) >= -1, 'Episode start/end must be denoted by index of -1.'
        if w == []: self.w = np.ones(self.num_samples)
        else: self.w = w
        self.w_sum = self.w.sum()
        if self.classifier:
            # Define a canonical (alphanumeric) order for the action classes so can work in terms of indices.
            self.action_classes = sorted(list(set(a) | set(self.pairwise_action_loss)))
            self.num_actions = len(self.action_classes)
            self.action_loss_to_matrix() 
            self.a = np.array([self.action_classes.index(c) for c in a]) # Convert array into indices.
        else:
            self.a = a
            a_shp = a.shape
            # Deal with multivariate actions.
            if len(a_shp) == 2:
                assert a_shp[1] == self.action_dim
                if self.action_dim == 1: self.a = self.a.flatten()
            else: 
                assert len(a_shp) == 1  
                assert self.action_dim == 1    

        # Compute return for each sample.
        self.g = self.get_returns(self.r, self.p, self.n)
        # Compute time derivatives of features for each sample.
        self.d = self.get_derivatives(self.o, self.p, self.n)

        # Placeholder for storing the leaf at which each sample resides.
        # NOTE: Additional zero at the end handles episode termination cases.
        self.nint = np.zeros(self.num_samples+1).astype(int)
        # Placeholder for storing the next leaf after each sample.
        self.next_nint = np.zeros(self.num_samples).astype(int)

        # Set up min samples for split / leaf if these were specified as ratios.
        if type(self.min_samples_split) == float: self.min_samples_split_abs = int(np.ceil(self.min_samples_split * self.num_samples))
        else: self.min_samples_split_abs = self.min_samples_split
        if type(self.min_samples_leaf) == float: self.min_samples_leaf_abs = int(np.ceil(self.min_samples_leaf * self.num_samples))
        else: self.min_samples_leaf_abs = self.min_samples_leaf

        # Automatically create scaling vectors if not provided already.
        self.global_feature_lims = np.vstack((np.min(self.o, axis=0), np.max(self.o, axis=0))).T
        if self.feature_scales == []:
            if self.scale_features_by == 'range': 
                ranges = self.global_feature_lims[:,1] - self.global_feature_lims[:,0]
            elif self.scale_features_by == 'percentiles':
                ranges = (np.percentile(self.o, 95, axis=0) - np.percentile(self.o, 5, axis=0))
            else: raise ValueError('Invalid feature scaling method.')
            self.feature_scales = max(ranges) / ranges
        
        if self.derivative_scales == []:
            if self.scale_derivatives_by == 'same':
                self.derivative_scales = self.feature_scales
            else:
                if self.scale_derivatives_by == 'range':
                    ranges = np.nanmax(self.d, axis=0) - np.nanmin(self.d, axis=0)
                elif self.scale_derivatives_by == 'std':
                    ranges = np.nanstd(self.d, axis=0)
                self.derivative_scales = max(ranges) / ranges
        # Normalise feature derivatives. These are used for impurity calculations.
        self.d_norm = self.d * self.derivative_scales 

        if not self.classifier and self.action_dim > 1:
            if self.action_scales == []:
                self.global_action_lims = np.vstack((np.min(self.a, axis=0), np.max(self.a, axis=0))).T
                if self.scale_actions_by == 'range': 
                    ranges = self.global_action_lims[:,1] - self.global_action_lims[:,0]
                elif self.scale_actions_by == 'percentiles':
                    ranges = (np.percentile(self.a, 95, axis=0) - np.percentile(self.a, 5, axis=0))
                else: raise ValueError('Invalid action scaling method.')
                self.action_scales = max(ranges) / ranges
            # Normalise action dimensions. These are used for impurity calculations.
            self.a_norm = self.a * self.action_scales
                    

    def action_loss_to_matrix(self):
        """Convert dictionary representation to a matrix for use in action_impurity calculations."""
        # If unspecified, use 1 - identity matrix.
        if self.pairwise_action_loss == {}: self.pwl = 1 - np.identity(self.num_actions)
        # Otherwise use values provided.
        else:
            self.pwl = np.zeros((self.num_actions,self.num_actions))
            for c,losses in self.pairwise_action_loss.items():
                ci = self.action_classes.index(c)
                for cc,l in losses.items():
                    cci = self.action_classes.index(cc)
                    # NOTE: Currently symmetric.
                    self.pwl[ci,cci] = l
                    self.pwl[cci,ci] = l


    def seed(self):
        """Initialise a new tree with its root node."""
        assert hasattr(self, 'o'), 'No data loaded.' 
        self.tree = self.new_leaf([], np.arange(self.num_samples))
        self.num_leaves = 1

    
    def new_leaf(self, address, indices):
        """Create a new leaf, computing attributes where required."""
        nint = bits_to_int(address)
        node = Node(nint = nint,
                    indices = indices,
                    num_samples = len(indices),
                    )
        # Store this leaf as the site of each sample.
        self.nint[indices] = nint
        # Calculate the number of nonterminal samples.
        node.num_samples_nonterminal = np.count_nonzero(self.n[indices] != -1)

        # Action attributes for classification.
        if self.classifier: 
            # Action counts, unweighted and weighted.
            a_one_hot = np.zeros((len(indices), self.num_actions))
            a_one_hot[np.arange(len(indices)), self.a[indices]] = 1
            node.action_counts = np.sum(a_one_hot, axis=0)
            node.weighted_action_counts = np.sum(a_one_hot*self.w[indices].reshape(-1,1), axis=0)
            # Modal action from argmax of weighted_action_counts.
            node.action_best = np.argmax(node.weighted_action_counts)
            # Action probabilities from normalising weighted_action_counts.
            node.action_probs = node.weighted_action_counts / np.sum(node.weighted_action_counts)
            # Action impurity attributes.
            node.per_sample_action_loss = np.inner(self.pwl, node.weighted_action_counts) # This is the increase in action_impurity that would result from adding one sample of each action class.
            node.action_impurity_sum = np.dot(node.weighted_action_counts, node.per_sample_action_loss)  
            node.action_impurity = node.action_impurity_sum / (node.num_samples**2)

        # Action attributes for regression.
        else:
            node.action_best = np.mean(self.a[indices], axis=0)
            if self.action_dim == 1: # For univariate.
                var = np.var(self.a[indices])
                node.action_impurity = math.sqrt(var)
            else: # For multivariate.
                node.a_norm_mean = np.mean(self.a_norm[indices], axis=0)
                var = np.var(self.a_norm[indices], axis=0)
                # NOTE: Taking some of variance in normalised action dimensions, and using standard deviation.
                node.action_impurity = math.sqrt(var.sum()) 
            node.action_impurity_sum = var * node.num_samples

        # Value attributes.
        if self.g != []:
            node.value_mean = np.mean(self.g[indices])
            var = np.var(self.g[indices])
            node.value_impurity_sum = var * node.num_samples
            node.value_impurity = math.sqrt(var) # NOTE: Using standard deviation.
        else: node.value_mean = 0; node.value_impurity = 0      
         
        # Feature derivative attributes.
        if self.d != []:
            node.derivative_mean = np.nanmean(self.d[indices], axis=0)
            node.d_norm_mean = np.nanmean(self.d_norm[indices], axis=0)
            var = np.nanvar(self.d_norm[indices], axis=0)
            node.derivative_impurity_sum = var * node.num_samples_nonterminal
            # NOTE: Taking sum of variance in normalised derivatives, and using standard deviation.
            node.derivative_impurity = math.sqrt(var.sum()) 
        else: node.d_norm_mean = 0; node.derivative_impurity = 0 

        # Placeholder for counterfactual samples and criticality.
        node.cf_indices = []
        # NOTE: Initialising criticalities at zero.
        # This has an effect on both visualisation and HIGHLIGHTS in the case of leaves where no counterfactual data ends up.
        node.criticality_mean = 0; node.criticality_impurity = 0
        return node


    def split_next_best(self, pbar=None):
        """
        Find and split the single most impure leaf in the tree.
        """
        assert hasattr(self, 'tree'), 'Must have started growth process already.'
        if self.leaf_impurity_sums == []: return False
        imp_norm = np.array(self.leaf_impurity_sums) / self.root_impurity_sums
        if self.split_by == 'action': best = np.argmax(imp_norm[:,0])
        elif self.split_by == 'value': best = np.argmax(imp_norm[:,1])
        elif self.split_by == 'derivative': best = np.argmax(imp_norm[:,2])
        # NOTE: For split_by='pick', sum normalised impurities and find argmax.
        elif self.split_by == 'pick': best = np.argmax(imp_norm.sum(axis=1))
        # NOTE: For split_by='weighted', take weighted sum instead. 
        elif self.split_by == 'weighted': best = np.argmax(np.inner(imp_norm, self.impurity_weights))
        nint = self.untried_leaf_nints.pop(best)
        imp = self.leaf_impurity_sums.pop(best)
        node = self.node(nint)
        if self.split(node): # The split is tried here.
            if pbar: pbar.update(1)
            self.untried_leaf_nints.append(node.left.nint)
            self.leaf_impurity_sums.append(self.get_node_impurity_sums(node.left))
            self.untried_leaf_nints.append(node.right.nint)
            self.leaf_impurity_sums.append(self.get_node_impurity_sums(node.right))
            return True
        # If can't make a split, recurse to try the next best.
        else: return self.split_next_best()


    def split(self, node):
        """
        Split a leaf node to minimise some measure of impurity.
        """
        assert node.left == None, 'Not a leaf node.'
        # Check whether able to skip consideration of action, value or normalised derivative entirely.
        if (node.action_impurity > 0) and (self.split_by == 'pick' or self.impurity_weights[0] > 0): do_action = True
        else: do_action = False
        if (node.value_impurity > 0) and (self.split_by == 'pick' or self.impurity_weights[1] > 0): do_value = True
        else: do_value = False
        if (node.derivative_impurity > 0) and (self.split_by == 'pick' or self.impurity_weights[2] > 0): do_derivatives = True
        else: do_derivatives = False
        if (not do_action) and (not do_value) and (not do_derivatives): return False
        # Iterate through features and find best split(s) for each.
        candidate_splits = []
        for f in range(self.num_features):
            candidate_splits += self.split_feature(node, f, do_action, do_value, do_derivatives)
        # If beneficial split found on at least one feature...
        if sum([s[3][0] != None for s in candidate_splits]) > 0: 
            split_quality = [s[3][2] for s in candidate_splits]              
            # Choose one feature to split on.  
            if self.stochastic_splits:
                # Sample in proportion to relative impurity gain.
                chosen_split = np.random.choice(range(len(candidate_splits)), p=split_quality)
            else:
                # Deterministically choose the feature with greatest relative impurity gain.
                chosen_split = np.argmax(split_quality) # Ties broken by lowest index.       
            # Unpack information for this split and create child leaves.
            node.feature_index, node.split_by, indices_sorted, (node.threshold, split_index, _, _, _, _) = candidate_splits[chosen_split]  
            address = int_to_bits(node.nint)
            node.left = self.new_leaf(list(address)+[0], indices_sorted[:split_index])
            node.right = self.new_leaf(list(address)+[1], indices_sorted[split_index:])        
            self.num_leaves += 1
            # Store impurity gains, scaled by node.num_samples, to measure feature importance.
            node.feature_importance = np.zeros((4, self.num_features))
            if do_action:
                fi_action = np.array([s[3][3] for s in candidate_splits if s[1] in ('action','weighted')]) * node.num_samples     
                node.feature_importance[2,:] = fi_action # Potential.
                node.feature_importance[0,node.feature_index] = max(fi_action) # Realised.
            if do_value:
                fi_value = np.array([s[3][4] for s in candidate_splits if s[1] in ('value','weighted')]) * node.num_samples     
                node.feature_importance[3,:] = fi_value # Potential.
                node.feature_importance[1,node.feature_index] = max(fi_value) # Realised.
            # Back-propagate importances to all ancestors.
            while address != ():
                ancestor, address = self.parent(address)
                ancestor.feature_importance += node.feature_importance
            return True
        return False


    # TODO: Make variance calculations sensitive to sample weights.
    def split_feature(self, parent, f, do_action, do_value, do_derivatives): 
        """
        Find the split(s) along feature f that minimise(s) the impurity of the children.
        Impurity gain could be measured for action / value / normalised derivative individually, or as a weighted sum.
        """
        # Sort this node's indices along selected feature.
        indices_sorted = parent.indices[np.argsort(self.o[parent.indices,f])]

        # Initialise variables that will be iteratively modified.
        if do_action:    
            if self.classifier:
                per_sample_action_loss_left = np.zeros(self.num_actions)
                action_impurity_sum_left = 0.
                per_sample_action_loss_right = parent.per_sample_action_loss.copy()
                action_impurity_sum_right = parent.action_impurity_sum.copy()
            else:
                if self.action_dim == 1:
                    action_mean_left = 0.
                    action_impurity_sum_left = 0.
                    action_mean_right = parent.action_best.copy()
                else:
                    a_norm_mean_left = np.zeros(self.action_dim)
                    action_impurity_sum_left = np.zeros(self.action_dim)
                    a_norm_mean_right = parent.a_norm_mean.copy()
                action_impurity_sum_right = parent.action_impurity_sum.copy()
        else: action_impurity_gain = action_rel_impurity_gain = 0
        if do_value: 
            value_mean_left = 0.
            value_impurity_sum_left = 0.
            value_mean_right = parent.value_mean.copy()
            value_impurity_sum_right = parent.value_impurity_sum.copy()
        else: value_impurity_gain = value_rel_impurity_gain = 0
        if do_derivatives:
            d_norm_mean_left = np.zeros(self.num_features)
            derivative_impurity_sum_left = np.zeros(self.num_features)
            num_nonterminal_left = 0
            d_norm_mean_right = parent.d_norm_mean.copy()
            derivative_impurity_sum_right = parent.derivative_impurity_sum.copy()
            num_nonterminal_right = int(parent.num_samples_nonterminal)
        derivative_impurity_gain = d_norm_rel_impurity_gain = 0 # Needs to be defined to catch if first sample has NaN derivatives.

        # Iterate through thresholds.
        if self.split_by == 'pick': best_split = [[f,'action',indices_sorted,[None,None,0,0,0,0]],
                                                  [f,'value',indices_sorted,[None,None,0,0,0,0]],
                                                  [f,'derivative',indices_sorted,[None,None,0,0,0,0]]]
        else: best_split = [[f,self.split_by,indices_sorted,[None,None,0,0,0,0]]]
        for num_left in range(self.min_samples_leaf_abs, parent.num_samples+1-self.min_samples_leaf_abs):
            i = indices_sorted[num_left-1]
            num_right = parent.num_samples - num_left
            
            if do_action:             
                if self.classifier:
                    # Action impurity for classification: use (weighted) Gini.
                    w = self.w[i]
                    a = self.a[i]
                    loss_delta = w * self.pwl[a]
                    per_sample_action_loss_left += loss_delta
                    per_sample_action_loss_right -= loss_delta
                    action_impurity_sum_left += 2 * (w * per_sample_action_loss_left[a]) # NOTE: Assumes self.pwl is symmetric.
                    action_impurity_sum_right -= 2 * (w * per_sample_action_loss_right[a])
                    action_impurity_gain = parent.action_impurity - (((action_impurity_sum_left / num_left) + (action_impurity_sum_right / num_right)) / parent.num_samples) # Divide twice, multiply once.     
    
                else: 
                    # Action impurity for regression: use standard deviation.
                    # Incremental variance computation from http://datagenetics.com/blog/november22017/index.html.
                    if self.action_dim == 1:
                        # For univariate.
                        a = self.a[i]
                        action_mean_left, action_impurity_sum_left = self.increment_mu_and_var_sum(action_mean_left, action_impurity_sum_left, a, num_left, 1)
                        action_mean_right, action_impurity_sum_right = self.increment_mu_and_var_sum(action_mean_right, action_impurity_sum_right, a, num_right, -1)     
                        # Square root turns into standard deviation.
                        # The multiplication by num_left and num_right is correct: sqrt([var*n]*n) = sqrt(var) * sqrt(n^2) = std * n.
                        action_impurity_gain = parent.action_impurity - ((math.sqrt(action_impurity_sum_left*num_left) + math.sqrt(max(0,action_impurity_sum_right)*num_right)) / parent.num_samples)  
                    else:
                        # For multivariate.
                        a_norm = self.a_norm[i]
                        a_norm_mean_left, action_impurity_sum_left = self.increment_mu_and_var_sum(a_norm_mean_left, action_impurity_sum_left, a_norm, num_left, 1)
                        a_norm_mean_right, action_impurity_sum_right = self.increment_mu_and_var_sum(a_norm_mean_right, action_impurity_sum_right, a_norm, num_right, -1)
                        # NOTE: Take sum across features.
                        action_impurity_gain = parent.action_impurity \
                                             - ((math.sqrt(action_impurity_sum_left.sum()*num_left) \
                                             + math.sqrt(max(0,action_impurity_sum_right.sum())*num_right)) \
                                             / parent.num_samples)

                        # print(num_left, '/', parent.num_samples)
                        # print("Left inc ", action_impurity_sum_left)
                        # print("Left     ", num_left * np.var(self.a_norm[indices_sorted[:num_left]], axis=0))
                        # print("Right inc", action_impurity_sum_right)
                        # print("Right    ", num_right * np.var(self.a_norm[indices_sorted[num_left:]], axis=0))
                        # print('')

            if do_value:
                # Value impurity: use standard deviation.
                g = self.g[i]
                value_mean_left, value_impurity_sum_left = self.increment_mu_and_var_sum(value_mean_left, value_impurity_sum_left, g, num_left, 1)
                value_mean_right, value_impurity_sum_right = self.increment_mu_and_var_sum(value_mean_right, value_impurity_sum_right, g, num_right, -1)
                value_impurity_gain = parent.value_impurity - ((math.sqrt(value_impurity_sum_left*num_left) + math.sqrt(max(0,value_impurity_sum_right)*num_right)) / parent.num_samples)

            if do_derivatives:
                # Normalised derivative impurity: use standard deviation.
                d_norm = self.d_norm[i]
                if not(np.isnan(d_norm[0])): # Skip if NaN, which means derivative not defined for this sample.
                    num_nonterminal_left += 1
                    num_nonterminal_right -= 1
                    d_norm_mean_left, derivative_impurity_sum_left = self.increment_mu_and_var_sum(d_norm_mean_left, derivative_impurity_sum_left, d_norm, num_nonterminal_left, 1)
                    if num_nonterminal_right <= 0: 
                        # This prevents a div/0 error.
                        derivative_impurity_sum_right = np.zeros_like(parent.derivative_impurity_sum)
                    else:
                        d_norm_mean_right, derivative_impurity_sum_right = self.increment_mu_and_var_sum(d_norm_mean_right, derivative_impurity_sum_right, d_norm, num_nonterminal_right, -1)
                # NOTE: Derivative is multivariate so take sum across features.
                derivative_impurity_gain = parent.derivative_impurity \
                                         - ((math.sqrt(derivative_impurity_sum_left.sum()*num_nonterminal_left) \
                                         + math.sqrt(max(0,derivative_impurity_sum_right.sum())*num_nonterminal_right)) \
                                         / parent.num_samples_nonterminal)

            # Skip if this sample's feature value is the same as the next one.
            o_f = self.o[i,f]
            o_f_next = self.o[indices_sorted[num_left],f]
            if o_f == o_f_next: continue

            # print('      ', f, (o_f + o_f_next) / 2, num_left, num_right, derivative_impurity_gain)
           
            # Compute relative gains by dividing by the impurities at the root.
            rel_impurity_gains = [action_impurity_gain, value_impurity_gain, derivative_impurity_gain] / self.root_impurities

            if self.split_by == 'pick':
                # Look at action, value and normalised derivative individually.
                if rel_impurity_gains[0] > max(self.min_split_quality, best_split[0][3][2]): # Action.
                    best_split[0][3] = [(o_f + o_f_next) / 2, num_left, rel_impurity_gains[0], action_impurity_gain, value_impurity_gain, derivative_impurity_gain]  
                if rel_impurity_gains[1] > max(self.min_split_quality, best_split[1][3][2]): # Value.
                    best_split[1][3] = [(o_f + o_f_next) / 2, num_left, rel_impurity_gains[1], action_impurity_gain, value_impurity_gain, derivative_impurity_gain]  
                if rel_impurity_gains[2] > max(self.min_split_quality, best_split[2][3][2]): # Normalised derivative.
                    best_split[2][3] = [(o_f + o_f_next) / 2, num_left, rel_impurity_gains[2], action_impurity_gain, value_impurity_gain, derivative_impurity_gain]  
            
            else: 
                # Calculate combined relative gain as weighted sum.
                combined_rel_impurity_gain = np.dot(self.impurity_weights, rel_impurity_gains)
                if combined_rel_impurity_gain > max(self.min_split_quality, best_split[0][3][2]): 
                    best_split[0][3] = [(o_f + o_f_next) / 2, num_left, combined_rel_impurity_gain, action_impurity_gain, value_impurity_gain, derivative_impurity_gain]  

        # print('   ', best_split[0][3])
        return best_split


    def increment_mu_and_var_sum(_, mu, var_sum, x, n, sign):
        """Incremental sum-of-variance computation from http://datagenetics.com/blog/november22017/index.html."""
        d_last = x - mu
        mu += sign * (d_last / n)
        d = x - mu
        var_sum += sign * (d_last * d)
        return mu, var_sum


    def get_node_impurity_sums(self, node):
        """Get the list of impurity sums for any node."""
        if not self.classifier and self.action_dim > 1:
            a_i_s = node.action_impurity_sum.sum() # NOTE: Take sum for multivariate.
        else: a_i_s = node.action_impurity_sum
        return np.array([a_i_s, node.value_impurity_sum, node.derivative_impurity_sum.sum()]) # NOTE: Take sum for derivatives.


# ===================================================================================================================
# METHODS FOR PRUNING.


    def prune_zero_cost(self): return # TODO:


    def prune_MCCP(self): 
        """
        Perform one iteration of pruning for the minimal cost complexity approach.
        See http://mlwiki.org/index.php/Cost-Complexity_Pruning for details.
        NOTE: This could be made a lot more efficient by not re-running everything each time.
        """
        # Subfunction for calculating costs.
        def recurse_compute_cost(node):
            # Total impurity = impurity sums normalised by those at root, dotted with impurity weights.
            node_impurity = np.dot(self.get_node_impurity_sums(node) / self.root_impurity_sums, self.impurity_weights)
            if node.left:
                # Recurse down to left and right subtrees.
                left_impurity, left_num_leaves = recurse_compute_cost(node.left) 
                right_impurity, right_num_leaves = recurse_compute_cost(node.right)
                # Calculate and store cost.
                subtree_impurity = left_impurity + right_impurity
                num_leaves = left_num_leaves + right_num_leaves
                costs.append((node.nint, (node_impurity - subtree_impurity) / (num_leaves + 1)))
                return subtree_impurity, num_leaves
            return node_impurity, 1
        # Run function and sort by cost.
        costs = []
        recurse_compute_cost(self.tree)
        # Remove the subtree below the lowest-cost node.
        node = self.node(sorted(costs, key=lambda x: x[1])[0][0])
        node.left = node.right = None
        # Remove variables associated with the split and register all samples as residing at this node (now a leaf!)
        del node.feature_index; del node.split_by; del node.threshold; del node.feature_importance
        self.nint[node.indices] = node.nint
        # Recompute leaf population.
        self.leaf_nints = self.get_leaf_nints() 
        self.num_leaves = len(self.leaf_nints)
        

# ===================================================================================================================
# METHODS FOR PREDICTION AND SCORING.


    def predict(self, o, attributes=['action'], stochastic_actions=False, use_action_names=True):
        """
        Predict actions for a set of observations, 
        optionally returning some additional information.
        """
        # Test if just one sample has been provided.
        o = np.array(o)
        shp = o.shape
        if len(shp)==1: o = [o]
        if type(attributes) == str: attributes = [attributes]
        R = {}
        for attr in attributes: R[attr] = []
        for oi in o:
            # Propagate each sample to its respective leaf.
            leaf = self.propagate(oi, self.tree)
            if 'action' in attributes:
                if stochastic_actions: 
                    if self.classifier: 
                        # For classification, sample according to action probabilities.
                        a_i = np.random.choice(range(self.num_actions), p=leaf.action_probs)                    
                    else: 
                        # For regression, pick a random member of the leaf.
                        a = self.a[np.random.choice(leaf.indices)]
                else: a_i = leaf.action_best
                # Convert to action names if applicable.
                if self.classifier and use_action_names: R['action'].append(self.action_classes[a_i])
                else: R['action'].append(a_i)
            if 'nint' in attributes: 
                R['nint'].append(leaf.nint)
            if 'action_impurity' in attributes: 
                if self.classifier: R['action_impurity'].append(leaf.action_probs)
                else: R['action_impurity'].append(leaf.action_impurity)
            if 'value' in attributes:
                # NOTE: value/criticality estimation just uses members of same leaf. 
                # This has high variance if the population is small, so could perhaps do better
                # by considering ancestor nodes (lower weight).
                R['value'].append(leaf.value_mean)
            if 'value_impurity' in attributes:
                R['value_impurity'].append(leaf.value_impurity)
            if 'derivative' in attributes:
                R['derivative'].append(leaf.derivative_mean)
            if 'd_norm' in attributes: 
                R['d_norm'].append(leaf.d_norm_mean)
            if 'derivative_impurity' in attributes:
                R['derivative_impurity'].append(leaf.derivative_impurity)
            if 'criticality' in attributes:
                R['criticality'].append(leaf.criticality_mean)
            if 'criticality_impurity' in attributes:
                R['criticality_impurity'].append(leaf.criticality_impurity)
        # Turn into numpy arrays.
        for attr in attributes: R[attr] = np.array(R[attr]) 
        # Clean up what is returned if just one sample or attribute to include.
        if len(attributes) == 1: R = R[attributes[0]]
        return R

    
    def propagate(self, o, node):
        """
        Propagate an unseen sample to a leaf node.
        """
        if node.left: 
            if o[node.feature_index] < node.threshold: return self.propagate(o, node.left)
            return self.propagate(o, node.right)  
        return node


    def score(self, o, a=[], g=[], d_norm=[], 
              action_metric=None, value_metric='mse', d_norm_metric='mse',
              return_predictions=False):
        """
        Score action and/or return and/or normalised_derivative prediction performance on a test set.
        """
        if action_metric == None: 
            if self.classifier: action_metric = 'error_rate'
            else: action_metric = 'mse'
        R = self.predict(o, attributes=['action','value','d_norm'])
        S = []
        if a != []: 
            if action_metric == 'error_rate': order = 0
            elif action_metric == 'mae': order = 1
            elif action_metric == 'mse': order = 2
            S.append(np.linalg.norm(R['action'] - a, ord=order) / len(a))
        if g != []:
            if value_metric == 'mae': order = 1
            elif value_metric == 'mse': order = 2
            S.append(np.linalg.norm(R['value'] - g, ord=order) / len(g))
        if d_norm != []:
            if d_norm_metric == 'mae': order = 1
            elif d_norm_metric == 'mse': order = 2
            diff = R['d_norm'] - d_norm
            diff = diff[~np.isnan(diff).any(axis=1)] # Remove any NaN instances.            
            S.append(np.linalg.norm(diff, ord=order) / len(d_norm))
        if return_predictions: S.append(R)
        return tuple(S)
    
    
# ===================================================================================================================
# METHODS FOR TRAVERSING THE TREE GIVEN VARIOUS LOCATORS.


    def get_leaf_nints(self):
        """
        List the integers of all leaves in the tree.
        """
        def recurse(node):
            if node.left:
                return recurse(node.left) + recurse(node.right) 
            return [node.nint]
        return recurse(self.tree)


    def node(self, identifier):
        """
        Navigate to a node using its address or integer.
        """
        if identifier == None: return None
        elif type(identifier) in (int, np.int64): identifier = int_to_bits(identifier)
        node = self.tree
        for lr in identifier:
            if lr == 0:   assert node.left, 'Invalid identifier.'; node = node.left
            elif lr == 1: assert node.right, 'Invalid identifier.'; node = node.right
            else: raise ValueError('Invalid identifier.')
        return node

    
    def parent(self, address):
        """
        Navigate to a node's parent and return it and its address (not integer!)
        """
        parent_address = address[:-1]
        return self.node(parent_address), parent_address


# ===================================================================================================================
# METHODS FOR WORKING WITH DYNAMIC TRAJECTORIES.

    
    def sample_episode(_, p, n, index):
        """
        Return the full episode before and after a given sample.
        """
        before = []; index_p = index
        if p != []:
            while True: 
                index_p = p[index_p] 
                if index_p == -1: break # Index = -1 indicates the start of a episode.
                before.insert(0, index_p)
        after = [index]; index_n = index
        if n != []:
            while True: 
                index_n = n[index_n] 
                if index_n == -1: break # Index = -1 indicates the end of a episode.
                after.append(index_n)
        return np.array(before), np.array(after)

    
    def split_into_episodes(self, p, n):
        """
        Given lists of predecessor / successor relations (p, n), 
        split the indices by episode and put in temporal order.
        """
        return [self.sample_episode([], n, index[0])[1] 
                for index in np.argwhere(p == -1)]

            
    def get_returns(self, r, p, n): 
        """
        Compute returns for a set of samples.
        """
        if r == []: return []
        if not (p != [] and n != []): return r
        g = np.zeros_like(r)
        # Find indices of terminal observations.
        for index in np.argwhere((n == -1) | (np.arange(len(n)) == len(n)-1)): 
            g[index] = r[index]
            index_p = p[index]
            while index_p >= 0:
                g[index_p] = r[index_p] + (self.gamma * g[index])
                index = index_p; index_p = p[index] 
        return g

    
    def get_returns_n_step_ordered_episode(self, r, p, n, steps):
        """
        Compute returns for an *ordered episode* of samples, 
        with a limit on the number of lookahead steps.
        """
        if steps == None: return self.get_returns(r, p, n)
        if r == []: return []
        if (not (p != [] and n != [])) or steps == 1: return r
        assert steps > 0, 'Steps must be None or a positive integer.'
        # Precompute discount factors.
        discount = [1]
        for t in range(1, steps): discount.append(discount[-1]*self.gamma)
        discount = np.array(discount)
        # Iterate through samples.
        g = np.zeros_like(r)
        for index in range(len(g)):
            next_rewards = r[index:index+steps]
            g[index] = np.dot(next_rewards, discount[:len(next_rewards)])
        return g

    
    def get_derivatives(_, o, p, n):
        """
        Compute the time derivatives of observation features for a set of samples.
        - Use only the next sample (n) because practically interested in what comes after.
        - Where n = -1 (terminal state), return NaNs.
        """
        o_n = o[n]; nans = np.full_like(o[0], np.nan)
        return np.array([(o_n[i] - o[i]) if n[i] != -1 else nans for i in range(len(o))])


    def get_next_nint(self, index):
        """
        Given a sample, find the next leaf encountered in the successor sequence.
        Also return the sequence of samples upto this time.
        """
        nint = self.nint[index]; sequence = [] 
        while True:
            sequence.append(index); index = self.n[index]; next_nint = self.nint[index]
            if next_nint != nint: break
        return next_nint, sequence
    
    
    def get_leaf_transitions(self, nint):
        """
        Given a leaf integer, find all constituent samples whose predecessors are not in this leaf.
        For each of these, step through the sequence of successors until this leaf is departed.
        Record both the previous and next leaf (or 0 if terminal).
        """
        leaf = self.node(nint)
        assert leaf.left == None and leaf.right == None, 'Node must be a leaf.'
        # Filter samples down to those whose predecessor is *not* in this leaf.
        first_indices = leaf.indices[np.nonzero(self.nint[self.p[leaf.indices]] != nint)]
        prev, nxt, both = {}, {}, {}
        for index in first_indices:
            # Get the integer for the previous leaf.
            nint_p = self.nint[self.p[index]]
            # Get the integer for the next leaf, and the sequence of successors up to that time.
            next_nint, sequence = self.get_next_nint(index)
            # Store information about this sequence with previous, next and both together.
            info = [len(sequence), self.g[index]] # Sequence length and return from first sample.
            if nint_p in prev: prev[nint_p].append(info)
            else: prev[nint_p] = [info]
            if next_nint in nxt: nxt[next_nint].append(info)
            else: nxt[next_nint] = [info]
            pn = (nint_p, next_nint) 
            if pn in both: both[pn].append(info)
            else: both[pn] = [info]
        # Convert dictionary entries into numpy arrays.
        prev = {k:np.array(v) for k,v in prev.items()}
        nxt = {k:np.array(v) for k,v in nxt.items()}
        both = {k:np.array(v) for k,v in both.items()}
        return prev, nxt, both, len(first_indices)

    
    def get_leaf_transition_probs(self, nint):
        """
        Convert the output of the get_leaf_transitions method into probabilities:
            - Previous/next leaf marginal.
            - Previous/next conditional (on next/previous).
        """
        prev, nxt, both, n = self.get_leaf_transitions(nint)
        P = {'prev':{},'next':{}}
        # Function for processing sequences into the right form.
        f = lambda seqs, n : [len(seqs)/n, len(seqs)] + list(np.mean(seqs, axis=0))
        # For marginals, normalise by total number of sequences.
        P['prev']['marginal'] = {k:f(v, n) for k,v in prev.items()}
        P['next']['marginal'] = {k:f(v, n) for k,v in nxt.items()}
        # For conditionals, normalise by number of sequences matching condition.
        for cond, (_,n,_,_) in P['next']['marginal'].items():
            P['prev'][cond] = {k[0]:f(v, n) for k,v in both.items()
                               if k[1] == cond} # Filter with condition.
        for cond, (_,n,_,_) in P['prev']['marginal'].items():
            P['next'][cond] = {k[1]:f(v, n) for k,v in both.items()
                               if k[0] == cond} # Filter with condition.
        return P

    
    def compute_all_leaf_transition_probs(self):
        """
        Run the get_leaf_transition_probs method for all leaves and store.
        """
        self.P = {}
        with tqdm(desc='Transition probability calculation', total=self.num_leaves) as pbar:
            for nint in self.leaf_nints:
                self.P[nint] = self.get_leaf_transition_probs(nint)
                pbar.update(1)
        

    def get_paths_from_source(self, costs, source_index, 
                              best_cost, worst_cost, higher_is_better,
                              combine, better, p_n, conditional):
        """
        Use a variant of the Dijkstra algorithm to find the best sequence of transitions from one leaf to all others.
        Where "best" is currently measured by total probability.
        Conditional argument conditions transition probabilities on previous leaf. 
        This makes for sparser data: greater chance of failure but better quality when succeeds.
        """
        costs[source_index][1] = False
        costs[source_index][2] = best_cost
        depth = 0; cond = 'marginal'
        while True:
            depth += 1
            # Sort unvisited leaves by total cost and identify the best one to visit next.
            priority = sorted([c for c in costs if c[4] == False], key=lambda c: c[2], reverse=higher_is_better)
            if priority == []: break # All leaves visited.
            index, previous_index, cost_so_far, _, _ = priority[0]
            # Check if we have reached the end of the accessible leaves.
            if previous_index == None: break 
            # Mark the leaf as visited.
            costs[index][4] = True
            # For conditional transition, condition on previous leaf.
            if conditional and index != source_index: cond = self.leaf_nints[previous_index]
            for next_nint, vals in self.P[self.leaf_nints[index]][p_n][cond].items():
                if next_nint != 0:
                    # Compute cost to this leaf.
                    cost_to_here = combine(cost_so_far, vals[0])
                    # If this is better than the stored one, overwrite.
                    next_index = self.leaf_nints.index(next_nint)
                    if better(cost_to_here, costs[next_index][2]):
                        costs[next_index] = [next_index, index, cost_to_here, vals[0], False]
        # Information to return is previous leaf indices, total costs and one-step costs.
        _, prev, costs_total, costs_one_step, _ = [list(x) for x in zip(*costs)]
        return prev, costs_total, costs_one_step

    
    def compute_paths_matrix(self, reverse=False, cost_by='prob', conditional=False):
        """
        Run the get_paths_from_source method for all leaves and store.
        """
        # Set up some parameters for the search process.
        if cost_by == 'prob':
            best_cost = 1; worst_cost = 0; higher_is_better = True
            combine = lambda a, b: a * b
            better = lambda a, b: a > b
        else: raise Exception(f'cost_by {cost_by} not yet implemented.')
        if reverse: p_n = 'prev'
        else: p_n = 'next'
        self.path_prev, self.path_costs_total, self.path_costs_one_step = [], [], []
        with tqdm(desc='Transition paths matrix calculation', total=self.num_leaves) as pbar:
            for source_index in range(self.num_leaves):
                # Initialise costs.
                # Elements are [index, index of previous leaf, total cost, immediate cost, visited?]
                # TODO: Pre-populate with places we already know how to get to.
                costs_init = [[i, None, worst_cost, None, False] for i in range(self.num_leaves)]
                # Run the search.
                p, t, o = self.get_paths_from_source(costs_init, source_index, 
                                                     best_cost, worst_cost, higher_is_better,
                                                     combine, better, p_n, conditional)
                self.path_prev.append(p)
                self.path_costs_total.append(t)
                self.path_costs_one_step.append(o)
                pbar.update(1)
        

    def get_leaf_to_leaf_path(self, source, dest): 
        """
        Given a source and destination leaf, get the lowest-cost path between them.
        """
        source_index = self.leaf_nints.index(source)
        dest_index = self.leaf_nints.index(dest)
        prev = self.path_prev[source_index]
        costs_one_step = self.path_costs_one_step[source_index]
        # Reconstruct the path by backtracking.
        index = dest_index; path = [(source, None)]
        while index != source_index:
            path.insert(1, (self.leaf_nints[index], costs_one_step[index]))
            index = prev[index]
            if index == None: return False, False # No path found.
        return path, self.path_costs_total[source_index][dest_index]


    def path_between(self, source=False, source_features={}, source_attributes={}, 
                           dest=False, dest_features={}, dest_attributes={}, 
                           feature_mode = 'contain', try_reuse_df=True):
        """
        Use the get_leaf_to_leaf_path method to find paths between pairs of leaves matching condtions.
        Conditions could be on feature or attribute values.
        Can optionally specify a single start or end leaf.
        """
        if not(try_reuse_df and hasattr(self, 'df')): df = self.to_dataframe()
        df = self.df.loc[self.df['kind']=='leaf'] # Only care about leaves.
        # List source leaf integers.
        if source == False:
            source = self.df_filter(df, source_features, source_attributes, mode=feature_mode).index.values
            print("Num sources =", len(source))
            #print(source)
        elif type(source) == tuple: source = [source]
        # List destination leaf integers.
        if dest == False:
            dest = self.df_filter(df, dest_features, dest_attributes, mode=feature_mode).index.values
            print("Num dests =", len(dest))
        elif type(dest) == tuple: dest = [dest]
        # Find the best path to each leaf matching the condition.
        paths = []
        with tqdm(desc='Path finding', total=len(source)*len(dest)) as pbar:
            for s in source:
                for d in dest:
                    path, cost = self.get_leaf_to_leaf_path(s, d)
                    pbar.update(1)
                    if path != False:
                        paths.append((path, cost))
        # Sort the paths by their cost.
        paths.sort(key=lambda x:x[1], reverse=True)
        print("Num paths =", len(paths))
        return paths


# ===================================================================================================================
# METHODS FOR WORKING WITH COUNTERFACTUAL DATA.


    def cf_load_data(self, o, a, r, p, n, regret_steps=np.inf, append=True):
        """
        Counterfactual data looks a lot like target data, 
        but is assumed to originate from a policy other than the target one,
        so must be kept separate.
        """
        assert hasattr(self, 'tree'), 'Must have already grown tree.'
        assert len(o) == len(a) == len(r) == len(p) == len(n), 'All inputs must be the same length.'
        assert min(p) == min(n) >= -1, 'Episode start/end must be denoted by index of -1.'
        if not hasattr(self, 'cf'): self.cf = counterfactual()
        # Convert actions into indices.
        if self.classifier: a = np.array([self.action_classes.index(c) for c in a]) 
        # Compute return for each new sample.
        g = self.get_returns(r, p, n)
        # Use the extant tree to get a leaf integer for each sample, and predict its value under the target policy.
        R = self.predict(o, attributes=['nint','value'])       
        nints = R['nint']; v_t = R['value']
        # Store the counterfactual data, appending if applicable.
        num_samples_prev = self.cf.num_samples
        if append == False or num_samples_prev == 0:
            self.cf.o, self.cf.a, self.cf.r, self.cf.p, self.cf.n, self.cf.g, self.cf.v_t = o, a, r, p, n, g, v_t
            self.cf.regret = np.empty_like(g) # Empty; compute below.
            self.cf.num_samples = len(o)
        else:
            self.cf.o = np.vstack((self.cf.o, o))
            self.cf.a = np.hstack((self.cf.a, a))
            self.cf.r = np.hstack((self.cf.r, r))
            self.cf.p = np.hstack((self.cf.p, p))
            self.cf.n = np.hstack((self.cf.n, n))
            self.cf.g = np.hstack((self.cf.g, g))
            self.cf.v_t = np.hstack((self.cf.g, v_t))
            self.cf.regret = np.hstack((self.cf.regret, np.empty_like(g))) # Empty; compute below.
            self.cf.num_samples += len(o)
        # Compute regret for each new sample. 
        self.cf_compute_regret(regret_steps, num_samples_prev)
        # Store new samples at nodes by back-propagating.
        samples_per_leaf = {nint:[] for nint in set(nints)}
        for index, nint in zip(np.arange(num_samples_prev, self.cf.num_samples), nints):
            samples_per_leaf[nint].append(index)
        for nint, indices in samples_per_leaf.items():
            address = int_to_bits(nint) 
            self.node(address).cf_indices += indices 
            while address != ():
                ancestor, address = self.parent(address)
                ancestor.cf_indices += indices
        # (Re)compute criticality for all nodes in the tree.
        self.cf_compute_node_criticalities()
        # Finally, use the leaves to estimate criticality for every sample in the training dataset.
        self.c = self.predict(self.o, attributes=['criticality'])


    def cf_compute_regret(self, steps, start_index=0):
        """
        Compute n-step regrets vs the estimated value function
        for samples in the counterfactual dataset,
        optionally specifying a start index to prevent recomputing.
        """
        assert not (start_index > 0 and steps != self.cf.regret_steps), "Can't use different values of regret_steps in an appended dataset; recompute first."
        self.cf.regret[start_index:] = np.nan
        for index in np.argwhere(self.cf.p == -1):
            if index >= start_index:
                ep_indices, regret = self.cf_get_regret_trajectory(index[0], steps)
                self.cf.regret[ep_indices] = regret
        self.cf.regret_steps = steps


    def cf_get_regret_trajectory(self, index, steps=np.inf):
        """
        Compute n-step regrets vs the estimated value function
        for a trajectory of counterfactual samples starting at index.
        """
        # Retrieve all successive samples in the counterfactual episode.
        _, indices = self.sample_episode(self.cf.p, self.cf.n, index)
        o = self.cf.o[indices]
        r = self.cf.r[indices]
        p = self.cf.p[indices]
        n = self.cf.n[indices]
        v_t = self.cf.v_t[indices]
        # Verify steps.
        num_samples = len(r)
        if steps >= num_samples: steps = num_samples-1
        else: assert steps > 0, 'Steps must be None or a positive integer.'

        # Compute n-step returns.
        g = self.get_returns_n_step_ordered_episode(r, p, n, steps)
        # Compute regret = target v - (n-step return + discounted value of sample in n-steps' time).
        v_t_future = np.pad(v_t[steps:], (0, steps), mode='constant') # Pad end of episodes.
        regret = v_t - (g + (v_t_future * (self.gamma ** steps)))
        return indices, regret

        
    def cf_compute_node_criticalities(self):
        """
        The criticality of a node is defined as the mean regret
        of the counterfactual samples lying within it. 
        It can be defined for all nodes, not just leaves.
        """
        assert hasattr(self, 'tree') and hasattr(self, 'cf') and self.cf.num_samples > 0, 'Must have tree and counterfactual data.'
        def recurse(node):
            if node.cf_indices != []: 
                regrets = self.cf.regret[np.array(node.cf_indices)]
                node.criticality_mean = np.nanmean(regrets)
                node.criticality_impurity = np.nanstd(regrets)
            if node.left:
                recurse(node.left)
                recurse(node.right)  
        recurse(self.tree)


    def init_highlights(self, crit_smoothing=5, max_length=np.inf):
        """
        Precompute all the data required for our interpretation 
        of the HIGHLIGHTS algorithm (Amir et al, AAMAS 2018).
        """
        assert hasattr(self, 'tree') and hasattr(self, 'cf') and self.cf.num_samples > 0, 'Must have tree and counterfactual data.'
        smooth_window = (2 * crit_smoothing) + 1
        ep_obs, ep_crit, all_crit, all_timesteps = [], [], [], []
        # Split the training data into episodes and iterate through.
        episodes = self.split_into_episodes(self.p, self.n)
        for ep, indices in enumerate(episodes):
            # Temporally smooth the criticality time series for this episode.
            if crit_smoothing == 0: crit = self.c[indices]
            else: crit = running_mean(self.c[indices], smooth_window)
            ep_crit.append(crit)
            # Store 'unravelled' criticality and timestep indices so can sort globally.
            all_crit += list(crit)
            all_timesteps += [(ep, t) for t in range(len(indices))]            
        # Sort timesteps (across all episodes) by smoothed criticality.
        all_timesteps_sorted = [t for _,t in sorted(zip(all_crit, all_timesteps))]
        # Now iterate through the sorted timesteps and construct highlights in order of criticality.
        self.highlights_keypoint, self.highlights_indices, self.highlights_crit, used = [], [], [], set()
        if max_length != np.inf: max_length = max_length // 2
        with tqdm(desc='Assembling HIGHLIGHTS', total=len(all_timesteps_sorted)) as pbar:
            while all_timesteps_sorted != []:
                # Pop the most critical timestep off all_timesteps_sorted.
                timestep = all_timesteps_sorted.pop()
                pbar.update(1)
                if timestep not in used: # If hasn't been used in a previous highlight.
                    ep, t = timestep
                    # Assemble a list of timesteps either side.
                    trajectory = np.arange(max(0, t-max_length), min(len(episodes[ep]), t+max_length))
                    # Store this highlight and its smoothed criticality time series.
                    self.highlights_keypoint.append(episodes[ep][t])
                    self.highlights_indices.append(episodes[ep][trajectory])
                    self.highlights_crit.append(ep_crit[ep][trajectory])
                    # Record all these timesteps as used.
                    used = used | {(ep, tt) for tt in trajectory}
        
        
    def top_highlights(self, k=1, diversity=0.1, p=2): 
        """
        Assuming init_highlights has already been run, return the k best highlights.
        """
        assert hasattr(self, 'highlights_indices'), 'Must run init_highlights first.'
        assert diversity >= 0 and diversity <= 1, 'Diversity must be in [0,1].'
        if k > 1:
            # Calculate the distance between the opposite corners of the global feature lims box.
            # This is used for normalisation.
            dist_norm = minkowski_distance(self.global_feature_lims[:,0] * self.feature_scales,
                                           self.global_feature_lims[:,1] * self.feature_scales, p=p)
        chosen_highlights_keypoint, chosen_highlights_indices, chosen_highlights_crit, i, n = [], [], [], -1, 0
        while n < k: 
            i += 1
            if i >= len(self.highlights_indices): break
            if diversity > 0 and n > 0:
                # Find the feature-scaled Frechet distance to each existing highlight and normalise.
                # dist_to_others = np.array([scaled_frechet_dist(self.highlights_obs[i],
                #                                                chosen_highlights_obs[j],
                #                                                self.feature_scales, p=p)
                #                            for j in range(n)]) / dist_norm 
                dist_to_others = np.array([minkowski_distance(self.o[self.highlights_keypoint[i]] * self.feature_scales,
                                                              self.o[chosen_highlights_keypoint[j]] * self.feature_scales, p=p)
                                           for j in range(n)]) / dist_norm                
                
                print(i, n, min(dist_to_others), max(dist_to_others))
                
                assert max(dist_to_others) <= 1, 'Error: Scaled-and-normalised distance should always be <= 1!'
                # Ignore this highlight if the smallest Frechet distance is below a threshold.
                if min(dist_to_others) < diversity: continue
            # Store this highlight.
            chosen_highlights_keypoint.append(self.highlights_keypoint[i])
            chosen_highlights_indices.append(self.highlights_indices[i])
            chosen_highlights_crit.append(self.highlights_crit[i])
            n += 1
        return chosen_highlights_indices, chosen_highlights_crit


# ===================================================================================================================
# METHODS WHICH INVOLVE REPRESENTING THE TREE AS A PANDAS DATAFRAME.


    def to_dataframe(self, out_file=None, leaves_only=False):
        """
        Represent the nodes of the tree as rows of a Pandas dataframe.
        """
        def recurse(node, partitions=[]):
            if (not leaves_only) or (not node.left):

                # Basic identification.
                data['nint'].append(node.nint)
                data['depth'].append(len(int_to_bits(node.nint)))
                data['kind'].append(('internal' if node.left else 'leaf'))
                # Feature ranges.
                ranges = self.global_feature_lims.copy()
                for f in range(self.num_features):
                    for lr, sign in enumerate(('>','<')):
                        thresholds = [p[2] for p in partitions if p[0] == f and p[1] == lr]
                        if len(thresholds) > 0: 
                            # The last partition for each (f, lr) pair is always the most restrictive.
                            ranges[f,lr] = thresholds[-1]
                        data[f'{self.feature_names[f]} {sign}'].append(ranges[f,lr])
                # Population information.
                data['num_samples'].append(node.num_samples)
                data['sample_fraction'].append(node.num_samples / self.num_samples)
                weight_sum = sum(self.w[node.indices])
                data['weight_sum'].append(weight_sum)
                data['weight_fraction'].append(weight_sum / self.w_sum)
                # Volume and density information.
                #   NOTE: Volume of a leaf = product of feature ranges, scaled by feature_scales_norm.
                volume = np.prod((ranges[:,1] - ranges[:,0]) * feature_scales_norm)
                data['volume'].append(volume)
                data['sample_density'].append(node.num_samples / volume)
                data['weight_density'].append(weight_sum / volume)
                # Action information.
                data['action_impurity'].append(node.action_impurity)
                data['action_impurity_sum'].append(node.action_impurity_sum)
                if self.classifier: 
                    data['action'].append(node.action_best)
                    data['action_counts'].append(node.action_counts)
                    data['weighted_action_counts'].append(node.weighted_action_counts)
                elif self.action_dim == 1: data['action'].append(node.action_best)
                else: 
                    for d in range(self.action_dim): data[f'action_{d}'].append(node.action_best[d]) 
                # Value information.
                data['value'].append(node.value_mean)
                data['value_impurity'].append(node.value_impurity)
                data['value_impurity_sum'].append(node.value_impurity_sum)
                # Derivative information.
                data['derivatives'].append(node.derivative_mean)
                data['derivative_impurity'].append(node.derivative_impurity)
                data['derivative_impurity_sum'].append(node.derivative_impurity_sum)
                # Criticality information.
                data['criticality'].append(node.criticality_mean)
                data['criticality_impurity'].append(node.criticality_impurity)
                # Total impurity.
                data['total_impurity'].append(np.dot(self.get_node_impurity_sums(node) / self.root_impurity_sums, self.impurity_weights))

            # For decision nodes, recurse to children.
            if node.left:
                recurse(node.left, partitions+[(node.feature_index, 1, node.threshold)])
                recurse(node.right, partitions+[(node.feature_index, 0, node.threshold)])

        # Set up dictionary keys.
        #    Basic identification.
        keys = ['nint','depth','kind']
        #    Feature ranges.
        keys += [f'{f} {sign}' for f in self.feature_names for sign in ('>','<')] 
        #    Population information.
        keys += ['num_samples','sample_fraction','weight_sum','weight_fraction','volume','sample_density','weight_density'] 
        #    Action information.
        if self.classifier: keys += ['action','action_counts','weighted_action_counts']
        elif self.action_dim == 1: keys += ['action'] 
        else: keys += [f'action_{d}' for d in range(self.action_dim)] 
        keys += ['action_impurity','action_impurity_sum']
        #    Value information.
        keys += ['value','value_impurity','value_impurity_sum']
        #    Derivative information.
        keys += ['derivatives','derivative_impurity','derivative_impurity_sum']
        #    Criticality information.
        keys += ['criticality','criticality_impurity']
        #    Total impurity.
        keys += ['total_impurity']
        data = {k:[] for k in keys}
        # NOTE: For volume calculations, normalise feature scales by geometric mean.
        # This tends to keep hyperrectangle volumes reasonable.   
        feature_scales_norm = self.feature_scales / np.exp(np.mean(np.log(self.feature_scales)))
        # Populate dictionary by recursion through the tree.
        recurse(self.tree)        
        # Convert into dataframe.
        self.df = pd.DataFrame.from_dict(data).set_index('nint')
        # If no out file specified, just return.
        if out_file == None: return self.df
        else: self.df.to_csv(out_file+'.csv', index=True)


    def df_filter(self, df, features={}, attributes={}, mode=None):
            """
            Filter the subset of nodes that overlap with (or are entirely contained within)
            a set of feature ranges, and / or have attributes within specified ranges.
            If using features, compute the proportion of overlap for each.
            """
            # Build query.
            query = []; feature_ranges = []
            for f, r in features.items():
                feature_ranges.append(r)
                # Filter by features.
                if mode == 'overlap':
                    # Determine whether two ranges overlap:
                    query.append(f'`{f} <`>={r[0]}')
                    query.append(f'`{f} >`<={r[1]}')
                elif mode == 'contain':
                    query.append(f'`{f} >`>={r[0]}')
                    query.append(f'`{f} <`<={r[1]}')
                else: raise Exception('Need to specify mode: overlap or contain.')
            for attr, r in attributes.items():
                # Filter by attributes.
                query.append(f'{attr}>={r[0]} & {attr}<={r[1]}')
            # Filter dataframe.
            df = df.query(' & '.join(query))
            if features != {}:
                # If using features, compute overlap proportions,
                # and store this in a new column of the dataframe.
                # There's a lot of NumPy wizardry going on here!
                feature_ranges = np.array(feature_ranges)
                node_ranges = np.dstack((df[[f'{f} >' for f in features]].values,
                                    df[[f'{f} <' for f in features]].values))
                overlap = np.maximum(0, np.minimum(node_ranges[:,:,1], feature_ranges[:,1]) 
                                    - np.maximum(node_ranges[:,:,0], feature_ranges[:,0]))
                df['overlap'] = np.prod(overlap / (node_ranges[:,:,1] - node_ranges[:,:,0]), axis=1)                         
            return df


    def node_lims(self, nint, try_reuse_df=True):
        """
        Given a node integer, return its feature limits.
        """
        if not(try_reuse_df and hasattr(self, 'df')): self.to_dataframe()
        node = self.df.loc[nint]    
        lims = []
        for f in self.feature_names: 
            lims.append([node[f'{f} >'], node[f'{f} <']])
        return np.array(lims)
    
    
    def closest_point_in_node(self, o, nint, try_reuse_df=True):
        """
        Given an observation vector and a node integer, find the closest observation 
        within the node's hyperrectangle.
        """
        if not(try_reuse_df and hasattr(self, 'df')): self.to_dataframe()
        node = self.df.loc[nint]
        o_nearest = []; inside = True
        for i, f in enumerate(self.feature_names): 
            # Get lower and upper boundaries for each feature.
            l, u = node[f'{f} >'], node[f'{f} <']
            # Three possible cases:
            if o[i] < l: o_nearest.append(l); inside = False
            elif o[i] > u: o_nearest.append(u); inside = False
            else: o_nearest.append(o[i])
        return np.array(o_nearest), inside


# ===================================================================================================================
# METHODS FOR RULE-BASED EXPLANATION.


    def explain():
        # TODO: Generalise. 
        return 


    def counterfactual(self, o, foil, try_reuse_df=True, sort_by='L0_L2', return_all=False):
        """
        Counterfactual explanation of a sample o by finding the minimal foil.
        """
        
        # Get the nints for all leaves which match the foil specification.
        if not(try_reuse_df and hasattr(self, 'df')): self.to_dataframe()
        foil_nints = list(self.df_filter(self.df.loc[self.df['kind']=='leaf'], attributes=foil).index)
        options = []
        for nint in foil_nints: 
            # For each, find the L0 and L2 norms to the closest point in that leaf.
            o_foil, inside = self.closest_point_in_node(o, nint)
            if inside: return None, o, 0., 0. # If o is inside any of these leaves, no counterfactual needed. Return it back.
            delta = np.multiply(o_foil - o, self.feature_scales)
            options.append([nint, o_foil, np.linalg.norm(delta, ord=0), np.linalg.norm(delta, ord=2)])
        if sort_by == 'L0_L2': 
            # Sort foil options by L0, then by L2.
            options = sorted(options, key=lambda x: (x[2], x[3]))
        if sort_by == 'L2':
            # Sort foil options by L2.
            options = sorted(options, key=lambda x: x[3])
        if return_all: return options
        else: return options[0]


# ===================================================================================================================
# METHODS FOR SHAP VALUE CALCULATION.


    # def SHAP_optimised(self, o):
    #     """

    #     unique_path is the path of unique features we have split on so far.
    #     """
        
    #     phi = [0] * self.num_features
        
    #     def recurse(node, unique_path, zero_fraction, one_fraction, fi):
    #         """
    #         xxx
    #         """
    #         unique_path = extend(unique_path, zero_fraction, one_fraction, fi)
    #         if not node.left:
    #             # If have reached a leaf node, calculate contributions from every feature in the path.
    #             for i in range(1, len(unique_path)):
    #                 # Undo the weight extension for this feature.
    #                 w = sum(p[3] for p in unwind(unique_path, i))
    #                 # Contribution from subsets matching this leaf.
    #                 phi[i] += w * (unique_path[i][2] - unique_path[i][1]) * node.action_best
    #         else:
    #             # Otherwise, determine which child is "hot", meaning o would follow it.
    #             if o[fi] < node.feature_index: h, c = node.left, node.right
    #             else:                          h, c = node.right, node.left
    #             iz = iz = 1
    #             k = findfirst()

    #     def extend(unique_path, zero_fraction, one_fraction, fi):
    #         """
    #         Grow subset path according to a given fraction of ones (o) and zeros (z).
    #         """
    #         l = len(unique_path)
    #         # Initialise subsets of size l.
    #         unique_path.append([fi, zero_fraction, one_fraction, (1 if l == 0 else 0)])
    #         # Grow subsets using z and o.
    #         for i in range(l, 0, -1):
    #             unique_path[i][3] += one_fraction * unique_path[i-1][3] * i / (l + 1) # Subsets that grow by one.
    #             unique_path[i-1][3] *= zero_fraction * (l - i) / (l + 1) # Subsets that stay the same.
    #         return unique_path

    #     def unwind(unique_path, i):
    #         """
    #         Inverts the ith call to extend().
    #         """
    #         return

    #     recurse(self.tree, [], 1, 1, 0)


    def SHAP_naive(self, o, attribute='value', try_reuse_df=True):
        """
        Naive / slow computation of per-feature SHAP values for a sample o.
        """
        if not(try_reuse_df and hasattr(self, 'df')): self.to_dataframe()
        leaves = self.df.loc[self.df['kind']=='leaf'] # Only care about leaves.
        # For each feature in o, identify the set of leaves compatible with it and store their indices.
        leaves = leaves.reset_index() # These indices are *not* the same as nints. Allows fast NumPy indexing later.
        indices = []
        for f in range(self.num_features):
            indices.append(set(self.df_filter(leaves, 
                                              features={self.feature_names[f]: [o[f], o[f]]},
                                              mode='overlap').index))
        # Get predictions and weights into a big NumPy array, much faster than Pandas.
        assert attribute == 'value' and not self.classifier, 'Only one implemented.'
        preds_and_weights = leaves[['value','weight_sum']].values
        marginals = {}; components = [{} for _ in range(self.num_features)]
        # Iterate through powerset of features (from https://stackoverflow.com/a/1482316).
        for subset in tqdm(chain.from_iterable(combinations(range(self.num_features), r) for r in range(1,self.num_features+1))):
            # Identify the set of leaves compatible with this feature set.
            matching_leaves = set.intersection(*(indices[fj] for fj in subset))
            d = preds_and_weights[np.array(list(matching_leaves))]
            # Compute marginal prediction for this subset.
            marginals[subset] = np.average(d[:,0], weights=d[:,1])
            if len(subset) > 1:
                for i, f in enumerate(subset):
                    # For each feature in the subset, compute the effect of adding it.
                    subset_without = subset[:i]+subset[i+1:]
                    components[f][subset_without] = marginals[subset] - marginals[subset_without]
        # Finally, compute SHAP values.
        n_fact = math.factorial(self.num_features)
        w = [math.factorial(i) * math.factorial(self.num_features - i - 1) / n_fact for i in range(1,self.num_features)]
        print(w)
        SHAP = [sum(w[len(s)-1] * val # weighted sum of contributions...
                for s, val in c.items()) # from each subset...     
                for c in components # for each feature.
                ]
        return SHAP


# ===================================================================================================================
# METHODS FOR TREE DESCRIPTION AND VISUALISATION.

    
    # TODO: This needs updating.
    def to_code(self, do_action=True, do_value=False, do_derivatives=False, 
                      do_comments=False, alt_action_names=None, out_file=None): 
        """
        Print tree rules as an executable function definition. Adapted from:
        https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree
        """
        lines = []
        lines.append(f"def tree({', '.join(self.feature_names)}):")
        def recurse(node, depth=1):
            indent = "    " * depth            
            # Decision nodes.
            if node.left:
                feature = self.feature_names[node.feature_index]
                #if comment: lines.append("{}if {} < {}: # depth = {}, best action = {}, confidence = {}%, weighted counts = {}, action_impurity = {}".format(indent, feature, node.threshold, depth, pred, conf, node.weighted_action_counts, node.action_impurity))
                #else: 
                lines.append("{}if {} < {}:".format(indent, feature, node.threshold))
                recurse(node.left, depth+1)
                if do_comments: lines.append("{}else: # if {} >= {}".format(indent, feature, node.threshold))
                else: lines.append("{}else:".format(indent))
                recurse(node.right, depth+1)
            # Leaf nodes.
            else:
                preds = []
                if do_action: 
                    a_pred = node.action_best
                    if self.classifier:
                        a_pred = self.action_classes[a_pred]
                        if alt_action_names != None: a_pred = alt_action_names[a_pred]
                    elif self.action_dim > 1:
                        a_pred = '{'+', '.join([f"'{n}': {a}" for n, a in zip(self.action_names, a_pred)])+'}'
                    preds.append(f"'action': {a_pred}") 
                if do_value: preds.append(f"'value': {node.value_mean}")
                if do_derivatives: preds.append(f"'derivatives': {node.derivative_mean}")
                preds = '{' + ', '.join(preds) + '}'
                if do_comments: lines.append("{}# population = {}".format(indent, node.num_samples))
                lines.append("{}return {}".format(indent, preds))

        recurse(self.tree)
        if out_file == None:  # If no out file specified, just print.
           for line in lines: print(line) 
        else: 
            with open(out_file+'.py', 'w', encoding='utf-8') as f:
                for l in lines: f.write(l+'\n')


    def treemap(self, features, attributes=[None], 
                conditions=[], try_reuse_df=True,
                visualise=True, axes=[],
                action_colours=None, cmap_percentiles=[0,100], cmap_midpoints=[], density_percentile=50,
                density_shading=False, edge_colour=None, show_cbar=True, show_nints=False):
        """
        Create a treemap visualisation across one or two features, 
        possibly projecting across all others.
        """
        if type(features) in (str, int): features = [features]
        n_f = len(features)
        assert n_f in (1,2), 'Can only plot in 1 or 2 dimensions.'
        cmaps = {'action':matplotlib.cm.copper,#custom_cmap, # For regression only.
                 'action_impurity':matplotlib.cm.Reds,#custom_cmap.reversed(),
                 'value':matplotlib.cm.viridis,#custom_cmap,
                 'value_impurity':matplotlib.cm.Reds,#custom_cmap.reversed(),
                 'derivatives':matplotlib.cm.copper,
                 'derivative_impurity':matplotlib.cm.Reds,#custom_cmap.reversed(),
                 'criticality':custom_cmap,
                 'criticality_impurity':matplotlib.cm.Reds,#custom_cmap.reversed(),
                 'total_impurity':matplotlib.cm.Reds,#custom_cmap,
                 'sample_density':matplotlib.cm.gray,
                 'weight_density':matplotlib.cm.gray,
                 None:None
                 }
        if type(attributes) == str or attributes == None: attributes = [attributes]
        n_a = len(attributes)
        for attr in attributes: 
            assert attr in cmaps or attr[:6] == 'action', 'Invalid attribute.'
        # If doing density_shading, need to evaluate sample_density even if not requested.
        attributes_plus = attributes.copy()
        if density_shading and not 'sample_density' in attributes: 
            attributes_plus += ['sample_density']
        
        # Get the dataframe representation of the tree.
        if not(try_reuse_df and hasattr(self, 'df')): self.to_dataframe()
        df = self.df.loc[self.df['kind']=='leaf'] # Only care about leaves.

        # If conditions specified, use them to filter down the dataframe.
        if conditions == []: conditions = [None] * self.num_features
        else: assert len(conditions) == self.num_features
        filter_spec = {}
        sliced = {f:False for f in self.feature_names}
        filtered = {f:True for f in self.feature_names}
        for fi, cond in enumerate(conditions):
            f = self.feature_names[fi]
            glob = self.global_feature_lims[fi]
            if cond == None: 
                filter_spec[f] = list(glob)
                filtered[f] = False
            elif type(cond) in (int, float, np.float64):
                filter_spec[f] = list(np.clip([cond, cond], glob[0], glob[1]))
                sliced[f] = True 
            elif type(cond) in (tuple, list):
                filter_spec[f] = list(np.clip(cond, glob[0], glob[1]))
            else: raise Exception()
        if any(filtered.values()): # If have filtered along at least one feature dimension.
            df = self.df_filter(df, features=filter_spec, mode='overlap')
            assert len(df) > 0, 'No leaves meet the filter specification!'
            # If have sliced at least once, all overlaps will be 0.0, so drop them. 
            if any(sliced.values()): df = df.drop(columns=['overlap'])
            else: df = df.rename(columns={'overlap': 'overlap_filter'}) # Otherwise rename because use 'overlap' later.

        # Conditions for the visualisation features become limits of the axes.
        axis_lims = np.array([filter_spec[f] for f in features])
        assert not any(axis_lims[:,1]- axis_lims[:,0] <= 0), 'Must have positive range for visualised attributes.'

        # Marginalisation is required if at least one non-visualised feature is *not* sliced.
        marginalise = any((not s for f,s in sliced.items() if f not in features))

        regions = {}  
        if not (attributes == [None] and edge_colour == None):                
            if not marginalise:
                # This is easy: can just use leaves directly.
                if n_f == 1: height = 1
                for nint, leaf in df.iterrows():
                    xy = []
                    for i, (f, lim) in enumerate(zip(features, axis_lims)):
                        f_min = max(leaf[f'{f} >'], lim[0])
                        f_max = min(leaf[f'{f} <'], lim[1])
                        xy.append(f_min)
                        if i == 0: width = f_max - f_min 
                        else:      height = f_max - f_min 
                    if n_f == 1: xy.append(0)
                    m = nint
                    regions[m] = {'xy':xy, 'width':width, 'height':height, 'alpha':1}  
                    for a, attr in enumerate(attributes_plus): 
                        if attr == None: continue
                        key = attr; val = leaf[attr]
                        regions[m][key] = val
            else:
                # Get all unique values mentioned in partitions for these features.
                f1 = features[0]
                p1 = np.unique(df[[f1+' >',f1+' <']].values) # Sorted by default.
                r1 = np.vstack((p1[:-1],p1[1:])).T # Ranges.
                if n_f == 2: 
                    f2 = features[1]
                    p2 = np.unique(df[[f2+' >',f2+' <']].values) 
                    r2 = np.vstack((p2[:-1],p2[1:])).T
                else: r2 = [[None,None]]
                # Iterate through the linear or rectangular grid created by all these partitions. 
                for m, ((min1, max1), (min2, max2)) in enumerate(tqdm(np.array([[i,j] for i in r1 for j in r2]), desc='Projecting hypercubes')):
                    min1 = max(min1, axis_lims[0][0])
                    max1 = min(max1, axis_lims[0][1])
                    width = max1 - min1
                    feature_ranges = {features[0]: [min1, max1]}
                    if n_f == 1: xy = [min1, 0]; height = 1
                    else: 
                        min2 = max(min2, axis_lims[1][0])
                        max2 = min(max2, axis_lims[1][1])
                        feature_ranges[features[1]] = [min2, max2]
                        xy = [min1, min2]
                        height = max2 - min2   
                    regions[m] = {'xy':xy, 'width':width, 'height':height, 'alpha':1}  
                    if attributes != [None]:           
                        # Find "core": the leaves that overlap with the feature range(s).
                        core = self.df_filter(df, features=feature_ranges, mode='overlap')
                        if 'overlap_filter' in core:
                            # Multiply overlaps through by the original overlaps of the filtering itself.
                            core['overlap'] = core.overlap_filter * core.overlap
                        for a, attr in enumerate(attributes_plus):
                            if attr == None: pass
                            elif attr == 'action' and self.classifier:
                                # Special case for action with classification: discrete values.
                                regions[m][attr] = np.argmax(np.dot(np.vstack(core['weighted_action_counts'].values).T, 
                                                                            core['overlap'].values.reshape(-1,1)))
                            else: 
                                if attr in ('sample_density','weight_density'): normaliser = 'volume' # Another special case for densities.
                                else:                                           normaliser = 'weight_sum'
                                # Take contribution-weighted mean across the core.
                                norm_sum = np.dot(core[normaliser].values, core['overlap'].values)
                                # NOTE: Averaging process assumes uniform data distribution within leaves.
                                core['contrib'] = core.apply(lambda row: (row[normaliser] * row['overlap']) / norm_sum, axis=1)
                                key = attr; vals = core[attr].values
                                regions[m][key] = np.nansum(vals * core['contrib'].values)          
        
        if visualise:     
            # If have filtered dataframe, assemble string to summarise this.
            filter_string = []
            # Rounding to 2 s.f. https://stackoverflow.com/a/48812729.
            round_sf = lambda x: '{:g}'.format(float('{:.{p}g}'.format(x, p=2)))
            for f, fil in filtered.items():
                if fil: 
                    rng = filter_spec[f]
                    if rng[0] == rng[1]:        
                        filter_string.append(f'{f}={round_sf(rng[0])}') # Slice.
                    else: filter_string.append(f'{f}$\in$[{round_sf(rng[0])},{round_sf(rng[1])}]') # Range.
            filter_string = ', '.join(filter_string)
                
            # If cmap midpoints not specified, use defaults.
            if cmap_midpoints == []: cmap_midpoints = [None for f in attributes]   
            # If doing density_shading, precompute alpha values.
            if density_shading:
                density_list = [r['sample_density'] for r in regions.values()]
                amin = np.nanmin(density_list)
                amax = np.nanpercentile(density_list, density_percentile) 
                alpha_norm = matplotlib.colors.LogNorm(vmin=amin, vmax=amax)
                for m, region in regions.items():
                    regions[m]['alpha'] = min(alpha_norm(region['sample_density']), 1)
            if n_f == 1:
                if axes != []: ax = axes
                else: _, ax = matplotlib.pyplot.subplots(); axes = ax
                if filter_string: ax.set_title(filter_string)
                ax.set_xlabel(features[0]); ax.set_xlim(axis_lims[0])
                ax.set_yticks(np.arange(n_a)+0.5)
                ax.set_yticklabels(attributes)
                ax.set_ylim([0,max(1,n_a)])
                ax.invert_yaxis()
                if density_shading:
                    # If doing density_shading, add background.
                    ax.add_patch(matplotlib.patches.Rectangle(xy=[axis_lims[0][0],0], width=axis_lims[0][1]-axis_lims[0][0], height=n_a, 
                                            facecolor='k', hatch=None, edgecolor=None, zorder=-11))
            else: offset = np.array([0,0])
            
            # Iterate through attributes.
            for a, attr in enumerate(attributes):
                if n_f == 1: offset = np.array([0,a])
                else:
                    if len(axes) <= a: 
                        axes.append(matplotlib.pyplot.subplots(figsize=(4.4,4))[1])
                    ax = axes[a]
                    # Assemble title.
                    title = ''
                    if attr: 
                        title += attr
                        if density_shading: title += ' (density shaded)'
                        if filter_string: title += '\n'+filter_string
                    elif filter_string: title = filter_string
                    ax.set_title(title)
                    ax.set_xlabel(features[0]); ax.set_xlim(axis_lims[0])
                    ax.set_ylabel(features[1]); ax.set_ylim(axis_lims[1])    
                    if density_shading:
                        # If doing density_shading, add background.
                        ax.add_patch(matplotlib.patches.Rectangle(xy=[axis_lims[0][0],axis_lims[1][0]], width=axis_lims[0][1]-axis_lims[0][0], height=axis_lims[1][1]-axis_lims[1][0], 
                                    facecolor='k', hatch=None, edgecolor=None, zorder=-11))
                if attr == None: pass
                elif attr == 'action' and self.classifier:
                    assert action_colours != None, 'Specify colours for discrete actions.'
                else: 
                    attr_list = [r[attr] for r in regions.values()]
                    if attr in ('sample_density','weight_density'):
                        # For density attributes, use a logarithmic cmap and clip at a specified percentile.
                        vmin = np.nanmin(attr_list)
                        vmax = np.nanpercentile(attr_list, density_percentile) 
                        colour_norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
                    else: 
                        vmin = np.nanpercentile(attr_list, cmap_percentiles[0])
                        vmax = np.nanpercentile(attr_list, cmap_percentiles[1])
                        vmid = cmap_midpoints[a]
                        if vmid != None: 
                            # Make symmetric.
                            half_range = max(vmax-vmid, vmid-vmin)
                            vmin = vmid - half_range; vmax = vmid + half_range
                        colour_norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=vmid)
                    dummy = ax.imshow(np.array([[vmin,vmax]]), aspect='auto', cmap=(cmaps['action'] if 'action' in attr else cmaps[attr]), norm=colour_norm)
                    dummy.set_visible(False)
                    if show_cbar:
                        if n_f == 1:
                            axins = inset_axes(ax,
                                            width='3%',  
                                            height=f'{(100/n_a)-1}%',  
                                            loc='lower left',
                                            bbox_to_anchor=(1.01, (n_a-a-1)/n_a, 1, 1),
                                            bbox_transform=ax.transAxes,
                                            borderpad=0,
                                            )
                            matplotlib.pyplot.colorbar(dummy, cax=axins)
                        else: cb = matplotlib.pyplot.colorbar(dummy, ax=ax)
                for m, region in regions.items():
                    if attr == None: colour = 'w'
                    elif attr == 'action' and self.classifier: colour = action_colours[region[attr]]
                    else: colour = (cmaps['action'] if 'action' in attr else cmaps[attr])(colour_norm(region[attr]))
                    # Don't apply alpha to density plots themselves.
                    if attr in ('sample_density','weight_density'): alpha = 1
                    else: alpha = region['alpha']
                    ax.add_patch(matplotlib.patches.Rectangle(
                                 xy=region['xy']+offset, width=region['width'], height=region['height'], 
                                 facecolor=colour, edgecolor=edge_colour, lw=0.1, alpha=alpha, zorder=-10))
                    # Add leaf integer.
                    if not marginalise and show_nints: 
                        ax.text(region['xy'][0]+region['width']/2, region['xy'][1]+region['height']/2, m, 
                                horizontalalignment='center', verticalalignment='center')
            
            # Some quick margin adjustments.
            #matplotlib.pyplot.tight_layout()
            if n_f == 1: matplotlib.pyplot.subplots_adjust(right=0.85)
            elif n_a == 1: axes = axes[0]
        return regions, axes


    def plot_transitions_2D(self, path, features=[], align=True,
                            alignment_iterations=1000, lr=1e-2,
                            ax=None, colour='k', alpha=1):
        """
        Given a sequence of transitions between leaves, plot on the specified feature axes.
        Optionally make use of derivatives to orient the path segments in realistic directions.
        """
        # Start by iteratively finding the nearest point in each leaf in the path.
        o = []; lims = []
        for i, (nint, cost) in enumerate(path): 
            lim = self.node_lims(nint)
            d_norm = self.node(nint).d_norm_mean
            lims.append(lim)
            if i > 0: 
                # NOTE: As a fallback, compute nearest point not from the current location, 
                # but from the centroid of the previous leaf.
                # This prevents repeated points which makes optimisation a nightmare.
                oi = self.closest_point_in_node(o[-1], nint)[0]
                if np.array_equal(oi, o[-1]):
                    o.append(self.closest_point_in_node(prev_centroid, nint)[0])
                else: o.append(oi)
            prev_centroid = np.mean(lim, axis=1)
            if i == 0: 
                oi = []
                for f in range(self.num_features):
                    if d_norm[f] > 0: oi.append(lim[f][0])
                    else: oi.append(lim[f][1])
                o.append(oi)
                                
        o = np.array(o); lims = np.array(lims)

        # Precompute some terms using the d_norm vectors.
        d_norm = np.array([self.node(nint).d_norm_mean for nint, _ in path[:-1]])
        d_norm_mag = np.linalg.norm(d_norm, axis=1, ord=2).reshape(-1,1) # Row-wise L2 norm.
    
        #self.plot_trajectory_2D(o, ax=ax, colour='b', alpha=alpha)

        if align:
            for iteration in range(int(alignment_iterations)):

                # Precompute some constituent terms.
                path_norm = (o[1:] - o[:-1]) * self.derivative_scales # Normalise path vectors.
                path_norm_mag = np.linalg.norm(path_norm, axis=1, ord=2).reshape(-1,1) # Row-wise L2 norm.
                dot_prod = np.einsum('ij,ij->i', path_norm, d_norm).reshape(-1,1) # Row-wise dot product.
                mag_prod = d_norm_mag * path_norm_mag # Product of L2 norms.
                cosine_sim = dot_prod / mag_prod # Row-wise cosine similarities. 
                angle = np.arccos(cosine_sim)

                # Calculate partial derivatives of loss with respect to each feature of each point.
                # A = (1 / (path_norm_mag * d_norm_mag))
                # B = (dot_prod / (path_norm_mag ** 2))
                # cosine_diff = 1 - (dot_prod * A) # NOTE: Experiment!
                # derivative_terms = np.pad(cosine_diff.reshape(-1,1) * A.reshape(-1,1) * (d_norm - (path_norm * B.reshape(-1,1))), ((1,1),(0,0)))
                # derivative_terms = A.reshape(-1,1) * (d_norm - (path_norm * B.reshape(-1,1)))
                #derivative_terms = np.pad(angle * ((d_norm/mag_prod) - (path_norm*cosine_sim/(path_norm_mag**2))),
                derivative_terms = np.pad(((d_norm/mag_prod) - (path_norm*cosine_sim/(path_norm_mag**2))),
                                          #/ ((1-(cosine_sim**2))**0.5),
                                          ((1,1),(0,0))) # Padding required to handle first and last segments.
                # derivative_terms = np.pad(-angle * path_norm / (path_norm_mag**2), ((1,1),(0,0))) # Padding required to handle first and last segments.
                dl_do_norm = derivative_terms[1:] - derivative_terms[:-1]

                # Mask out invalid derivatives.     
                # If at lower lim, cannot move down along this feature (vice versa for upper/up).
                at_lower_lim = o <= lims[:,:,0]
                at_upper_lim = o >= lims[:,:,1]
                at_lim = np.logical_or(at_lower_lim, at_upper_lim)
                # If on a face normal to the feature direction, cannot move *either* up or down.
                on_face = at_lim * (at_lim.sum(axis=1) == 1).reshape(-1,1)
                # Apply mask.
                up = dl_do_norm < 0
                dl_do_norm *= np.logical_not(on_face) \
                            * np.logical_not(np.logical_not(up) * at_lower_lim) \
                            * np.logical_not(up * at_upper_lim)

                # Now iterate through points in order.
                o_prev = o.copy()
                for i, update in enumerate(dl_do_norm):

                    # print(path[i][0])
                    # print('update before', update)

                    if i > 0:
                        # Prevent any movement that would render this point "invisible" from the previous one.
                        # This stops the trajectory crossing through a leaf before its respective point.
                        # Can move normal to a direction if either its lower or upper face is visible,
                        # and the point is currently on that face.
                        can_move_normal_to = np.logical_or((o[i-1] < lims[i,:,0]) * at_lower_lim[i], 
                                                           (o[i-1] > lims[i,:,1]) * at_upper_lim[i])
                        # print('lims', lims[i])
                        # print('prev', o[i-1])
                        # print('at lower', at_lower_lim[i])
                        # print('can move normal to', can_move_normal_to)
                    else:
                        can_move_normal_to = np.ones(self.num_features).astype(bool)

                    # Must have >= 1 zero component in a "can_move_normal_to" direction.
                    # Find the smallest component in one of these directions, and set it to zero.
                    if can_move_normal_to.sum() == 0: 
                        # print(f"***ITERATION {iteration} COLLAPSE?***")
                        can_move_normal_to = np.ones(self.num_features).astype(bool)
                    mag = np.abs(update) 
                    update *= mag != mag[can_move_normal_to].min()

                    # print('update final', update)
                    # print()

                    # Perform gradient update, remembering to divide through by derivative_scales.
                    # NOTE: OR multiply?
                    o[i] = np.clip(o[i] - (lr * update / self.derivative_scales),
                                   lims[i,:,0], lims[i,:,1]) # Clip to leaf boundaries.

                #self.plot_trajectory_2D(o, ax=ax, colour=colour, alpha=alpha)
                #matplotlib.pyplot.pause(0.001)
                # print(np.linalg.norm(o-o_prev))

        return self.plot_trajectory_2D(o, features=features, ax=ax, colour=colour, alpha=alpha)


    def plot_trajectory_2D(self, o, features=[], ax=None, colour='k', alpha=1):
        """
        Plot a trajectory of observations in a two-feature plane.
        """
        o = np.array(o)
        n_f = o.shape[1]
        if n_f != 2:
            assert n_f == self.num_features, 'Data shape must match num_features.'
            assert len(features) == 2, 'Need to specify two features for projection.'
            # This allows plotting of a projected path onto two specified features.
            o = o[:,[self.feature_names.index(f) for f in features]]   
        if ax == None: _, ax = matplotlib.pyplot.subplots()
        ax.plot(o[:,0], o[:,1], c=colour, alpha=alpha)
        #ax.scatter(o[0,0], o[0,1], c='g', alpha=alpha, zorder=10)
        #ax.scatter(o[-1,0], o[-1,1], c='r', alpha=alpha, zorder=10)
        #ax.scatter(o[1:-1,0], o[1:-1,1], c=colour, alpha=alpha, s=10, zorder=10)
        return o, ax


    def cf_scatter_regret(self, features, indices=None, lims=[], ax=None):
        """
        Create a scatter plot showing all counterfactual samples,
        coloured by their regret.
        """
        if not ax: _, ax = matplotlib.pyplot.subplots()
        assert len(features) == 2, 'Can only plot in 1 or 2 dimensions.'
        # If indices not specified, use all.
        if indices == None: indices = np.arange(len(self.cf.o))
        indices = indices[~np.isnan(self.cf.regret[indices])] # Ignore NaNs.
        o, regret = self.cf.o[indices], self.cf.regret[indices]
        # NOTE: Scaling by percentiles.
        upper_perc = np.percentile(regret, 95)
        lower_perc = np.percentile(regret, 5)
        perc_range = upper_perc - lower_perc
        # Set lims.
        if lims == []:
            # If lims not specified, use global lims across dataset.
            fi = [self.feature_names.index(f) for f in features]
            lims = np.vstack((np.min(self.o[:,fi], axis=0), np.max(self.o[:,fi], axis=0))).T
        ax.set_xlim(lims[0]); ax.set_ylim(lims[1])    
        # Define colours.
        dummy = ax.imshow(np.array([[lower_perc,upper_perc]]), aspect='auto', cmap='Reds')
        dummy.set_visible(False)
        matplotlib.pyplot.colorbar(dummy, ax=ax, orientation='horizontal') 
        colours = matplotlib.matplotlib.cm.Reds((regret - lower_perc) / perc_range)
        # Plot.
        ax.scatter(o[:,0], o[:,1], s=0.5, color=colours)
        return ax


    def plot_leaf_derivatives_2D(self, features=[], ax=None, 
                                 colour='k', lengthscale=1, linewidth=0.02, try_reuse_df=True):
        """
        Visualise leaves' mean derivative vectors using arrows.
        """
        if not(try_reuse_df and hasattr(self, 'df')): self.to_dataframe()
        if len(features) != 2: 
            assert self.num_features == 2; features = self.feature_names
        # List the columns to query from the dataframe.
        cols = []
        for f in features: cols += [f+' >']
        for f in features: cols += [f+' <']
        lims = np.stack(np.split(self.df.loc[self.leaf_nints][cols].values, 2, axis=1))
        centroids = np.mean(lims, axis=0) # Position arrows at leaf centroids.
        # Get mean derivative of specified features for each leaf.
        fi = np.array([self.feature_names.index(f) for f in features])
        d_mean = np.array([self.node(nint).derivative_mean[fi] for nint in self.leaf_nints])
        # Plot arrows.
        if ax == None: _, ax = matplotlib.pyplot.subplots()
        matplotlib.pyplot.quiver(centroids[:,0], centroids[:,1], d_mean[:,0], d_mean[:,1], 
                                 pivot='tail', angles='xy', scale_units='xy', units='inches', 
                                 color=colour, scale=1/lengthscale, width=linewidth, minshaft=1)
        return ax


    def display_counterfactual(self, o, o_foil, attributes=[None], **kwargs):
        """
        Display a fact and foil sample on a treemap visualisation.
        Only works if the two differ in 1 or 2 features.
        """
        # Determine which features are different between fact and foil.
        diff = [fi for fi,d in enumerate(o_foil-o != 0) if d]
        assert len(diff) in (1,2), 'Fact and foil must differ in 1 or 2 features.'
        
        # Visualise on corresponding axes.
        conditions = [o[fi] if fi not in diff else None for fi in range(self.num_features)]
        
        _, ax = self.treemap(np.array(self.feature_names)[diff], attributes=attributes, conditions=conditions, **kwargs)
        pts = np.vstack((o, o_foil)).T[diff]
        # If just one feature is different, visualise in 1D.
        if pts.shape[0] == 1: pts = np.pad(pts, ((0,1),(0,0)), constant_values=0.5)
        ax.plot(pts[0], pts[1])
        ax.scatter(pts[0,0],pts[1,0], c='g', zorder=10)
        ax.scatter(pts[0,1],pts[1,1], c='r', zorder=10)


# ===================================================================================================================
# NODE CLASS.


class Node():
    def __init__(self, 
                 nint, 
                 num_samples, 
                 indices, 
                 ):
        # These are the basic attributes; more are added elsewhere.
        self.nint = nint
        self.indices = indices
        self.num_samples = num_samples
        self.left = None
        self.right = None

    
# ===================================================================================================================
# CLASS FOR HOLDING COUNTERFACTUAL DATA.


class counterfactual(): 
    def __init__(self): 
        self.num_samples = 0 # Initially dataset is empty.
        self.regret_steps = None