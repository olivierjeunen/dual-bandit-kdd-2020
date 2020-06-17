import numpy as np
from numpy.random.mtrand import RandomState

from agents import AbstractFeatureProvider, ViewsFeaturesProvider, Model, ModelBasedAgent
from reco_gym import Configuration

organic_user_count_args = {
    'num_products': 10,
    'random_seed': np.random.randint(2 ** 31 - 1),

    # Select a Product randomly with the highest probability for the most frequently viewed product.
    'select_randomly': True,

    # Weight History Function: how treat each event back in time.
    'weight_history_function': None,

    # reverse popularity.
    'reverse_pop': False,

    # Epsilon-greedy - if none-zero, this ensures the policy has support over all products
    'epsilon': .0
}

#/EXPERIMENTAL#
def fast_choice(n_options, probs, rng):
    # Numpy.random.choice is fast when vectorised, but we call it for single choices
    # This very simple algorithm is faster for a small number of options (empirically - < 150)
    # Generate a random number
    pchoice = rng.random_sample()
    # Get the probablity for the first option
    running_sum = probs[0]
    running_choice = 0
    # Loop over options
    for p in probs[1:]:
        # If our random number is bigger than the sum of preceding probabilities
        # Pick this one
        if pchoice <= running_sum:
            break
        else:
            running_sum += p
            running_choice += 1
    # Machine precision issues make that the probabilities sometimes do not sum exactly to one
    # If this becomes an issue, divide the remaining probability mass over all items
    if running_choice >= n_options:
        return fast_choice(n_options, [1/n_options]*n_options, rng)
    return running_choice

from numba import jit
@jit(nopython=True)
def numba_fast_choice(probs, p):
    return np.searchsorted(probs.cumsum(),p)
#/EXPERIMENTAL#

class OrganicUserEventCounterModelBuilder(AbstractFeatureProvider):
    def __init__(self, config):
        super(OrganicUserEventCounterModelBuilder, self).__init__(config)

    def build(self):

        class OrganicUserEventCounterModel(Model):
            """
            Organic Event Count Model (per a User).
            """

            def __init__(self, config):
                super(OrganicUserEventCounterModel, self).__init__(config)
                if config.select_randomly:
                    self.rng = RandomState(self.config.random_seed)

            def act(self, observation, features):
                # Preparations for epsilon-greedy
                if self.config.epsilon > 0:
                    # Compute current mass       
                    sum_features = np.sum(features, axis = 0)
                    # Get non-zero features
                    mask = features == 0
                    # Rescale to (1 - eps) % of the mass
                    features[~mask] = (1.0 - self.config.epsilon) * sum_features
                    # Uniformly redistribute eps % of the mass
                    features += self.config.epsilon * sum_features

                if not self.config.reverse_pop:
                    action_proba = features / np.sum(features, axis = 0)
                else:
                    action_proba = 1 - features / np.sum(features, axis = 0)
                    action_proba = action_proba/sum(action_proba)                    
                if self.config.select_randomly:
                    #action = self.rng.choice(self.config.num_products, p = action_proba)
                    #action = fast_choice(self.config.num_products, action_proba, self.rng)
                    action = numba_fast_choice(action_proba, self.rng.random_sample())
                    ps = action_proba[action]
                    ps_all = action_proba
                else:
                    action = np.argmax(action_proba)
                    ps = 1.0
                    ps_all = np.zeros(self.config.num_products)
                    ps_all[action] = 1.0
                return {
                    **super().act(observation, features),
                    **{
                        'a': action,
                        'ps': ps,
                        'ps-a': ps_all,
                    },
                }

        return (
            ViewsFeaturesProvider(self.config),
            OrganicUserEventCounterModel(self.config)
        )


class OrganicUserEventCounterAgent(ModelBasedAgent):
    """
    Organic Event Counter Agent

    The Agent that counts Organic views of Products (per a User)
    and selects an Action for the most frequently shown Product.
    """

    def __init__(self, config = Configuration(organic_user_count_args)):
        super(OrganicUserEventCounterAgent, self).__init__(
            config,
            OrganicUserEventCounterModelBuilder(config)
        )
