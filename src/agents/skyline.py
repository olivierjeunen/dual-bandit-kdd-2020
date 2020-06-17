import numpy as np
from numpy.random.mtrand import RandomState

from agents import AbstractFeatureProvider, ViewsFeaturesProvider, Model, ModelBasedAgent
from reco_gym import Configuration

skyline_args = {
    'num_products': 10
}

class SkylineModelBuilder(AbstractFeatureProvider):
    def __init__(self, config):
        super(SkylineModelBuilder, self).__init__(config)

    def build(self):

        class SkylineModel(Model):
            def __init__(self, config):
                super(SkylineModel, self).__init__(config)
                self._true_pclick = np.zeros(config.num_products)

            def act(self, observation, features):
                # Always take the optimal action
                print(self._true_pclick)
                action = np.argmax(self._true_pclick)
                ps_all = np.zeros(self._true_pclick.shape)
                ps_all[action] = 1.0
                return {
                    **super().act(observation, features),
                    **{
                        'a': action,
                        'ps': 1.0,
                        'ps-a': ps_all,
                    },
                }

        return (
            ViewsFeaturesProvider(self.config),
            SkylineModel(self.config)
        )

class SkylineAgent(ModelBasedAgent):
    def __init__(self, config = Configuration(skyline_args)):
        super(SkylineAgent, self).__init__(
            config,
            SkylineModelBuilder(config)
        )
        self.skyline = True
