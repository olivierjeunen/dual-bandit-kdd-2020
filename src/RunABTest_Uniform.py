import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agents import OrganicUserEventCounterAgent, organic_user_count_args
from agents import SkylineAgent, skyline_args
from agents import RandomAgent, random_args
from reco_gym import (
    Configuration,
    build_agent_init,
    env_1_args,
    gather_agent_stats,
    plot_agent_stats
)

from agents import PyTorchMLRAgent, pytorch_mlr_args
from datetime import datetime

# General parameters for experiments
RandomSeed = 42
TrainingDataSamples = [1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000]
TestingDataSamples = 10000
StatEpochs = 5
StatEpochsNewRandomSeed = True

def grid_search_parameter(
        parameter_name, parameter_values,
        agent_name, agent_pytorch_mlr_args,
        env_args, extra_env_args,
        train_samples=5000, valid_samples=10000, valid_runs=5,
):
    ''' Perform a grid search by doing an A/B test with model variants that use different parameter values, log the rsult and return the optimal parameter value '''
    print(f'\tGrid search for {agent_name}')
    grid_search_agent_inits = dict()
    for parameter_value in parameter_values:
        agent_name_with_specified_parameter = f'{agent_name} ({parameter_name} = {parameter_value})'
        grid_search_agent_inits.update(
            build_agent_init(
                agent_name_with_specified_parameter,
                PyTorchMLRAgent,
                {
                    **agent_pytorch_mlr_args,
                    parameter_name: parameter_value
                }
            )
        )

    ### Run A/B test with model variants
    agent_stats = gather_agent_stats(
        env,
        env_args,
        {
            **extra_env_args
        },
        grid_search_agent_inits,
        [train_samples],
        valid_samples,
        valid_runs,
        StatEpochsNewRandomSeed
    )

    # Messy way to get everything into pandas, but it works 
    gg = agent_stats[list(agent_stats.keys())[1]]
    l = []
    for k in list(gg.keys()):
        f = pd.DataFrame(gg[k])
        f['Alg'] = k
        f['Samples'] = agent_stats[list(agent_stats.keys())[0]]
        l.append(f)
    agent_stats_df = pd.concat(l)
    agent_stats_df.columns = [str(c) for c in agent_stats_df.columns]
    values = agent_stats_df['AgentStats.Q0_500'].values
    optimal_parameter_value = parameter_values[np.argmax(values)]
    print(agent_stats_df)
    agent_stats_df.to_csv(f'GS_{agent_name.replace(" ", "_")}_uniform.csv', index=False)
    print(f'\t... optimal {parameter_name} = {optimal_parameter_value}!')
    return optimal_parameter_value

for num_products in [10, 25, 100]:
    print('#' * 48)
    print(f'{datetime.now()}\tLogging Uniform with {num_products} products...')
    print('#' * 48)

    # Environment parameters
    std_env_args = {
        **env_1_args,
        'random_seed': RandomSeed,
        'num_products': num_products,
    }
    env = gym.make('reco-gym-v1')

    # Logging policy
    logger = RandomAgent(Configuration({**random_args,**std_env_args}))

    std_extra_env_args = {
        'num_products': num_products,
        'number_of_flips': num_products // 2,
        'agent': logger,
    }

    ###################################################
    # Grid searches for POEM and Dual Bandit variants #
    ###################################################

    # Original POEM
    lambdas_poem_no_log = [.0, .05, .1, .25, .5, 1.0, 2.0]
    poem_no_log_mlr_args = {
        **pytorch_mlr_args,
        'logIPS': False,
    }
    optimal_svp_strength_nolog = grid_search_parameter(
        'variance_penalisation_strength', lambdas_poem_no_log,
        'POEM no log', poem_no_log_mlr_args,
        std_env_args, std_extra_env_args,
    )

    # Logarithmic POEM
    lambdas_poem_log = [.0, .05, .1, .25, .5, 1.0, 2.0]
    poem_log_mlr_args = {
        **pytorch_mlr_args,
        'logIPS': True,
    }
    optimal_svp_strength_withlog = grid_search_parameter(
        'variance_penalisation_strength', lambdas_poem_log,
        'POEM log', poem_log_mlr_args,
        std_env_args, std_extra_env_args,
    )

    # Dual Bandit
    alphas_dual_bandit_no_log = [.0, .80, .85, .90, .925, .95, .975, .99, .999, .9999, 1.0]
    dual_bandit_no_log_mlr_args = {
        **pytorch_mlr_args,
        'logIPS': False,
        'll_IPS': False,
    }
    optimal_ll_strength_nolog = grid_search_parameter(
        'alpha', alphas_dual_bandit_no_log,
        'Dual Bandit no log', dual_bandit_no_log_mlr_args,
        std_env_args, std_extra_env_args,
    )

    # Logarithmic Dual Bandit
    alphas_dual_bandit_log = [.0, .80, .85, .90, .925, .95, .975, .99, .999, .9999, 1.0]
    dual_bandit_log_mlr_args = {
        **pytorch_mlr_args,
        'logIPS': True,
        'll_IPS': False,
    }
    optimal_ll_strength_withlog = grid_search_parameter(
        'alpha', alphas_dual_bandit_log,
        'Dual Bandit log', dual_bandit_log_mlr_args,
        std_env_args, std_extra_env_args,
    )

    # Initialisation of different agents
    logging_agent = build_agent_init(
        'Logging',
        RandomAgent,
        {
            **random_args,
            **std_env_args,
        }
    )

    skyline_agent = build_agent_init(
        'Skyline',
        SkylineAgent,
        {
            **skyline_args,
        }
    )

    likelihood_agent = build_agent_init(
        'Likelihood',
        PyTorchMLRAgent,
        {
            **pytorch_mlr_args,
            'll_IPS': False,
            'alpha': 1.0
        }
    )

    ips_likelihood_agent = build_agent_init(
        'IPS Likelihood',
        PyTorchMLRAgent,
        {
            **pytorch_mlr_args,
            'll_IPS': True,
            'alpha': 1.0
        }
    )

    cb_no_log_agent = build_agent_init(
        'Contextual Bandit - no log',
        PyTorchMLRAgent,
        {
            **pytorch_mlr_args,
            'logIPS': False
        }
    )

    cb_log_agent = build_agent_init(
        'Contextual Bandit - with log',
        PyTorchMLRAgent,
        {
            **pytorch_mlr_args,
            'logIPS': True
        }
    )

    poem_no_log_agent = build_agent_init(
        'POEM (lambda = {0} - no log)'.format(optimal_svp_strength_nolog),
        PyTorchMLRAgent,
        {
            **pytorch_mlr_args,
            'logIPS': False,
            'variance_penalisation_strength': optimal_svp_strength_nolog
        }
    )

    poem_log_agent = build_agent_init(
        'POEM (lambda = {0} - with log)'.format(optimal_svp_strength_withlog),
        PyTorchMLRAgent,
        {
            **pytorch_mlr_args,
            'logIPS': True,
            'variance_penalisation_strength': optimal_svp_strength_withlog
        }
    )

    dual_bandit_no_log_agent = build_agent_init(
        'Dual Bandit (alpha = {0} - no log)'.format(optimal_ll_strength_nolog),
        PyTorchMLRAgent,
        {
            **pytorch_mlr_args,
            'logIPS': False,
            'll_IPS': False,
            'alpha': optimal_ll_strength_nolog
        }
    )

    dual_bandit_log_agent = build_agent_init(
        'Dual Bandit (alpha = {0} - with log)'.format(optimal_ll_strength_withlog),
        PyTorchMLRAgent,
        {
            **pytorch_mlr_args,
            'logIPS': True,
            'll_IPS': False,
            'alpha': optimal_ll_strength_withlog
        }
    )

    agent_inits = {
        **logging_agent,
        **skyline_agent,
        **likelihood_agent,
        **ips_likelihood_agent,
        **cb_no_log_agent,
        **cb_log_agent
    }

    if optimal_svp_strength_nolog != 0:
        agent_inits.update(poem_no_log_agent)

    if optimal_svp_strength_withlog != 0:
        agent_inits.update(poem_log_agent)

    if optimal_ll_strength_nolog != 0 and optimal_ll_strength_nolog != 1:
        agent_inits.update(dual_bandit_no_log_agent)

    if optimal_ll_strength_withlog != 0 and optimal_ll_strength_withlog != 1:
        agent_inits.update(dual_bandit_log_agent)

    # Gathering performance of agents for the logging policy: uniform.
    agent_stats01 = gather_agent_stats(
        env,
        std_env_args,
        {
            'num_products': num_products,
            'number_of_flips': num_products // 2,
            'agent': logger
        },
        agent_inits,
        TrainingDataSamples,
        TestingDataSamples,
        StatEpochs,
        StatEpochsNewRandomSeed
    )

    plot_agent_stats(agent_stats01, figname='performance_comparison_{0}_products_uniform.png'.format(num_products))

    def tocsv(stats, file_name):
        import pandas as pd
        gg = stats[list(stats.keys())[1]]

        l = []
        for k in list(gg.keys()):
            f = pd.DataFrame(gg[k])
            f['Alg'] = k.replace('aa', 'a').replace('bb', 'b')
            f['Samples'] = stats[list(stats.keys())[0]]
            l.append(f)
        pd.concat(l).to_csv(file_name)

    tocsv(agent_stats01, 'performance_comparison_{0}_products_uniform.csv'.format(num_products))
