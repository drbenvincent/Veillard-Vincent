''' Helper functions (independent of type of Effect Size) '''


from statsmodels.nonparametric.kde import KDEUnivariate
import numpy as np
import pandas as pd
import pymc3 as pm


def calc_bayes_factor(prior_samples, posterior_samples, x=0):
    '''Returns the Bayes Factor (BF01) such that values >1 indicate there is 
    more support for `x` under the posterior, relative to the prior.
    '''
    kde = KDEUnivariate(prior_samples)
    kde.fit()
    prior_density_at_zero = kde.evaluate([x])

    kde = KDEUnivariate(posterior_samples)
    kde.fit()
    posterior_density_at_zero = kde.evaluate([x])
    
    BF_prior_post = prior_density_at_zero/posterior_density_at_zero
    return BF_prior_post[0]


def _flip_effect_size_dir(row):
    if row['reverse_effect_size'] is True:
        return -row['effect_size']
    else:
        return row['effect_size']


def _add_results_to_df(data, trace):
    # Extract study level stats and place in the dataframe
    # effect_size_study_mean, effect_size_study_hdi = get_study_level_stats(
    #     trace)
    # data['effect_size_est_mean'] = effect_size_study_mean
    # data['effect_size_est_HDI_lower'] = effect_size_study_hdi[:, 0]
    # data['effect_size_est_HDI_upper'] = effect_size_study_hdi[:, 1]

    # Create new row for the meta-analytic result
    effect_size_population_mean, effect_size_population_hdi = get_population_level_stats(
        trace)
    info = {'study': ['Meta-analytic result'],
            'effect_size_est_mean': [effect_size_population_mean],
            'effect_size_est_HDI_lower': [effect_size_population_hdi[0]],
            'effect_size_est_HDI_upper': [effect_size_population_hdi[1]]}
    row = pd.DataFrame.from_dict(info)
    # Append new row to dataframe, corresponding to the meta-analytic estimate
    data = data.append(row, sort=False).reset_index(drop=True)
    return data


def _print_meta_analytic_result(trace):
    effect_size_population_mean, effect_size_population_hdi = get_population_level_stats(
        trace)
    print(
        f'Estimated true correlation coefficient:\n\t{effect_size_population_mean:.3f} [95% HDI: {effect_size_population_hdi[0]:.3f}, {effect_size_population_hdi[1]:.3f}]')
    return


def get_population_level_stats(trace):
    '''Get the stats of our meta-analytic result'''
    effect_size_population_mean = np.mean(trace['effect_size_population'])
    effect_size_population_hdi = pm.stats.hpd(trace['effect_size_population'])
    return (effect_size_population_mean, effect_size_population_hdi)
