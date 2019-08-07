'''
This module is for running Bayesian meta analyses based upon studies reporting correlations.
More specifically, the input data required is:
- R: a vector of reported correlation coefficients
- N: a vector of corresponding reported sample sizes

The first core point to make this work as a Bayesian meta analysis is that we
can convert observed (R, N) into z-transformed versions of R by using
Fisher's transformation https://en.wikipedia.org/wiki/Fisher_transformation.

Secondly, when we do this, we can also get an estimate of the variance of the
correlation coefficient (in z space), which is 1/sqrt(N-3).
'''

import numpy as np
import pymc3 as pm
import pandas as pd
from statsmodels.nonparametric.kde import KDEUnivariate
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#from .plot import plot
from .plot import forest_plot, bayes_factor_plot
from .funcs import _flip_effect_size_dir, _add_results_to_df, _print_meta_analytic_result, get_population_level_stats


def fit(data, sample_options=None, prior='stretched beta', type='random effects'):
    '''Conduct parameter estimation'''

    if sample_options is None:
        sample_options = {'tune': 5_000, 'draws': 5_000,
                          'chains': 4, 'cores': 2,
                          'nuts_kwargs': {'target_accept': 0.97},
                          'random_seed': 12345}

    # data preprocessing -------------------------------------------------

    # duplicate the specfic effect size column into a generic one
    data['effect_size'] = data['R']

    # reverse the effect size (if appropriate)
    if 'reverse_effect_size' in data.columns:
        data['effect_size'] = data.apply(
            lambda row: _flip_effect_size_dir(row), axis=1)

    # get observed data
    data = _remove_any_missing_data(data)
    n_obs, n_studies, z_obs = _extract_data_for_PyMC3(data)

    # construct the model ------------------------------------------------

    # use prior as specified by user with `prior` kwarg
    if prior == 'stretched beta':
        print('Prior over population effect size: Stretched Beta')
        with pm.Model() as model:
            r_pop = pm.Beta('r_pop', 2, 2)
            effect_size_population = pm.Deterministic('effect_size_population', (r_pop*2)-1)

    elif prior == 'uniform':
        print('Prior over population effect size: Uniform')
        with pm.Model() as model:
            effect_size_population = pm.Uniform('effect_size_population', -1, 1)

    else:
        raise Exception('Supplied value for prior not recognised')

    # define the rest of the model
    if type == 'random effects':
        with model:

            # study_sigma = pm.HalfCauchy('study_sigma', study_sigma_cauchy_prior_width)
            study_sigma = pm.HalfNormal('study_sigma', sigma=0.5)

            z_study = pm.Normal('z_study',
                                mu=_fisher_transformation(effect_size_population),
                                sd=study_sigma, shape=n_studies)
            # Standard error is 1/sqrt(N-3)
            # Variance is 1/(N-3)
            # Standard deviation is sqrt(1/(N-3)) = 1/sqrt(N-3)
            z_obs = pm.Normal('z_obs', mu=z_study,
                              sd=pm.math.sqrt(1/(n_obs-3)),
                              observed=z_obs)

    elif type == 'fixed effects':
        with model:
            # Standard error is 1/sqrt(N-3)
            # Variance is 1/(N-3)
            # Standard deviation is sqrt(1/(N-3)) = 1/sqrt(N-3)
            z_obs = pm.Normal('z_obs', mu=_fisher_transformation(effect_size_population),
                              sd=pm.math.sqrt(1/(n_obs-3)),
                              observed=z_obs)


    # run MCMC sampling --------------------------------------------------
    with model:
        print('Sampling from prior')
        prior = pm.sample_prior_predictive(
            100_000, random_seed=sample_options['random_seed'])
        print('Sampling from posterior')
        posterior = pm.sample(**sample_options)

    # post sampling activity ---------------------------------------------

    # get effect-size specific stuff with a local function here
    if type == 'random effects':
        effect_size_study_mean, effect_size_study_hdi = get_study_level_stats(
            posterior)
        data['effect_size_est_mean'] = effect_size_study_mean
        data['effect_size_est_HDI_lower'] = effect_size_study_hdi[:, 0]
        data['effect_size_est_HDI_upper'] = effect_size_study_hdi[:, 1]

    # now use an effect-size independent function to instert into df
    data = _add_results_to_df(data, posterior)


    _print_meta_analytic_result(posterior)

    return data, prior, posterior, model


''' Functions specific to effect size (Pearson's correlation coefficient) '''


def _fisher_transformation(r):
    '''Apply the Fisher transformation. Converts from r to z.'''
    return 0.5*np.log((1+r)/(1-r))


def _inverse_fisher_transformation(z):
    '''Apply the inverse Fisher transformation. Converts from z to r.'''
    return (np.exp(2*z)-1) / (np.exp(2*z)+1)


def _extract_data_for_PyMC3(data):
    r_obs = data['effect_size'].values
    n_obs = data['N'].values
    assert len(r_obs) == len(n_obs)
    n_studies = len(r_obs)
    z_obs = _fisher_transformation(r_obs)
    return (n_obs, n_studies, z_obs)


def _remove_any_missing_data(data):
    n_missing_effect_sizes = sum(data.R.isnull())
    n_missing_sample_sizes = sum(data.N.isnull())
    if n_missing_effect_sizes > 0 or n_missing_sample_sizes > 0:
        data = data[pd.notnull(data['R'])]
        data = data[pd.notnull(data['N'])]
        print('WARNING: We detected rows with missing effect sizes (R) or sample sizes (N). These have been removed!')
    return data


def get_study_level_stats(trace):
    effec_size_study = _inverse_fisher_transformation(trace['z_study'])
    effec_size_study_mean = np.mean(effec_size_study, axis=0)
    effec_size_study_hdi = pm.stats.hpd(effec_size_study)
    return (effec_size_study_mean, effec_size_study_hdi)










def plot(data, prior, posterior, sort_by=None, figsize=(10, 12), height_ratios=[3, 1]):
    plt.rcParams.update({'font.size': 12})
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=height_ratios)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    ax1 = forest_plot(ax1, data, sort_by)
    ax1.set(xlabel=None)
    ax1 = format_effect_size_axis(ax1)

    ax2 = bayes_factor_plot(ax2, prior, posterior)
    ax2 = format_effect_size_axis(ax2)

    fig.tight_layout()
    return [ax1, ax2]


def format_effect_size_axis(ax):
    '''format the effect size axis appropriately'''
    ax.set(xlabel='Pearson correlation coefficient',
           xlim=[-1, 1],
           xticks=np.arange(-1, 1.01, 0.2),)
    return ax
