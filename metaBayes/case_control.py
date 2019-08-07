'''
This module is for running Bayesian meta analyses based upon studies with
a case-control design.

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


def fit(data, sort_by=None, sample_options=None):
    '''Conduct parameter estimation'''

    if sample_options is None:
        sample_options = {'tune': 2_000, 'draws': 10_000,
                          'chains': 4, 'cores': 4,
                          'nuts_kwargs': {'target_accept': 0.95},
                          'random_seed': 1234}

    # data preprocessing -------------------------------------------------
    
    # duplicate the specfic effect size column into a generic one
    data['effect_size'] = data['cohen_d']

    # compute variance on effect size, add as new column
    data['effect_size_std'] = data.apply(
        lambda row: _calc_effect_size_std(row), axis=1)

    # reverse the effect size (if appropriate)
    if 'reverse_effect_size' in data.columns:
        data['effect_size'] = data.apply(
            lambda row: _flip_effect_size_dir(row), axis=1)

    # get observed data
    data = _remove_any_missing_data(data)
    n_studies, d_obs, study_sd = _extract_data_for_PyMC3(data)

    # construct the model ------------------------------------------------

    with pm.Model() as model:

        effect_size_population = pm.Cauchy('effect_size_population', 0, 1)
        study_sigma = pm.Uniform('study_sigma', 0, 10)

        d_study = pm.Normal('d_study', mu=effect_size_population,
                            sd=study_sigma, shape=n_studies)

        d_obs = pm.Normal('d_obs', mu=d_study, sd=study_sd, observed=d_obs)

    # run MCMC sampling ---------------------------------------------
    with model:
        print('Sampling from prior')
        prior = pm.sample_prior_predictive(
            10_000, random_seed=sample_options['random_seed'])
        print('Sampling from posterior')
        posterior = pm.sample(**sample_options)

    # post sampling activity ----------------------------------------
    
    # get effect-size specific stuff with a local function here
    effect_size_study_mean, effect_size_study_hdi = get_study_level_stats(
        posterior)
    data['effect_size_est_mean'] = effect_size_study_mean
    data['effect_size_est_HDI_lower'] = effect_size_study_hdi[:, 0]
    data['effect_size_est_HDI_upper'] = effect_size_study_hdi[:, 1]
    
    # now use an effect-size independent functinon to instert into df
    data = _add_results_to_df(data, posterior)
    

    _print_meta_analytic_result(posterior)

    return data, prior, posterior


''' Functions specific to effect size (Cohen's D) '''


def _extract_data_for_PyMC3(data):
    d_obs = data['effect_size'].values
    study_sd = data['effect_size_std'].values
    n_studies = len(d_obs)
    return (n_studies, d_obs, study_sd)


def _remove_any_missing_data(data):
    n_missing_effect_sizes = sum(data.cohen_d.isnull())
    if n_missing_effect_sizes > 0:
        data = data[pd.notnull(data['cohen_d'])]
        print('WARNING: We detected rows with missing effect sizes (cohen_d). These have been removed!')
    return data


def _calc_effect_size_std(row):
    '''Calculate variance of the Cohen's D effect size
    Based on Hedges & Olkin (1985, p.86)'''
    n_t = row['n_test']
    n_c = row['n_control']
    cohen_d = row['effect_size']
    return ((n_t+n_c)/(n_t*n_c)) + (cohen_d/(2*(n_t+n_c-2)))


def get_study_level_stats(trace):
    D_study = trace['d_study']
    D_study_mean = np.mean(D_study, axis=0)
    D_study_hdi = pm.stats.hpd(D_study)
    return (D_study_mean, D_study_hdi)






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

    # keep x-axis zoomed on what we do in ax1
    xlim = ax1.get_xlim()
    ax2.set_xlim(xlim)

    fig.tight_layout()
    return [ax1, ax2]


def format_effect_size_axis(ax):
    '''format the effect size axis appropriately'''
    ax.set(xlabel="Cohen's D")
    #xlim=[-2.5, 2.5])
    return ax
