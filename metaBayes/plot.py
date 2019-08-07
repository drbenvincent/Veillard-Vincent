import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.nonparametric.kde import KDEUnivariate
import numpy as np
import pymc3 as pm
from .funcs import calc_bayes_factor


# def plot(data, prior, posterior, sort_by=None, figsize=(10, 12), height_ratios=[3, 1]):
#     plt.rcParams.update({'font.size': 12})
#     fig = plt.figure(figsize=figsize)
#     gs = gridspec.GridSpec(2, 1, height_ratios=height_ratios)
#     ax1 = plt.subplot(gs[0])
#     ax2 = plt.subplot(gs[1])

#     ax1 = forest_plot(ax1, data, sort_by)
#     ax1.set(xlabel=None)
#     ax2 = bayes_factor_plot(ax2, prior, posterior)
#     fig.tight_layout()
#     return [ax1, ax2]


def forest_plot(ax, data, sort_by=None, marker_size=8):
    '''
    Generate a classic meta analytic forest plot.
    Inputs
    - ax: an axis to plot on
    - trace: an MCMC trace from PyMC3
    - R: a vector of observed correlation coefficients
    - study_names: a list of strings which will appear on the plot
    '''

    if sort_by is not None:
        data = (data.sort_values(by=sort_by, ascending=True)
                    .reset_index(drop=True))

    # Extract information from dataframe
    effect_size_obs = data['effect_size'].values
    study_names = data['study'].values
    n_studies = data.shape[0]-1
    effect_size_est_mean = data['effect_size_est_mean'].values
    effect_size_est_HDI_lower = data['effect_size_est_HDI_lower'].values
    effect_size_est_HDI_upper = data['effect_size_est_HDI_upper'].values

    # Plotting begins ----------------------------------------------------
    # y positions including population inference
    y_tick_pos = np.arange(0, n_studies+1)

    # plot observed correlation coefficients as reported in studies
    ax.plot(effect_size_obs, y_tick_pos, 'x', color='k',
            markersize=marker_size, label='reported')

    # plot inferred distributions of correlation coefficients for each study
    ax.errorbar(effect_size_est_mean, y_tick_pos,
                xerr=[effect_size_est_HDI_upper - effect_size_est_mean,
                      effect_size_est_mean - effect_size_est_HDI_lower],
                fmt='o', markersize=marker_size, color='k',
                label='inferred')

    # vertical line for inferred true population level correlation coefficient
    est_population_mean = effect_size_est_mean[-1]
    ax.axvline(x=est_population_mean, color='k', ls='--')

    # vertical line for effect line
    ax.axvline(x=0, color='k')

    # miscellaneous formatting
    ax.legend()
    # ax.legend(loc='center', bbox_to_anchor=(0.5, 1.1))
    y_clearance = 0.5
    ax.set(title='Forest Plot',
           ylim=[0-y_clearance, (n_studies)+y_clearance],
           yticks=y_tick_pos,
           yticklabels=study_names)
    ax.invert_yaxis()
    return ax


def bayes_factor_plot(ax, prior, posterior, bins=41, style='density'):

    prior_samples = prior['effect_size_population']

    posterior_samples = posterior['effect_size_population']
    posterior_mean = np.mean(posterior_samples)

    if style is 'density':
        xi = np.linspace(np.max([np.min(prior_samples), -3]), 
                         np.min([np.max(prior_samples), 3]), 1000)

        # posterior density
        kde = KDEUnivariate(prior_samples)
        kde.fit()
        prior_kde = kde.evaluate(xi)
        ax.plot(xi, prior_kde, '--', label='prior', c='k', lw=3)

        # prior density
        kde = KDEUnivariate(posterior_samples)
        kde.fit()
        posterior_kde = kde.evaluate(xi)
        ax.plot(xi, posterior_kde, '-', label='posterior', c='k', lw=3)

        # # prior density is always 0.5, so we'll just plot a line to
        # # avoid edge effects of kde
        # ax.plot([-1, 1], [0.5, 0.5], '--', label='prior', c='k', lw=3)

    elif style is 'histogram':

        ax.hist(posterior_samples, label='posterior', density=True, alpha=0.5,
                color='blue', edgecolor='black', range=(-1, 1), bins=bins)

        ax.hist(prior_samples, label='prior', density=True, alpha=0.5,
                color='red', edgecolor='black', range=(-1, 1), bins=bins)

    # no effect line
    ax.axvline(x=0, color='k')

    # line for inferred truth
    ax.axvline(x=posterior_mean, color='k', ls='--')

    # Bayes Factor
    BF_prior_post = calc_bayes_factor(prior_samples, posterior_samples)

    # Make bayes factor string
    BF10 = BF_prior_post
    BF01 = 1/BF_prior_post
    bf_string = (strength_of_effect(BF01) + ' evidence for ' + direction_of_effect(BF01) 
                 + '$BF_{10}=$' + f'{BF10:.2f}, '
                 + '$BF_{01}=$' + f'{BF01:.2f}.')

    # make posterior interval string
    effect_size_hdi = pm.stats.hpd(posterior_samples)
    interval_str = f'mean R = {posterior_mean:.2f}, 95% HDI [{effect_size_hdi[0]:.2f}, {effect_size_hdi[1]:.2f}]'
    
    # set title
    title_str = bf_string + '\n' + interval_str
    ax.set_title(title_str)

    # miscellaneous formatting
    ax.legend()
    ax.set(ylabel='density')
    ax.set_ylim(bottom=0)
    return ax

def direction_of_effect(BF01):
    if BF01 > 1:
        return 'no effect: '
    else:
        return 'an effect: '
    
def strength_of_effect(BF01):
    # strength of effect
    if BF01 < 1:
        bf = 1/BF01
    else:
        bf = BF01

    if bf > 100:
        strength = 'Extreme'
    elif bf > 30:
        strength = 'Very strong'
    elif bf > 10:
        strength = 'Strong'
    elif bf > 3:
        strength = 'Moderate'
    else:
        strength = 'Anecdotal'

    return strength
