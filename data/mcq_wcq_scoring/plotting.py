import numpy as np
import matplotlib.pyplot as plt
from inference import discount_function


def participant_plot(id, trace, data, n_samples_to_plot=100, legend=True):
    '''Plot information about a participant. 
    Posterior inferences in parameter space.
    Data and posterior predictive checking in data space.'''
    
    logk = trace['logk'][:,id]

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    # DATA SPACE =======================================
    plot_data_space(id, ax, data, logk, n_samples_to_plot)
    

def plot_data_space(id, ax, data, logk, n_samples_to_plot=50):
    
    # plot the data
    pdata = data.loc[data['id'] == id]
    plot_data(pdata, ax, legend=False)
    
    # plot discount functions
    max_delay = np.max(pdata['DB'].values) * 1.1
    delays = np.linspace(0, max_delay, 1000)
    
    if n_samples_to_plot > 0:
        '''
        TODO: plotting samples from posterior is a bit iffy. Currently
        we just take the first set of samples, but ideally we want to take
        a random set of samples.
        '''
        # plot discount functions, sampled from the posterior
        for n in range(n_samples_to_plot):
    #         RB = pdata['RB'].values[0]
            ax.plot(delays, 
                    discount_function(delays, np.exp(logk[n])),
                    c='k', alpha=0.1)
    else:
        # plot 95% CI region
        q = np.quantile(logk, [0.025, 0.975])
        ax.fill_between(delays, 
                        discount_function(delays, np.exp(q[0])), 
                        discount_function(delays, np.exp(q[1])), 
                        alpha=0.25, color='b', linewidth=0.0)
        
        # plot 95% CI region
        q = np.quantile(logk, [0.25, 0.75])
        ax.fill_between(delays, 
                        discount_function(delays, np.exp(q[0])), 
                        discount_function(delays, np.exp(q[1])), 
                        alpha=0.25, color='b', linewidth=0.0)
        
    # plot median discount rate
    ax.plot(delays, 
                discount_function(delays, np.exp(np.median(logk))),
                c='k', linewidth=3)
    
    # plot participant id info text
    if pdata['RB'].values[0] > 0:
        text_y = 1.
    elif pdata['RB'].values[0] < 0:
        text_y = -1.
        
#     ax.text(2, text_y, f'participant id: {id}',
#          horizontalalignment='left',
#          verticalalignment='center', #transform = ax.transAxes,
#          fontsize=10)
    
    ax.set_title(f'participant id: {id}')
        
    ax.set(xlabel='DB [days]', 
           ylabel='RA/RB')
    ax.set_xlim(left=0)
    
    
def plot_data(data, ax, legend=True):
    D = data['R'] == 1
    I = data['R'] == 0
    
    if np.sum(D) > 0:
        ax.scatter(x=data['DB'][D],
                   y=(data['RA']/data['RB'])[D],
                   c='k',
                   edgecolors='k',
                   label='chose delayed prospect')
    if np.sum(I) > 0:    
        ax.scatter(x=data['DB'][I],
                   y=(data['RA']/data['RB'])[I],
                   c='w',
                   edgecolors='k',
                   label='chose immediate prospect')
        
    # deal with y axis limit
    ax.set_ylim(bottom=0)