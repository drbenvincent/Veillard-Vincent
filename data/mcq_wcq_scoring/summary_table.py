import numpy as np
import pandas as pd
import numpy.matlib
from sklearn.metrics import log_loss, roc_auc_score
from utils import get_n_participants
from inference import discount_function


def build_final_table(trace, data):
    '''Takes in the trace (MCMC chains) and returns a dataframe of results.
    Each row is a participant.
    Columns are point esimtates for parameters (log(k)) and derived goodness of fit and AUC measures
    
    data is a dataframe of raw trial level data for all participants.
    Each '''
    rows = []
    print('Calculating derived measures and building summary table')
    for id in range(get_n_participants(data)):
        logk = trace['logk'][:,id]
        P_chooseB = trace['P'][:,id]

        pdata = data.loc[data['id'] == id]

        Ppredicted = trace.P[:, data['id'] == id]
        Ractual = pdata['R'].values
        
        participant_number = pdata['participant_number'].values[0]

        rowdata = make_rowdata(id, participant_number, logk, pdata, Ractual, Ppredicted)
        rows.append(rowdata)
        # print(f'{id+1} of {get_n_participants(data)}')

    parameter_estimates = pd.concat(rows, ignore_index=True)
    return parameter_estimates


def make_rowdata(id, participant_number, logk, pdata, Ractual, Ppredicted):
    '''Make a row of data (for a participant) that will be in the final dataframe'''
    logk_point_estimate = np.mean(logk)
    rowdata = {'id': [id],
               'participant_number': participant_number,
               'logk': [logk_point_estimate], 
               # 'AUC': calc_AUC(logk_point_estimate), 
               'percent_predicted': calc_percent_predicted(np.median(Ppredicted, axis=0), Ractual),
               # 'log_loss': calc_log_loss(np.median(Ppredicted, axis=0), Ractual)
               }
    return pd.DataFrame.from_dict(rowdata)


# def calc_AUC(logk, max_delay=101):
#     '''Calculate Area Under Curve measure'''
#     delays = np.linspace(0, max_delay, 500)
#     df = discount_function(delays, np.exp(logk))
#     normalised_delays = delays / np.max(delays)
#     AUC = np.trapz(df, x=normalised_delays)
#     return AUC


def calc_percent_predicted(R_predicted_prob, R_actual):
    nresponses = R_actual.shape[0]
    predicted_responses = np.where(R_predicted_prob>0.5, 1, 0)
    n_correct = sum(np.equal(predicted_responses, R_actual))
    return  n_correct / nresponses


# def calc_log_loss(R_predicted_prob, R_actual):
#     try: 
#         ll = log_loss(R_actual, R_predicted_prob)
#     except:
#         ll = None
#     return ll