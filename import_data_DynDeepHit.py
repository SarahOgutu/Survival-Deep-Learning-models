import pandas as pd
import numpy as np



##### USER-DEFINED FUNCTIONS
def f_get_Normalization(X, norm_mode):    
    num_Patient, num_Feature = np.shape(X)
    
    if norm_mode == 'standard': #zero mean unit variance
        for j in range(num_Feature):
            if np.nanstd(X[:,j]) != 0:
                X[:,j] = (X[:,j] - np.nanmean(X[:, j]))/np.nanstd(X[:,j])
            else:
                X[:,j] = (X[:,j] - np.nanmean(X[:, j]))
    elif norm_mode == 'normal': #min-max normalization
        for j in range(num_Feature):
            X[:,j] = (X[:,j] - np.nanmin(X[:,j]))/(np.nanmax(X[:,j]) - np.nanmin(X[:,j]))
    else:
        print("INPUT MODE ERROR!")
    
    return X


def f_get_fc_mask1(meas_time, num_Event, num_Category):
    '''
        mask3 is required to get the contional probability (to calculate the denominator part)
        mask3 size is [N, num_Event, num_Category]. 1's until the last measurement time
    '''
    mask = np.zeros([np.shape(meas_time)[0], num_Event, num_Category]) # for denominator
    for i in range(np.shape(meas_time)[0]):
        mask[i, :, :int(meas_time[i, 0]+1)] = 1 # last measurement time

    return mask


def f_get_fc_mask2(time, label, num_Event, num_Category):
    '''
        mask4 is required to get the log-likelihood loss 
        mask4 size is [N, num_Event, num_Category]
            if not censored : one element = 1 (0 elsewhere)
            if censored     : fill elements with 1 after the censoring time (for all events)
    '''
    mask = np.zeros([np.shape(time)[0], num_Event, num_Category]) # for the first loss function
    for i in range(np.shape(time)[0]):
        if label[i,0] != 0:  #not censored
            mask[i,int(label[i,0]-1),int(time[i,0])] = 1
        else: #label[i,2]==0: censored
            mask[i,:,int(time[i,0]+1):] =  1 #fill 1 until from the censoring time (to get 1 - \sum F)
    return mask


def f_get_fc_mask3(time, meas_time, num_Category):
    '''
        mask5 is required calculate the ranking loss (for pair-wise comparision)
        mask5 size is [N, num_Category]. 
        - For longitudinal measurements:
             1's from the last measurement to the event time (exclusive and inclusive, respectively)
             denom is not needed since comparing is done over the same denom
        - For single measurement:
             1's from start to the event time(inclusive)
    '''
    mask = np.zeros([np.shape(time)[0], num_Category]) # for the first loss function
    if np.shape(meas_time):  #lonogitudinal measurements 
        for i in range(np.shape(time)[0]):
            t1 = int(meas_time[i, 0]) # last measurement time
            t2 = int(time[i, 0]) # censoring/event time
            mask[i,(t1+1):(t2+1)] = 1  #this excludes the last measurement time and includes the event time
    else:                    #single measurement
        for i in range(np.shape(time)[0]):
            t = int(time[i, 0]) # censoring/event time
            mask[i,:(t+1)] = 1  #this excludes the last measurement time and includes the event time
    return mask

import pdb

##### TRANSFORMING DATA
def f_construct_dataset(df, feat_list):
    grouped  = df.groupby(['PID'])
    id_list  = pd.unique(df['PID'])
    max_meas = np.max(grouped.count())[0]

    # Make sure len(feat_list) matches the expected number of features
    data     = np.zeros([len(id_list), max_meas, len(feat_list)+1])  # Adjusted initialization
    pat_info = np.zeros([len(id_list), 5])

    for i, tmp_id in enumerate(id_list):
        tmp = grouped.get_group(tmp_id).reset_index(drop=True)

        pat_info[i,4] = tmp.shape[0]                                   # Number of measurements
        pat_info[i,3] = np.max(tmp['Follow_up_Month'])                 # Last measurement time
        pat_info[i,2] = tmp['HIV'][0]                                  # Cause
        pat_info[i,1] = tmp['months'][0]                               # Time to event
        pat_info[i,0] = tmp['PID'][0]                                  # Patient ID      
        
        data[i, :int(pat_info[i, 4]), 1:]  = tmp[feat_list]
        data[i, :int(pat_info[i, 4]-1), 0] = np.diff(tmp['Follow_up_Month'])
    
    return pat_info, data


def import_dataset(norm_mode = 'standard'):

    df_                = pd.read_csv('./DynDeepHit_data.csv')
    bin_list           = ['Treat', 'Site', 'p2v18_REG_PARTNER_LIVE_TOGETHER', 'p3v8_SELF_GEN_INCOME', 'p3v9_SALARY', 'p3v10_HUSBAND', 'p3v11_SOCIAL_GRANTS', 'p3v13_OTHER_INCOME_SOURCE', 'p3v16_AMOUNT_INCOME', 'p18v15_FREQ_CONDOM_USE', 'p19v14_ABNORMAL_DISCHARGE']
    cont_list          = ['p5v9_LENGTH_IN_DBNVUL', 'age1', 'agedebu', 'p17v10_PARTNERS', 'p17v11_YEAR_STABLE', 'p17v12_YEAR_CASUAL', 'p17v13_30DAYS_STABLE', 'p17v14_30DAYS_CASUAL', 'p17v15_SEX_30DAYS', 'p17v26_AGE_OLDEST_SEX_PART','BASIC_FGF', 'EOTAXIN', 'G_CSF', 'GM_CSF', 'IFN_G', 'IL10', 'IL_12P70', 'IL13', 'IL15', 'IL_17A', 'IL_1B', 'IL_1RA', 'IL2', 'IL4', 'IL5', 'IL6', 'IL7', 'IL8', 'IL9', 'MCP1', 'MIP_1A', 'MIP_1B', 'PDGF_BB', 'RANTES', 'TNF_A', 'VEGF', 'CTACK', 'GRO_A', 'HGF', 'IFN_A2', 'IL_12P40', 'IL16', 'IL18', 'IL_2RA', 'IL3', 'LIF', 'M_CSF', 'MCP3', 'MIF', 'MIG', 'SCF', 'SCGF_B', 'SDF_1A', 'TNF_B', 'TRAIL', 'B_NGF']
    feat_list          = cont_list + bin_list
    df_                = df_[['PID', 'months', 'Follow_up_Month', 'HIV']+feat_list]
    df_org_            = df_.copy(deep=True)

    df_[cont_list]     = f_get_Normalization(np.asarray(df_[cont_list]).astype(float), norm_mode)

    pat_info, data     = f_construct_dataset(df_, feat_list)
    _, data_org        = f_construct_dataset(df_org_, feat_list)

    data_mi                  = np.zeros(np.shape(data))
    data_mi[np.isnan(data)]  = 1
    data_org[np.isnan(data)] = 0
    data[np.isnan(data)]     = 0 

    x_dim           = np.shape(data)[2] # 1 + x_dim_cont + x_dim_bin (including delta)
    x_dim_cont      = len(cont_list)
    x_dim_bin       = len(bin_list) 

    last_meas       = pat_info[:,[3]]  #pat_info[:, 3] contains age at the last measurement
    label           = pat_info[:,[2]]  #two competing risks
    time            = pat_info[:,[1]]  #age when event occurred

    num_Category    = int(np.max(pat_info[:, 1]) * 1.2) #or specifically define larger than the max tte
    num_Event       = len(np.unique(label)) - 1

    if num_Event == 1:
        label[np.where(label!=0)] = 1 #make single risk

    mask1           = f_get_fc_mask1(last_meas, num_Event, num_Category)
    mask2           = f_get_fc_mask2(time, label, num_Event, num_Category)
    mask3           = f_get_fc_mask3(time, -1, num_Category)

    DIM             = (x_dim, x_dim_cont, x_dim_bin)
    DATA            = (data, time, label)
    # DATA            = (data, data_org, time, label)
    MASK            = (mask1, mask2, mask3)

    return DIM, DATA, MASK, data_mi
