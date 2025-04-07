# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 10:30:01 2025

@author: Sarah Ogutu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import train_test_split

import torch
import torchtuples as tt

from pycox.models import CoxPH
#from pycox.evaluation.concordance import concordance_td
#from pycox.evaluation import ipcw, admin


from Util import EvalSurv #updated EvalSurv for pd error
from Model import linear_model
from IPython import get_ipython

np.random.seed(2408)
_ = torch.manual_seed(2408)


# Load the datasets
df_diff1 = pd.read_excel('D:/Revised/dl_diff_cc.xlsx')# Complete Case
df_diff2 = pd.read_excel('D:/Revised/diff_data.xlsx')# Imputed data


# Set random seed
RANDOM_STATE = 2408

# Ensure both datasets have a common ID column for tracking individuals (assume 'PID' is the ID column)
id_col = 'PID'

# Split complete case dataset: 80% train, 20% test
df_cc_train1, df_cc_test1 = train_test_split(df_diff1, test_size=0.2, random_state=RANDOM_STATE, stratify=None)

# Get IDs from complete-case test set to avoid them in the imputed training set
cc_test_ids = df_cc_test1[id_col]

# Split imputed dataset:
# - Train: exclude any rows with IDs in cc_test_ids
# - Test: include all rows with IDs in cc_test_ids + remaining data to make up the test size

# Separate rows to exclude from train (those in cc_test_ids)
df_imputed_test_overlap = df_diff2[df_diff2[id_col].isin(cc_test_ids)]

# Remaining rows eligible for training
df_imputed_remaining = df_diff2[~df_diff2[id_col].isin(cc_test_ids)]

# Now split remaining into training data (keep size to match N=650)
df_imputed_train1, _ = train_test_split(
    df_imputed_remaining,
    train_size=650,
    random_state=RANDOM_STATE
)

# Imputed test set = complete-case test set + enough extra samples to make N=162
additional_test_size = 162 - len(df_imputed_test_overlap)

# Select additional samples not already in test
df_imputed_test_extra = df_imputed_remaining[
    ~df_imputed_remaining[id_col].isin(df_imputed_train1[id_col])
].sample(n=additional_test_size, random_state=RANDOM_STATE)

# Final imputed test set
df_imputed_test1 = pd.concat([df_imputed_test_overlap, df_imputed_test_extra])

# Check the shapes
print("Complete-case train:", df_cc_train1.shape)
print("Complete-case test:", df_cc_test1.shape)
print("Imputed train:", df_imputed_train1.shape)
print("Imputed test:", df_imputed_test1.shape)

# remove the PID column before training
df_cc_train1 = df_cc_train1.drop(df_cc_train1.columns[0], axis=1)
df_cc_test1 = df_cc_test1.drop(df_cc_test1.columns[0], axis=1)
df_imputed_train1 = df_imputed_train1.drop(df_imputed_train1.columns[0], axis=1)
df_imputed_test1 = df_imputed_test1.drop(df_imputed_test1.columns[0], axis=1)

#Feature transforms
#Feature transforms
cols_standardize1 = ['age1', 'agedebu', 'p17v26_AGE_OLDEST_SEX_PART', 'BASIC_FGF', 'EOTAXIN', 'G_CSF', 'GM_CSF', 'IFN_G', 'IL_10', 'IL_12P70', 'IL_13', 'IL_15', 'IL_17A', 'IL_1B', 'IL_1RA', 'IL_2', 'IL_4', 'IL_5', 'IL_6', 'IL_7', 'IL_8', 'IL_9', 'IP_10', 'MCP_1', 'MIP_1A', 'MIP_1B', 'PDGF_BB', 'RANTES', 'TNF_A', 'VEGF', 'CTACK', 'GRO_A', 'HGF', 'IFN_A2', 'IL_12P40', 'IL_16', 'IL_18', 'IL_1A', 'IL_2RA', 'IL_3', 'LIF', 'M_CSF', 'MCP_3', 'MIF', 'MIG', 'SCF', 'SCGF_B', 'SDF_1A', 'TNF_B', 'TRAIL', 'B_NGF']
cols_leave1 = ['p5v9_LENGTH_IN_DBNVUL', 'p17v10_PARTNERS', 'p17v11_YEAR_STABLE', 'p17v12_YEAR_CASUAL', 'p17v13_30DAYS_STABLE', 'p17v14_30DAYS_CASUAL', 'p17v15_SEX_30DAYS', 'Treat_Placebo', 'Treat_Tenofovir', 'Site_eThekwini', 'Site_Vulindlela', 'p2v18_REG_PARTNER_LIVE_TOGETHER_No', 'p2v18_REG_PARTNER_LIVE_TOGETHER_Yes', 'p2v22_HIGHEST_EDUCATION_HSchool', 'p2v22_HIGHEST_EDUCATION_Primary', 'p2v22_HIGHEST_EDUCATION_Tertiary', 'p3v8_SELF_GEN_INCOME_No', 'p3v8_SELF_GEN_INCOME_Yes', 'p3v9_SALARY_No', 'p3v9_SALARY_Yes', 'p3v10_HUSBAND_No', 'p3v10_HUSBAND_Yes', 'p3v11_SOCIAL_GRANTS_No', 'p3v11_SOCIAL_GRANTS_Yes', 'p3v13_OTHER_INCOME_SOURCE_No', 'p3v13_OTHER_INCOME_SOURCE_Yes', 'p3v16_AMOUNT_INCOME_G1Kless5K', 'p3v16_AMOUNT_INCOME_less1K', 'marital_Casual', 'marital_Married', 'marital_StabCas', 'marital_Stable', 'p17v28_SEX_PART_HAVE_OTHER_Dontknow', 'p17v28_SEX_PART_HAVE_OTHER_No', 'p17v28_SEX_PART_HAVE_OTHER_Yes', 'p18v15_FREQ_CONDOM_USE_Always', 'p18v15_FREQ_CONDOM_USE_Occasionally', 'p19v14_ABNORMAL_DISCHARGE_No', 'p19v14_ABNORMAL_DISCHARGE_Yes']
standardize1 = [([col], StandardScaler()) for col in cols_standardize1]
leave1 = [(col, None) for col in cols_leave1]
x_mapper1 = DataFrameMapper(standardize1 + leave1)

######################### Complete case #####################################
df_val2 = df_cc_train1.sample(frac=0.2)
x_train2 = x_mapper1.fit_transform(df_cc_train1).astype('float32')
x_val2 = x_mapper1.transform(df_val2).astype('float32')
x_test2 = x_mapper1.transform(df_cc_test1).astype('float32')


#Get Surv labels
get_target2 = lambda df: (df['months'].values, df['HIV'].values)
y_train2 = get_target2(df_cc_test1)
y_val2 = get_target2(df_val2)
y_test2 = get_target2(df_cc_test1)


#############################################################################
# Pre-process dataset
#############################################################################
in_features = x_train2.shape[1]
out_features  = 1
node_dims = [in_features,32,32,out_features]
layer_norm = True
batch_norm = True
dropout = 0.1
output_bias = False
activation_type = 'relu'
num_nodes = [32, 32]
batch_size = 128
epochs = 512

#Construct model
net2 = linear_model(node_dims, dropout, activation_type, layer_norm, batch_norm)
#Define optimizer
optimizer = tt.optim.Adam

#Define CoxPH model
model2 = CoxPH(net2, optimizer)

#Find best lr
lrfinder2 = model2.lr_finder(x_train2, y_train2, batch_size, tolerance=10)
#_ = lrfinder2.plot()
best_lr2 = lrfinder2.get_best_lr()

#Set best lr
model2.optimizer.set_lr(0.01)

#Set early stopping
callbacks = [tt.callbacks.EarlyStopping()]


##############################################################################
# Train Model
##############################################################################
get_ipython().run_line_magic('time', '')
log2 = model2.fit(x_train2, y_train2, batch_size, epochs, callbacks, 
                  verbose = True,
                val_data = (x_val2, y_val2), val_batch_size=batch_size)

#Plot loss
_ = log2.plot()



##############################################################################
# Test (Prediction)
##############################################################################
#compute baseline hazards at each time point
baseline_hazard2 = model2.compute_baseline_hazards()

#Predict
surv_pred2 = model2.predict_surv_df(x_test2) #rows-timepoints,cols-samples

#Plot survival for patients
num_pts = 10
surv_pred2.iloc[:, :num_pts].plot()
plt.ylabel('S(t | x)')
plt.title("Predicted Survival: Diff Model")
_ = plt.xlabel('Time')
plt.figure()
plt.close()


##############################################################################
#Evaluation
##############################################################################
ev2 = EvalSurv(surv_pred2, y_test2[0], y_test2[1], censor_surv='km')

#The time-dependent concordance index evaluated at the event times
cindex_td2 = ev2.concordance_td()
print("\nThe time-dep c-index is: ",round(cindex_td2,4))


#Get Time points in test sets
time_grid2 = np.linspace(y_test2[0].min(), y_test2[0].max(), 100)


#The integrated IPCW Brier score. 
inte_ipcw_bs2 = ev2.integrated_brier_score(time_grid2)
print("The integrated IPCW Brier Score is:", round(inte_ipcw_bs2,4))

#The integrated IPCW (negative) binomial log-likelihood.
inte_ipcw_nbl2 = ev2.integrated_nbll(time_grid2)
print("The integrated IPCW (negative) binomial log-likelihood is: ", 
      round(inte_ipcw_nbl2,4))

#The IPCW Brier score (inverse probability of censoring weighted Brier score)
ipcw_bs2 = ev2.brier_score(time_grid2)
print("The  IPCW Brier score is: \n",ipcw_bs2)
plt.figure()
plt.title("IPCW Brier score")
_ = ev2.brier_score(time_grid2).plot()

##################################### Imputed data #####################################
df_val2 = df_imputed_train1.sample(frac=0.2)
x_train2 = x_mapper1.fit_transform(df_imputed_train1).astype('float32')
x_val2 = x_mapper1.transform(df_val2).astype('float32')
x_test2 = x_mapper1.transform(df_imputed_test1).astype('float32')


#Get Surv labels
get_target2 = lambda df: (df['months'].values, df['HIV'].values)
y_train2 = get_target2(df_imputed_test1)
y_val2 = get_target2(df_val2)
y_test2 = get_target2(df_imputed_test1)


#############################################################################
# Pre-process dataset
#############################################################################
in_features = x_train2.shape[1]
out_features  = 1
node_dims = [in_features,32,32,out_features]
layer_norm = True
batch_norm = True
dropout = 0.1
output_bias = False
activation_type = 'relu'
num_nodes = [32, 32]
batch_size = 128
epochs = 512

#Construct model
net2 = linear_model(node_dims, dropout, activation_type, layer_norm, batch_norm)
#Define optimizer
optimizer = tt.optim.Adam

#Define CoxPH model
model2 = CoxPH(net2, optimizer)

#Find best lr
lrfinder2 = model2.lr_finder(x_train2, y_train2, batch_size, tolerance=10)
#_ = lrfinder2.plot()
best_lr2 = lrfinder2.get_best_lr()

#Set best lr
model2.optimizer.set_lr(0.01)

#Set early stopping
callbacks = [tt.callbacks.EarlyStopping()]


##############################################################################
# Train Model
##############################################################################
get_ipython().run_line_magic('time', '')
log2 = model2.fit(x_train2, y_train2, batch_size, epochs, callbacks, 
                  verbose = True,
                val_data = (x_val2, y_val2), val_batch_size=batch_size)

#Plot loss
_ = log2.plot()



##############################################################################
# Test (Prediction)
##############################################################################
#compute baseline hazards at each time point
baseline_hazard2 = model2.compute_baseline_hazards()

#Predict
surv_pred2 = model2.predict_surv_df(x_test2) #rows-timepoints,cols-samples

#Plot survival for patients
num_pts = 10
surv_pred2.iloc[:, :num_pts].plot()
plt.ylabel('S(t | x)')
plt.title("Predicted Survival: Diff Model")
_ = plt.xlabel('Time')
plt.figure()
plt.close()


##############################################################################
#Evaluation
##############################################################################
ev2 = EvalSurv(surv_pred2, y_test2[0], y_test2[1], censor_surv='km')

#The time-dependent concordance index evaluated at the event times
cindex_td2 = ev2.concordance_td()
print("\nThe time-dep c-index is: ",round(cindex_td2,4))


#Get Time points in test sets
time_grid2 = np.linspace(y_test2[0].min(), y_test2[0].max(), 100)


#The integrated IPCW Brier score. 
inte_ipcw_bs2 = ev2.integrated_brier_score(time_grid2)
print("The integrated IPCW Brier Score is:", round(inte_ipcw_bs2,4))

#The integrated IPCW (negative) binomial log-likelihood.
inte_ipcw_nbl2 = ev2.integrated_nbll(time_grid2)
print("The integrated IPCW (negative) binomial log-likelihood is: ", 
      round(inte_ipcw_nbl2,4))

#The IPCW Brier score (inverse probability of censoring weighted Brier score)
ipcw_bs2 = ev2.brier_score(time_grid2)
print("The  IPCW Brier score is: \n",ipcw_bs2)
plt.figure()
plt.title("IPCW Brier score")
_ = ev2.brier_score(time_grid2).plot()

