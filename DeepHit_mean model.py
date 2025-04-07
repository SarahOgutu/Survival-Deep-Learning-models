# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 14:32:01 2025

@author: Sarah Ogutu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 

import torch # For building the networks 
import torchtuples as tt # Some useful functions

from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv

####### Data Set ######
####### Complete Case
# Use the already split data from DeepSurv_mean model script

df_train3 = df_cc_train
df_test3 = df_cc_test
df_val3 = df_train3.sample(frac=0.2)


######## Fearture transformation #############
cols_standardize1 = ['age1', 'agedebu', 'p17v26_AGE_OLDEST_SEX_PART',  'BASIC_FGF', 'EOTAXIN', 'G_CSF', 'GM_CSF', 'IFN_G', 'IL_10', 'IL_12P70', 'IL_13', 'IL_15', 'IL_17A', 'IL_1B', 'IL_1RA', 'IL_2', 'IL_4', 'IL_5', 'IL_6', 'IL_7', 'IL_8', 'IL_9', 'IP_10', 'MCP_1', 'MIP_1A', 'MIP_1B', 'PDGF_BB', 'RANTES', 'TNF_A', 'VEGF', 'CTACK', 'GRO_A', 'HGF', 'IFN_A2', 'IL_12P40', 'IL_16', 'IL_18', 'IL_1A', 'IL_2RA', 'IL_3', 'LIF', 'M_CSF', 'MCP_3', 'MIF', 'MIG', 'SCF', 'SCGF_B', 'SDF_1A', 'TNF_B', 'TRAIL', 'B_NGF']
cols_leave1 = ['p5v9_LENGTH_IN_DBNVUL', 'p17v10_PARTNERS', 'p17v11_YEAR_STABLE', 'p17v12_YEAR_CASUAL', 'p17v13_30DAYS_STABLE', 'p17v14_30DAYS_CASUAL', 'p17v15_SEX_30DAYS', 'Treat_Placebo', 'Treat_Tenofovir', 'Site_eThekwini', 'Site_Vulindlela', 'p2v18_REG_PARTNER_LIVE_TOGETHER_No', 'p2v18_REG_PARTNER_LIVE_TOGETHER_Yes', 'p2v22_HIGHEST_EDUCATION_HSchool', 'p2v22_HIGHEST_EDUCATION_Primary', 'p2v22_HIGHEST_EDUCATION_Tertiary', 'p3v8_SELF_GEN_INCOME_No', 'p3v8_SELF_GEN_INCOME_Yes', 'p3v9_SALARY_No', 'p3v9_SALARY_Yes', 'p3v10_HUSBAND_No', 'p3v10_HUSBAND_Yes', 'p3v11_SOCIAL_GRANTS_No', 'p3v11_SOCIAL_GRANTS_Yes', 'p3v13_OTHER_INCOME_SOURCE_No', 'p3v13_OTHER_INCOME_SOURCE_Yes', 'p3v16_AMOUNT_INCOME_G1Kless5K', 'p3v16_AMOUNT_INCOME_less1K', 'marital_Casual', 'marital_Married', 'marital_StabCas', 'marital_Stable', 'p17v28_SEX_PART_HAVE_OTHER_Dontknow', 'p17v28_SEX_PART_HAVE_OTHER_No', 'p17v28_SEX_PART_HAVE_OTHER_Yes', 'p18v15_FREQ_CONDOM_USE_Always', 'p18v15_FREQ_CONDOM_USE_Occasionally', 'p19v14_ABNORMAL_DISCHARGE_No', 'p19v14_ABNORMAL_DISCHARGE_Yes']
standardize1 = [([col], StandardScaler()) for col in cols_standardize1]
leave1 = [(col, None) for col in cols_leave1]
x_mapper = DataFrameMapper(standardize1 + leave1)

x_train3 = x_mapper.fit_transform(df_train3).astype('float32')
x_val3 = x_mapper.transform(df_val3).astype('float32')
x_test3 = x_mapper.transform(df_test3).astype('float32')

######## Label Transform ##################
num_durations = 10
labtrans = DeepHitSingle.label_transform(num_durations)
get_target3 = lambda df: (df['months'].values, df['HIV'].values)
y_train3 = labtrans.fit_transform(*get_target3(df_train3))
y_val3 = labtrans.transform(*get_target3(df_val3))

train = (x_train3, y_train3)
val = (x_val3, y_val3)

# We don't need to transform the test labels
durations_test, events_test = get_target3(df_test3)
y_test3 = get_target3(df_test3)

type(labtrans)

###### Neural Net #######################
in_features = x_train3.shape[1]
num_nodes = [32, 32]
out_features = labtrans.out_features
batch_norm = True
dropout = 0.1

net3 = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                              dropout)
############ Training the model #############
model3 = DeepHitSingle(net3, tt.optim.Adam, alpha=0.2, sigma=0.1,
                       duration_index=labtrans.cuts)
batch_size = 256
lr_finder3 = model3.lr_finder(x_train3, y_train3, batch_size, tolerance=3)
_ = lr_finder3.plot()

lr_finder3.get_best_lr()
model3.optimizer.set_lr(0.01)

epochs = 512
callbacks = [tt.callbacks.EarlyStopping()]
log3 = model3.fit(x_train3, y_train3, batch_size, epochs, callbacks, 
                  val_data=val)
_ = log3.plot()

############# Prediction #############
surv3 = model3.predict_surv_df(x_test3)


surv3.iloc[:, :10].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')

surv33 = model3.interpolate(10).predict_surv_df(x_test3)

surv33.iloc[:, :10].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')

########### Evaluation ##########
# C-index
ev3 = EvalSurv(surv3, durations_test, events_test, censor_surv='km')
ev3.concordance_td()

# IPCW Brier score
time_grid3 = np.linspace(y_test3[0].min(), y_test3[0].max(), 100)
ev3.brier_score(time_grid3).plot()
plt.ylabel('Brier score')
_ = plt.xlabel('Time')

# Negative binomial likelihood
ev3.nbll(time_grid3).plot()
plt.ylabel('NBLL')
_ = plt.xlabel('Time')

# itegrated Brier Score
ev3.integrated_brier_score(time_grid3) 
ev3.integrated_nbll(time_grid3) 

####### Imputed Data
# Use the already split data from DeepSurv_mean model script

df_train3 = df_imputed_train
df_test3 = df_imputed_test
df_val3 = df_train3.sample(frac=0.2)


######## Fearture transformation #############
cols_standardize1 = ['age1', 'agedebu', 'p17v26_AGE_OLDEST_SEX_PART',  'BASIC_FGF', 'EOTAXIN', 'G_CSF', 'GM_CSF', 'IFN_G', 'IL_10', 'IL_12P70', 'IL_13', 'IL_15', 'IL_17A', 'IL_1B', 'IL_1RA', 'IL_2', 'IL_4', 'IL_5', 'IL_6', 'IL_7', 'IL_8', 'IL_9', 'IP_10', 'MCP_1', 'MIP_1A', 'MIP_1B', 'PDGF_BB', 'RANTES', 'TNF_A', 'VEGF', 'CTACK', 'GRO_A', 'HGF', 'IFN_A2', 'IL_12P40', 'IL_16', 'IL_18', 'IL_1A', 'IL_2RA', 'IL_3', 'LIF', 'M_CSF', 'MCP_3', 'MIF', 'MIG', 'SCF', 'SCGF_B', 'SDF_1A', 'TNF_B', 'TRAIL', 'B_NGF']
cols_leave1 = ['p5v9_LENGTH_IN_DBNVUL', 'p17v10_PARTNERS', 'p17v11_YEAR_STABLE', 'p17v12_YEAR_CASUAL', 'p17v13_30DAYS_STABLE', 'p17v14_30DAYS_CASUAL', 'p17v15_SEX_30DAYS', 'Treat_Placebo', 'Treat_Tenofovir', 'Site_eThekwini', 'Site_Vulindlela', 'p2v18_REG_PARTNER_LIVE_TOGETHER_No', 'p2v18_REG_PARTNER_LIVE_TOGETHER_Yes', 'p2v22_HIGHEST_EDUCATION_HSchool', 'p2v22_HIGHEST_EDUCATION_Primary', 'p2v22_HIGHEST_EDUCATION_Tertiary', 'p3v8_SELF_GEN_INCOME_No', 'p3v8_SELF_GEN_INCOME_Yes', 'p3v9_SALARY_No', 'p3v9_SALARY_Yes', 'p3v10_HUSBAND_No', 'p3v10_HUSBAND_Yes', 'p3v11_SOCIAL_GRANTS_No', 'p3v11_SOCIAL_GRANTS_Yes', 'p3v13_OTHER_INCOME_SOURCE_No', 'p3v13_OTHER_INCOME_SOURCE_Yes', 'p3v16_AMOUNT_INCOME_G1Kless5K', 'p3v16_AMOUNT_INCOME_less1K', 'marital_Casual', 'marital_Married', 'marital_StabCas', 'marital_Stable', 'p17v28_SEX_PART_HAVE_OTHER_Dontknow', 'p17v28_SEX_PART_HAVE_OTHER_No', 'p17v28_SEX_PART_HAVE_OTHER_Yes', 'p18v15_FREQ_CONDOM_USE_Always', 'p18v15_FREQ_CONDOM_USE_Occasionally', 'p19v14_ABNORMAL_DISCHARGE_No', 'p19v14_ABNORMAL_DISCHARGE_Yes']
standardize1 = [([col], StandardScaler()) for col in cols_standardize1]
leave1 = [(col, None) for col in cols_leave1]
x_mapper = DataFrameMapper(standardize1 + leave1)

x_train3 = x_mapper.fit_transform(df_train3).astype('float32')
x_val3 = x_mapper.transform(df_val3).astype('float32')
x_test3 = x_mapper.transform(df_test3).astype('float32')

######## Label Transform ##################
num_durations = 10
labtrans = DeepHitSingle.label_transform(num_durations)
get_target3 = lambda df: (df['months'].values, df['HIV'].values)
y_train3 = labtrans.fit_transform(*get_target3(df_train3))
y_val3 = labtrans.transform(*get_target3(df_val3))

train = (x_train3, y_train3)
val = (x_val3, y_val3)

# We don't need to transform the test labels
durations_test, events_test = get_target3(df_test3)
y_test3 = get_target3(df_test3)

type(labtrans)

###### Neural Net #######################
in_features = x_train3.shape[1]
num_nodes = [32, 32]
out_features = labtrans.out_features
batch_norm = True
dropout = 0.1

net3 = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                              dropout)
############ Training the model #############
model3 = DeepHitSingle(net3, tt.optim.Adam, alpha=0.2, sigma=0.1,
                       duration_index=labtrans.cuts)
batch_size = 256
lr_finder3 = model3.lr_finder(x_train3, y_train3, batch_size, tolerance=3)
_ = lr_finder3.plot()

lr_finder3.get_best_lr()
model3.optimizer.set_lr(0.01)

epochs = 512
callbacks = [tt.callbacks.EarlyStopping()]
log3 = model3.fit(x_train3, y_train3, batch_size, epochs, callbacks, 
                  val_data=val)
_ = log3.plot()

############# Prediction #############
surv3 = model3.predict_surv_df(x_test3)


surv3.iloc[:, :10].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')

surv33 = model3.interpolate(10).predict_surv_df(x_test3)

surv33.iloc[:, :10].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')

########### Evaluation ##########
# C-index
ev3 = EvalSurv(surv3, durations_test, events_test, censor_surv='km')
ev3.concordance_td()

# IPCW Brier score
time_grid3 = np.linspace(y_test3[0].min(), y_test3[0].max(), 100)
ev3.brier_score(time_grid3).plot()
plt.ylabel('Brier score')
_ = plt.xlabel('Time')

# Negative binomial likelihood
ev3.nbll(time_grid3).plot()
plt.ylabel('NBLL')
_ = plt.xlabel('Time')

# itegrated Brier Score
ev3.integrated_brier_score(time_grid3) 
ev3.integrated_nbll(time_grid3

################################# Hazard Plot
############################################################################### 
hazard3 = -np.diff(surv3.values, axis=0)  # Calculate the hazard function

# Convert the hazard function to a DataFrame and use the time index
hazard_df = pd.DataFrame(hazard3, index=surv3.index[1:], columns=surv3.columns)

# Plotting the hazard graph
plot1 = plt.figure(figsize=(8, 6))
for i in range(hazard_df.shape[1]):
    plt.plot(hazard_df.index, hazard_df.iloc[:, i], alpha=0.6)
    
plt.title("Hazard Graph Single Risk: mean model")
plt.xlabel("Evaluation time (months)")
plt.ylabel("Probability of risk")
plt.show()
#plot1.savefig('D:/Revised/DHmean.pdf', format='pdf', dpi=600)