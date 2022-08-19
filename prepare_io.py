import csv
import numpy as np
import pandas as pd
from config import *
from sklearn.preprocessing import MinMaxScaler

B_CRE_MAX, B_CRE_MIN = 0, 0

# read data
data = pd.read_csv('../data/preprocessed_with_duration.csv', encoding="cp1252", usecols = ['PERSONID2', 'DURATION', 'LABDATE', 'B_CRE', 'B_UN', 'DURATION']) 
print(data.shape) # should be 628610 * 6
patient_measures = pd.read_csv('../data/number_of_measures_per_patient.csv', encoding="cp1252") 
patient_id = patient_measures['PERSONID2']

# ignore patients with fewer than 4 measurements
valid_patients = patient_measures.loc[patient_measures['Number of Measures']>=4]
valid_patients_set = list(valid_patients['PERSONID2'])
valid_patients_set_df = pd.DataFrame(valid_patients_set)
valid_patients_set_df.to_csv('../data/valid_patients_set.csv')
valid_patients_measures = pd.DataFrame(valid_patients) # create dataframe to store valid patients
valid_patients_measures.to_csv('../data/valid_patients_measures.csv')
data_preprocessed = data.loc[data['PERSONID2'].isin(valid_patients_set)]
data_preprocessed.to_csv('../data/data_preprocessed.csv') # data for valid patients only

# train test split
# for each patient, first 80%->train, last 20%->test
train_data, test_data  = pd.DataFrame(), pd.DataFrame()
train_list, test_list = [], []
for i, id in enumerate(valid_patients_set): 
  measures = patient_measures.loc[patient_measures['PERSONID2']==id]['Number of Measures'].values[0]
  patient_data = (data_preprocessed.loc[data_preprocessed['PERSONID2']==id]).reset_index(drop = True)
  train_list.append(patient_data.loc[:int(0.8*measures), :])
  test_list.append(patient_data.loc[int(0.8*measures): measures, :])
train_data = train_data.append(train_list, ignore_index=True)
test_data = test_data.append(test_list, ignore_index=True)
train_data.to_csv('../data/train_data.csv')
test_data.to_csv('../data/test_data.csv')
print(train_data) # 499569 * 4
print(test_data) # 128694 * 4

# normalize train and test data
scaler = MinMaxScaler(feature_range=(0,1))
print('train data bcre max and min before scaling: ', train_data['B_CRE'].max(), train_data['B_CRE'].min())
train_data[['B_CRE', 'B_UN', 'DURATION']] = scaler.fit_transform(train_data[['B_CRE', 'B_UN', 'DURATION']]) # fit_transform on training data
test_data[['B_CRE', 'B_UN', 'DURATION']] = scaler.transform(test_data[['B_CRE', 'B_UN', 'DURATION']]) # transform on testing data
B_CRE_MAX = scaler.data_max_[0]
B_CRE_MIN = scaler.data_min_[0]
print('scaler max and min: ', B_CRE_MAX, B_CRE_MIN)

#region
# if age and sex are features
# background = pd.read_csv('../data/backgrounds.csv', usecols=['PERSONID2', 'SEX', 'AGE'], encoding="cp1252") # get sex and age
# background = background.drop_duplicates(subset='PERSONID2', keep='first')
# sex_dict = pd.Series(background.SEX.values,index=background.PERSONID2).to_dict() # key: ID, value: sex
# age_dict = pd.Series(background.AGE.values,index=background.PERSONID2).to_dict() # key: ID, value: age
 
# for i in range(len(train_data)):
#   id = train_data.loc[i, 'PERSONID2']
#   train_data.at[i, 'SEX'] = sex_dict.get(id) # sex
#   train_data.at[i, 'AGE'] = age_dict.get(id) # age
# for i in range(len(test_data)):
#   id = test_data.loc[i, 'PERSONID2']
#   test_data.at[i, 'SEX'] = sex_dict.get(id) # sex
#   test_data.at[i, 'AGE'] = age_dict.get(id) # age

# train_data['SEX'] = train_data['SEX'].fillna(value=1.0) # fill missing values with 1.0 (sex)
# test_data['SEX'] = test_data['SEX'].fillna(value=1.0)
# train_data['AGE'] = train_data['AGE'].fillna(value=62.0) # fill missing values with mean (age)
# test_data['AGE'] = test_data['AGE'].fillna(value=62.0)
# train_data['AGE'] = scaler.fit_transform(train_data['AGE'].to_numpy().reshape(-1, 1)) # normalize age
# test_data['AGE'] = scaler.transform(test_data['AGE'].to_numpy().reshape(-1, 1)) 
#endregion

# split x and y
def get_accum_duration(df, start, end): # start date - end date = # days
  ans = 0
  for i in range(start, end+1):
    ans += df.loc[i, ['DURATION']]
  return ans
def split_sequence (df, n_steps_back=N_STEPS_BACK, n_steps_forward=N_STEPS_FORWARD): # multistep forecasts
    x, y = [], []
    for i in range(len(df)):
        back_end_ix = i + n_steps_back -1
        forward_end_ix = back_end_ix + n_steps_forward
        if forward_end_ix > len(df)-1:
            break
        seq_x = df.loc[i : back_end_ix, ['B_CRE', 'B_UN']] # modify here if more features are added
        seq_y = df.loc[back_end_ix+1:forward_end_ix, ['B_CRE']]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)
X_train, y_train = split_sequence(train_data)
X_test, y_test = split_sequence(test_data)

# reshape to 2d to save to csv file
y_train_reshaped = y_train.reshape(y_train.shape[0], -1)
y_test_reshaped = y_test.reshape(y_test.shape[0], -1)
y_train_df = pd.DataFrame(y_train_reshaped)
y_test_df = pd.DataFrame(y_test_reshaped)
y_train_df.to_csv('../data/y_train_df.csv') 
y_test_df.to_csv('../data/y_test_df.csv')

X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
X_train_df = pd.DataFrame(X_train_reshaped)
X_test_df = pd.DataFrame(X_test_reshaped)
X_train_df.to_csv('../data/X_train_df.csv')
X_test_df.to_csv('../data/X_test_df.csv')

print(X_train.shape, X_test.shape) # should be [#data points, #n_steps_back, n_features]
print(y_train.shape, y_test.shape) # should be [#data points, #n_steps_back, 1]








