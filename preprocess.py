import csv
import numpy as np
import pandas as pd
from datetime import datetime

# interpolate missing values
# add time duration column
# calculate number_of_measures per patient

# read raw data
data = pd.read_csv('../data/measures.csv', encoding="cp1252") 
B_CRE = data[['PERSONID2', 'LABDATE', 'B_CRE', 'B_UN']]

# Linear interpolation for missing values
B_CRE = B_CRE.interpolate(method ='linear', limit_direction ='forward')
B_CRE.to_csv('../data/processed.csv')

# calculate time duration between two consecutive measurements
B_CRE.insert(len(B_CRE.columns), 'DURATION', 0) # add new column
delete_rows = [] # delete rows with illegal LABDATE
for row in range(B_CRE.shape[0]):
  if not isinstance(B_CRE.iat[row, 1], str):
    print(row, B_CRE.iat[row, 1])
    delete_rows.append(row)
B_CRE = B_CRE.drop(labels = delete_rows, inplace=False) # drop illegal rows

# calculate number of measures per patient
def cal_number_of_measures_per_patient(dataframe, col_name):
  number_of_measures_per_patient = dataframe.groupby(col_name)[col_name].count().to_frame()
  number_of_measures_per_patient.columns = ['Number of Measures']
  return number_of_measures_per_patient

number_of_measures_per_patient = cal_number_of_measures_per_patient(B_CRE, 'PERSONID2')
number_of_measures_per_patient.to_csv('../data/number_of_measures_per_patient.csv')

# find time_0 indices of each patient
# index 0, 95, 95+48, 95+48+113, ....
set_time_0_indices = pd.DataFrame(number_of_measures_per_patient.values.cumsum() , columns=['index'])   
first_index = pd.DataFrame(data={'index': [0]}, columns=['index'])
set_time_0_indices = first_index.append(set_time_0_indices)
set_time_0_indices = set_time_0_indices.drop(labels=(len(number_of_measures_per_patient)-1), axis = 0)

def time_subtraction(date, prev_date):
  diff = datetime.strptime(date, '%Y-%m-%d') - datetime.strptime(prev_date, '%Y-%m-%d')
  diff_days = diff.days
  return diff_days

def calculate_time_duration(dataframe, set_0_indices, ignore_rows):
  for i in range(dataframe.shape[0]):
    # For each patient, sets "Time Duration" at the first measurement to be 0
    if i in set_time_0_indices.values:
      # set time duration to 0 (beginning of this patient)
      dataframe.at[i, 'DURATION'] = 0
    else: # time subtraction
      if not ( isinstance(dataframe.at[i, 'LABDATE'], str) & isinstance(dataframe.at[i-1, 'LABDATE'], str) ):
        dataframe.at[i, 'DURATION'] = 0
      else:
        dataframe.at[i, 'DURATION'] = time_subtraction(date = dataframe.at[i, 'LABDATE'], prev_date = dataframe.at[i-1, 'LABDATE'])
  return dataframe

B_CRE = calculate_time_duration(dataframe = B_CRE, set_0_indices = set_time_0_indices, ignore_rows = delete_rows)
B_CRE = B_CRE.dropna() # assure no NaN values
B_CRE.to_csv('../data/preprocessed_with_duration.csv')

# sort patients based on number_of_measures
# most frequeny -> less frequent
measures_descend = number_of_measures_per_patient.sort_values(by = 'Number of Measures', ascending = False) # descending order
measures_descend.to_csv('../data/measures_descend.csv') # 706 (P214610005308)-> 1 (P214610003882)

# analyze number_of_measures
measures_numpy = number_of_measures_per_patient.loc[:, 'Number of Measures'].to_numpy()
mean_measures = np.mean(measures_numpy)
median_measures = np.median(measures_numpy)
print('Analysis on number_of_meausures:')
print("Mean: ", mean_measures)
print("Median: ", median_measures)
