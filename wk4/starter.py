#!/usr/bin/env python
# coding: utf-8

#get_ipython().system('pip freeze | grep scikit-learn')


import pickle
import pandas as pd
import sys


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df



#2022
year = int(sys.argv[1])

#3
month = int(sys.argv[2])

color = "yellow"

filetoread = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{color}_tripdata_{year:04d}-{month:02d}.parquet'
print(filetoread)

df = read_data(filetoread)

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


standev = y_pred.std()
meanval = y_pred.mean()

print ("Std dev:")
print (standev)

print ("Mean:")
print (meanval)

df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['pred']= y_pred


output_file="first_parquet"

df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)




