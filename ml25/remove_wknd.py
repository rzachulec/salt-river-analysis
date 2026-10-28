import pandas as pd

df = pd.read_csv('../aggregate_data_f_cost.csv', )

df['date'] = pd.to_datetime(df['date'])


df.to_csv('../aggregate_data_f_cost_weekdays.csv', index=False)