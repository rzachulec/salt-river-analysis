import pandas as pd

data_df = pd.read_csv('aggregate_data_f.csv')

cost_df = pd.read_csv('./CSV_Load_files/ac_daily_all(in).csv')
cost_df['date'] = pd.to_datetime(cost_df['DATE'], format='%m/%d/%Y').dt.strftime('%Y-%m-%d')
cost_df = cost_df.drop(columns=['month', 'DATE', 'RATE'])

merged_df = pd.merge(data_df, cost_df, on=['CustomerID', 'date'], how='inner')



print(merged_df.head(10))

merged_df.to_csv('aggregate_data_f_cost.csv', index=False)