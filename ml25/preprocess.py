import pandas as pd

csv_files = ['2021_SRP_R.csv', '2022_SRP_R.csv', '2023_SRP_R.csv']
csv_path = '../CSV_Load_files'

rates_df = pd.read_csv('./ml25/customer_id_rate.csv')
rates_df = rates_df.groupby(['CUSTOMER_KEY', 'RATE'], as_index=False).mean()

info_df = pd.read_csv('./CSV_Load_files/SRP_HHinfo.csv')

rates_df = rates_df.merge(info_df, on='CUSTOMER_KEY')
print(rates_df.columns)

all_customers_data = pd.DataFrame()

for file in csv_files:
    year = file[:4]
    df = pd.read_csv('./CSV_Load_files/' + file)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # df['temperature'] = (df['temperature'] - 32) * 5.0 / 9.0
    
    initial_columns = ['date_time', 'temperature', 'humidity']
    initial_data = df[initial_columns]
    customer_columns = [col for col in df.columns if col.startswith(( 'AC_SUB_', 'BILL_', 'TEMP_' ))]
    customer_ids = list(set(col.split('_')[-1] for col in customer_columns if col.startswith('AC_SUB_')))
    customer_ids = [int(id) for id in customer_ids]
    customer_ids.sort()
    print("Number of participants in " + year + ": ", len(customer_ids))
    print("Processing user data...")
    
    for customer_id in customer_ids:
        ac_col = f'AC_SUB_{year}_{customer_id}'
        bill_col = f'BILL_{year}_{customer_id}'
        temp_col = f'TEMP_{year}_{customer_id}'

        # Check if all required columns exist
        if ac_col in df.columns and bill_col in df.columns and temp_col in df.columns:
            customer_data = df[["date_time", "temperature", ac_col, bill_col, temp_col]].copy()
            # customer_data[temp_col] = (customer_data[temp_col] - 32) * 5.0 / 9.0
            # Convert date_time to datetime and extract date
            customer_data['date_time'] = pd.to_datetime(customer_data['date_time'], format='mixed')
            customer_data['date'] = customer_data['date_time'].dt.date
            customer_data['hour'] = customer_data['date_time'].dt.hour

            # Group by date to calculate daily aggregates
            daily_aggregates = customer_data.groupby('date', group_keys=False).apply(lambda group: pd.Series({
                'mean_external_temp': group['temperature'].mean(),
                'mean_internal_temp': group[temp_col].mean(),
                'std_internal_temp': group[temp_col].std(),
                'mean_ac_use': group[ac_col].mean(),
                'std_ac_use': group[ac_col].std(),
                'avg_daytime_temp': group[(group['hour'] >= 8) & (group['hour'] < 20)][temp_col].mean(),
                'avg_night_temp': group[(group['hour'] >= 20) | (group['hour'] < 8)][temp_col].mean(),
                'temp_range': group[temp_col].max() - group[temp_col].min(),
                'fraction_above_27': (group[temp_col] > 80).mean(),
                'fraction_below_24': (group[temp_col] < 75).mean(),
                'AC_usage_correlation': group[temp_col].corr(group[ac_col]) if group[ac_col].nunique() > 1 else None
            }), include_groups=False).reset_index()
            

            # Add customer ID to daily aggregates
            daily_aggregates['CustomerID'] = customer_id

            # Select only relevant columns from rates_df
            selected_columns = ['CustomerID', 'RATE', 'HH_Age', 'HH_Income_Desc', 'HH_NumberOfPersons', 
                                'LVBL_SQ_FT']
            filtered_rates_df = rates_df[selected_columns].copy()
            filtered_rates_df['HH_Income_Desc'] = filtered_rates_df['HH_Income_Desc'].str.extract(r'\$(\d+),')[0].astype(float).astype('Int64')

            # Merge with filtered rates_df to include customer information
            merged_data = daily_aggregates.merge(filtered_rates_df, on='CustomerID', how='left')
            # Discard rows with NaN values in any column
            
            merged_data = merged_data.dropna()
            all_customers_data = pd.concat([all_customers_data, merged_data], ignore_index=True)
                        
all_customers_data.to_csv("aggregate_data_f.csv", index=False)

    