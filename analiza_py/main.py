import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

csv_files = ['2021_SRP_R.csv', '2022_SRP_R.csv', '2023_SRP_R.csv']
csv_path = '../CSV_Load_files'

rates_df = pd.read_csv('customer_id_rate.csv')
rates_df = rates_df.groupby(['CUSTOMER_KEY', 'RATE'], as_index=False).mean()

info_df = pd.read_csv('../CSV_Load_files/SRP_HHinfo.csv')

rates_df = rates_df.merge(info_df, on='CUSTOMER_KEY')

rates_cleaned_df = rates_df[['RATE', 'CustomerID']]

rates_cleaned_df.columns = ['RATE', 'customer_id']

rates_cleaned_df = rates_cleaned_df.drop_duplicates(subset='customer_id')

print('Rate counts:')
print(rates_cleaned_df['RATE'].value_counts())


def mark_four_consecutive_zeros(series):
    n = len(series)
    streak = np.zeros(n, dtype=bool)
    streak_id = np.full(n, np.nan)

    current_streak = 0
    for i in range(n):
        if series.iloc[i] == 0:  # Use .iloc[i] to access element by position
            current_streak += 1
            if current_streak >= 4:
                streak[i - current_streak + 1:i + 1] = True
                streak_id[i - current_streak + 1:i + 1] = streak_id[i - current_streak] if i - current_streak >= 0 else 1
        else:
            if current_streak >= 4:
                streak_id[i - current_streak:i] = np.nanmax(streak_id) + 1 if np.nanmax(streak_id) >= 0 else 1
            current_streak = 0

    # Handle case if streak continues till the end
    if current_streak >= 4:
        streak_id[n - current_streak:n] = np.nanmax(streak_id) + 1 if np.nanmax(streak_id) >= 0 else 1

    return streak, streak_id


def filter_daytime_streaks(results_df):
    # Filter streaks where both 'start_time' and 'end_time' are between 8:00 and 21:00
    filtered_df = results_df[
        (results_df['start_time'].dt.hour >= 8) & (results_df['start_time'].dt.hour < 21) &
        (results_df['end_time'].dt.hour >= 8) & (results_df['end_time'].dt.hour < 21)
    ]
    return filtered_df


def filter_afternoon_streaks(results_df):
    # Filter streaks where both 'start_time' and 'end_time' are between 14:00 and 18:00
    filtered_df = results_df[
        (results_df['start_time'].dt.hour >= 14) & (results_df['start_time'].dt.hour < 18) &
        (results_df['end_time'].dt.hour >= 14) & (results_df['end_time'].dt.hour < 18)
    ]
    return filtered_df


def compute_streaks(customer_ids, year, df):
    results = []
    # Step 4: Filter the rows for each customer where the 'AC_SUB' column is 0
    for customer_id in customer_ids:
        ac_col = f'AC_SUB_{year}_{customer_id}'
        bill_col = f'BILL_{year}_{customer_id}'
        temp_col = f'TEMP_{year}_{customer_id}'

        # Check if all required columns exist
        if ac_col in df.columns and bill_col in df.columns and temp_col in df.columns:
            customer_data = df[["date_time", "temperature", ac_col, bill_col, temp_col]].copy()

            customer_data[temp_col] = (customer_data[temp_col] - 32) * 5.0 / 9.0

            # Mark entire streak of 4 or more consecutive zeros
            customer_data['streak'], customer_data['streak_id'] = mark_four_consecutive_zeros(customer_data[ac_col])

            if customer_data['streak'].notna().any():
                # Filter rows that are part of the streak
                streak_data = customer_data[customer_data['streak'].notna()].copy()

                # Group by streak IDs
                streak_groups = streak_data.groupby('streak_id')

                for _, group in streak_groups:
                    if len(group) >= 4:
                        full_hours = int((len(group) - len(group) % 4)/4)
                        part_hours = len(group) % 4
                        for i in range(full_hours):
                            start_index = i*4
                            end_index = i*4+3
                            start_temp = group[temp_col].iloc[start_index]
                            end_temp = group[temp_col].iloc[end_index]
                            max_temp = np.max(group[temp_col].loc[:])
                            external_temp = np.average(group['temperature'].iloc[start_index:end_index])
                            bill = group[bill_col].iloc[start_index]
                            time_diff = 1  # Convert time difference to hours

                            temp_increase_rate = (end_temp - start_temp) / time_diff
                            results.append({
                                'customer_id': customer_id,
                                'external_temperature': external_temp,
                                'temp_increase_rate': temp_increase_rate,
                                'internal_temperature': start_temp,
                                'internal_temperature_end': end_temp,
                                'max_temperature': max_temp,
                                'start_time': group['date_time'].iloc[start_index],
                                'end_time': group['date_time'].iloc[end_index],
                                'bill': bill
                            })
                        if part_hours != 0:
                            start_index = full_hours*4
                            end_index = full_hours*4 + part_hours - 1
                            start_temp = group[temp_col].iloc[start_index]
                            end_temp = group[temp_col].iloc[end_index]
                            max_temp = np.max(group[temp_col].loc[:])
                            external_temp = group['temperature'].iloc[end_index]
                            bill = group[bill_col].iloc[start_index]
                            time_diff = part_hours * 15 / 60  # Convert time difference to hours

                            temp_increase_rate = (end_temp - start_temp) / time_diff
                            results.append({
                                'customer_id': customer_id,
                                'external_temperature': external_temp,
                                'temp_increase_rate': temp_increase_rate,
                                'internal_temperature': start_temp,
                                'internal_temperature_end': end_temp,
                                'max_temperature': max_temp,
                                'start_time': group['date_time'].iloc[full_hours*4],
                                'end_time': group['date_time'].iloc[end_index],
                                'bill': bill
                            })
    return results


dfs = []
dfs_day = []
dfs_afternoon = []

df_full = []


def create_streak_csv():
    for file in csv_files:

        year = file[:4]
        df = pd.read_csv('../CSV_Load_files/' + file)

        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        df['temperature'] = (df['temperature'] - 32) * 5.0 / 9.0

        initial_columns = ['date_time', 'temperature', 'humidity']
        initial_data = df[initial_columns]
        customer_columns = [col for col in df.columns if col.startswith(( 'AC_SUB_', 'BILL_', 'TEMP_' ))]

        # Step 3: Extract the customer IDs from the column names
        customer_ids = list(set(col.split('_')[-1] for col in customer_columns if col.startswith('AC_SUB_')))

        customer_ids.sort()
        # Create a new DataFrame to store the filtered data
        result = initial_data.copy()

        results = compute_streaks(customer_ids, year, df)

        results_df = pd.DataFrame(results)

        results_df['customer_id'] = results_df['customer_id'].astype(int)

        results_df = results_df.merge(rates_cleaned_df, on='customer_id', how='left')

        results_df['start_time'] = pd.to_datetime(results_df['start_time'])
        results_df['end_time'] = pd.to_datetime(results_df['end_time'])

        results_day_df = filter_daytime_streaks(results_df)

        results_afternoon_df = filter_afternoon_streaks(results_df)

        dfs.append(results_df)
        dfs_day.append(results_day_df)
        dfs_afternoon.append(results_afternoon_df)

    results_concat_df = pd.concat(dfs, ignore_index=True)
    results_concat_day_df = pd.concat(dfs_day, ignore_index=True)
    results_concat_afternoon_df = pd.concat(dfs_afternoon, ignore_index=True)

    results_concat_df.to_csv('temperature_increase_analysis.csv', index=False)
    results_concat_day_df.to_csv('temperature_increase_analysis_day.csv', index=False)
    results_concat_afternoon_df.to_csv('temperature_increase_analysis_afternoon.csv', index=False)

    print('done saving streak csvs')


# create_streak_csv()


def find_non_off_days():
    final_df = pd.DataFrame(columns=['customer_id', 'date_time', 'max_temperature'])

    ac_off_ids = pd.read_csv('IDs_July23_AC_OFF.csv')

    ac_off_ids_list = ac_off_ids['customer_id'].to_list()
    ac_off_ids_list.sort()


    df = pd.read_csv('../CSV_Load_files/2023_SRP_R.csv')

    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    df['temperature'] = (df['temperature'] - 32) * 5.0 / 9.0

    # Assuming 'df' is your original DataFrame
    # Filter for date_time in July or August 2023 between 2 PM and 6 PM
    df['date_time'] = pd.to_datetime(df['date_time'])  # Ensure date_time is in datetime format

    # Filter for date_time in July or August 2023 between 2 PM and 6 PM
    df_filtered = df[(df['date_time'].dt.month.isin([7])) &  # July or August
                     (df['date_time'].dt.year == 2023) &  # Year 2023
                     (df['date_time'].dt.hour >= 8) &
                     (df['date_time'].dt.hour < 21)] # Between 2 PM and 6 PM

    # Initialize a list to collect results
    results = []

    # Extract unique customer IDs from the column names
    customer_ids1 = [col.split('_')[-1] for col in df.columns if (col.startswith('AC_SUB_'))]

    customer_ids1 = [int(customer_id) for customer_id in customer_ids1 if int(customer_id) not in ac_off_ids_list]

    year = '2023'
    # Iterate through each customer ID
    for customer_id in customer_ids1:
        # Define column names for AC, BILL, and TEMP
        ac_col = f'AC_SUB_{year}_{customer_id}'
        temp_col = f'TEMP_{year}_{customer_id}'

        # Check if all required columns are present in the DataFrame
        if ac_col in df_filtered.columns and temp_col in df_filtered.columns:
            # Filter the DataFrame where AC is not 0 for this customer
            customer_df = df_filtered[df_filtered[ac_col] != 0].copy()

            # Keep only date_time and temperature columns
            customer_df = customer_df[['date_time', temp_col]]

            # Convert temperature to Celsius
            customer_df[temp_col] = (customer_df[temp_col] - 32) * 5 / 9

            # Group by date and get the maximum temperature for each day
            customer_df_max = customer_df.groupby('date_time')[temp_col].max().reset_index()

            # Merge this customer's data into the final dataframe on 'date'
            if final_df.empty:
                final_df = customer_df_max
            else:
                # Use 'date' as the key for merging; keep the date unique
                final_df = pd.merge(final_df, customer_df_max, on='date_time', how='outer')

    # Sort the final dataframe by date
    final_df = final_df.sort_values('date_time').reset_index(drop=True)

    # Display the resulting DataFrame
    print(final_df)

    final_df.to_csv('max_temps_AC_on.csv', index=False)



find_non_off_days()
