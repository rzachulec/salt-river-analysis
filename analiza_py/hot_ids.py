from cProfile import label

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.dates import DateFormatter

# Load the CSV files
df1 = pd.read_csv('hot_ids.csv')  # Replace with your actual CSV file path
df2 = pd.read_csv('../CSV_Load_files/2023_SRP_R.csv')  # Replace with your actual CSV file path

user_ids = df1['customer_id'].tolist()
# user_ids = [33, 36, 93, 94, 105, 113, 116, 129, 138]
# user_ids = [32]
print(user_ids)

all_dataframes = []

def farenheit_to_celsius(x):
    return (x- 32) * 5/9

def celsius_to_fahrenheit(x):
    return x * 9/5 + 32


for year in [2021, 2022, 2023]:
    # Construct the file name for the current year
    file_name = f'{year}_SRP_R.csv'  # Replace 'xxx' with the actual file name pattern
    try:
        # Load the CSV file for the current year
        df = pd.read_csv('../CSV_Load_files/' + file_name)
        user_count = 0
        prev_user = None
        for user_id in user_ids:
            if prev_user == user_id:
                user_count += 1
            else:
                user_count = 0
            prev_user = user_id
            # Extract relevant data for the user from the first DataFrame
            user_data = df1[df1['customer_id'] == user_id]

            # Extract streak times and calculate the 2-hour window
            streak_start_time = pd.to_datetime(user_data['streak_start_time'].values[user_count])
            streak_end_time = pd.to_datetime(user_data['streak_end_time'].values[user_count])
            start_time_window = streak_start_time - pd.Timedelta(hours=2)
            end_time_window = streak_start_time + pd.Timedelta(hours=12)

            # Construct column names for AC_SUB and TEMP based on the year and user ID
            ac_col = f'AC_SUB_{year}_{user_id}'
            temp_col = f'TEMP_{year}_{user_id}'

            # Filter the relevant temperature column for the user in the DataFrame
            df['date_time'] = pd.to_datetime(df['date_time'])

            # Filter data based on the date range
            filtered_df = df[(df['date_time'] >= start_time_window) & (df['date_time'] <= end_time_window)]

            if temp_col in df.columns and ac_col in df.columns:
                filtered_df['temperature'] = farenheit_to_celsius(filtered_df['temperature'])
                filtered_df[temp_col] = farenheit_to_celsius(filtered_df[temp_col])

                filtered_df['user_id'] = user_id

                filtered_df = filtered_df[['temperature', 'date_time', temp_col, ac_col, 'user_id']]

                if filtered_df[temp_col].nunique() > 0:
                    # Append the filtered dataframe to the list
                    all_dataframes.append(filtered_df)

    except FileNotFoundError:
        print(f"File {file_name} not found. Skipping this year.")


# Concatenate all dataframes into a single dataframe
collective_df = pd.concat(all_dataframes, ignore_index=True)
collective_df = collective_df.sort_values(by='date_time').reset_index()


# Plot the collective data

# for user_id in user_ids:
#     user_df = collective_df[collective_df['user_id'] == user_id]
#     plt.scatter(user_df['date_time'], user_df[f'TEMP_2023_{user_id}'], marker=None, linestyle='-'
#                  , c=user_df[f'AC_SUB_2023_{user_id}'], cmap='bwr'
#                  )
#     plt.plot(user_df['date_time'], user_df['temperature'], linestyle='--', label=f'User ID: {user_id}')
for user_id in user_ids:
    for year in [2021, 2022, 2023]:
        temp_col = f'TEMP_{year}_{user_id}'
        user_df = collective_df[collective_df['user_id'] == user_id]
        if temp_col in user_df.columns:
            fig = plt.figure(figsize=(12, 6))
            ax1 = fig.add_subplot(1, 1, 1)
            plt.title(f'Internal temperature vs external temperature over time, Date: {user_df['date_time'].iloc[0].date()}, Customer ID: {user_id}')

            sc1 = ax1.scatter(user_df['date_time'],
                              user_df[f'TEMP_{year}_{user_id}'],
                              c=user_df[f'AC_SUB_{year}_{user_id}'], cmap='bwr')

            ax1.plot(user_df['date_time'],
                     user_df['temperature'],
                     linestyle='-',
                     label='External temperature')

            ax1.set(ylabel = 'Temperature [°C]',
                    xlabel = 'Date and time')
            ax1.grid(True)
            ax1.legend()

            plt.colorbar(sc1, label='AC load [kWh]', pad=0.1)
            ax11 = ax1.twinx()
            ax11.set(ylabel = 'Temperature [°F]',
                    ylim = (celsius_to_fahrenheit(ax1.get_ylim()[0]),
                            celsius_to_fahrenheit(ax1.get_ylim()[1])))

            ax11.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))

            plt.show()
