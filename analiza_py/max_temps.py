import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates



plt.rcParams["font.family"] = "serif"
xformatter = mdates.DateFormatter('%H')

def plot_max_temps_scatter(max_temps_file: str, streak_maxtemp_file: str):
    """
    Load 'max_temps_AC_on.csv' and 'Streak-maxtemp.csv', and plot a scatter plot
    of the maximum temperatures that occurred at the same time.

    Parameters:
        max_temps_file (str): Path to the 'max_temps_AC_on.csv' file.
        streak_maxtemp_file (str): Path to the 'Streak-maxtemp.csv' file.
    """
    # Load the data from the CSV files
    max_temps_df = pd.read_csv(max_temps_file)
    streaks_df = pd.read_csv(streak_maxtemp_file)

    # Ensure that 'date_time' and 'streak_end_time' columns are in datetime format
    max_temps_df['date_time'] = pd.to_datetime(max_temps_df['date_time'])
    streaks_df['streak_end_time'] = pd.to_datetime(streaks_df['streak_end_time'])

    print('loaded files')

    # Extract all columns except 'date_time' and melt them into a single 'temperature_ac' column
    melted_max_temps_df = max_temps_df.melt(id_vars=['date_time'], value_name='temperature_ac').drop('variable', axis=1)

    print('melted cols')


    # Merge the melted dataframes on matching date and time
    merged_df = pd.merge(
        melted_max_temps_df, streaks_df, left_on='date_time', right_on='streak_end_time', how='inner', suffixes=('_ac', '_streak')
    )
    merged_df['time'] = merged_df['date_time'].dt.strftime('%H:%M').astype(str)
    merged_df.drop_duplicates(subset=['customer_id', 'streak_id'], inplace=True)
    merged_df = merged_df.sort_values('time').reset_index(drop=True)

    print('merged dfs')

    hourly_avg_noac = merged_df.groupby(merged_df['time'])['max_temperature'].mean().reset_index()
    hourly_avg_noac.columns = ['time', 'avg_max_temperature']
    # hourly_avg_noac['time'] = pd.to_time(hourly_avg_noac['time'], errors='ignore')

    hourly_avg_ac = merged_df.groupby(merged_df['time'])['temperature_ac'].mean().reset_index()
    hourly_avg_ac.columns = ['time', 'avg_max_ac_temperature']
    # hourly_avg_ac['time'] = pd.to_time(hourly_avg_ac['time'], errors='ignore')

    # Plot a scatter plot with maximum temperatures from both dataframes
    fig, ax1 = plt.subplots(figsize=(9, 4))

    # Plotting in Celsius
    # ax1.scatter(merged_df['time'], merged_df['temperature_ac'], color='blue', alpha=0.01, label='Max Temps AC on')
    ax1.plot(hourly_avg_ac['time'], hourly_avg_ac['avg_max_ac_temperature'], 'b--', label='Average maximum temperature (AC on)')

    ax1.scatter(merged_df['time'], merged_df['max_temperature'], color='red', alpha=0.1, label='Max Temps AC OFF')
    ax1.plot(hourly_avg_noac['time'], hourly_avg_noac['avg_max_temperature'], 'r--', label='Average maximum temperature (AC OFF)')

    # Labels, title, and legend for the primary y-axis (Celsius)
    ax1.set_ylabel('Maximum temperature [°C]', color='black')
    ax1.set_xlabel('Hour of day')
    ax1.legend(loc='upper left')
    ax1.grid(axis='y')
    ax1.set_title('Maximum indoor temperatures with AC On and AC Off, July 2023 8:00-21:00')

    # Create a secondary y-axis for Fahrenheit
    ax2 = ax1.twinx()
    ax2.set_ylabel('Maximum temperature [°F]', color='black')

    hour_ticks = pd.date_range(start='08:00', end='21:00', freq='h').strftime('%H:%M')
    ax1.set_xticks(hour_ticks)
    ax1.set_xticklabels(hour_ticks)
    # ax1.xaxis.set_major_formatter(xformatter)

    # Set the limits for the secondary y-axis to match the Celsius scale
    def celsius_to_fahrenheit(x):
        return x * 9/5 + 32

    ax2.set_ylim(celsius_to_fahrenheit(ax1.get_ylim()[0]), celsius_to_fahrenheit(ax1.get_ylim()[1]))

    # Ensure the tick marks correspond correctly between the two scales
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))

    # Display the plot
    plt.show()


# Example usage:
plot_max_temps_scatter('max_temps_AC_on.csv', 'Streak-maxtemp.csv')