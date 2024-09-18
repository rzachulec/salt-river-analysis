import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

plt.rcParams["font.family"] = "serif"

# load main data
results_df = pd.read_csv('temperature_increase_analysis_day.csv')
# results_df = results_df[results_df['customer_id'] != 33]

results_df['start_time'] = pd.to_datetime(results_df['start_time'])
results_df['end_time'] = pd.to_datetime(results_df['end_time'])

results_df = results_df[(
                         results_df['start_time'].dt.to_period('M').astype(str).str.startswith('2023')
                         # | results_df['start_time'].dt.to_period('M').astype(str).str.startswith('2022-08')
                         # | results_df['start_time'].dt.to_period('M').astype(str).str.startswith('2023-09')
                         )]


results_df_noduplicateid = results_df.drop_duplicates(subset=['customer_id', 'RATE']).reset_index(drop=True)
noduplicateid = results_df_noduplicateid['customer_id']

print(noduplicateid)

noduplicateid.to_csv('IDs_July23_AC_OFF.csv', index=False)

print("Number of households with AC load = 0 kWh for more than an hour between 2PM and 6PM: ",
      results_df_noduplicateid['customer_id'].nunique())

rate_counts = results_df_noduplicateid['RATE'].value_counts()

print(rate_counts)


def plot_rate_vs_time(df):
    df1 = df[(df['temp_increase_rate'] >= -5) & (df['temp_increase_rate'] <= 10)]
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['hour'] = df['start_time'].dt.hour

    hourly_avg_temp = df.groupby('hour')['external_temperature'].mean().reset_index()

    hourly_avg_temp.columns = ['hour', 'avg_external_temp']
    print(hourly_avg_temp)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('time of the day')
    ax1.set_ylabel('internal temperatute increase rate [degC/h]')

    ax2 = ax1.twinx()
    ax2.set_ylabel('average external temperatute [degC]')
    ax2.tick_params(axis='y', color='blue')

    sc1 = ax1.scatter(df1['start_time'].dt.hour, df1['temp_increase_rate'], alpha=0.2, s=17.5, c=df1['RATE'],
                      cmap='rainbow')
    ax2.plot(hourly_avg_temp['hour'], hourly_avg_temp['avg_external_temp'])

    # fig.tight_layout()
    plt.title('Average external temperature vs temperature increase rate with AC off')
    plt.colorbar(sc1, ticks=[21, 23, 26], pad=0.12)
    plt.show()


def plot_results(df):
    # Calculate mean and standard deviation of 'temp_increase_rate'
    mean_temp_increase = df['temp_increase_rate'].mean()
    std_temp_increase = df['temp_increase_rate'].std()

    # Define lower and upper bounds for outliers (3 standard deviations away from the mean)
    lower_bound = mean_temp_increase - 5 * std_temp_increase
    upper_bound = mean_temp_increase + 5 * std_temp_increase

    # Remove outliers
    filtered_df = df[(df['temp_increase_rate'] >= lower_bound) &
                     (df['temp_increase_rate'] <= upper_bound)]

    # Compute correlations
    pearson_corr, pearson_pval = pearsonr(filtered_df['external_temperature'], filtered_df['temp_increase_rate'])
    spearman_corr, spearman_pval = spearmanr(filtered_df['external_temperature'], filtered_df['temp_increase_rate'])

    plt.figure(figsize=(10, 6))
    sc1 = plt.scatter(filtered_df['external_temperature'], filtered_df['temp_increase_rate'], alpha=0.7, s=4.5,
                      c=filtered_df['RATE'], cmap='rainbow')
    # plt.colorbar(sc1, ticks=[21, 23, 26])
    plt.title('Rate of Internal Temperature Increase vs. External Temperature')
    plt.xlabel('External Temperature (°C)')
    plt.ylabel('Rate of Temperature Increase (°C/hour)')
    plt.grid(True)

    # Display Pearson and Spearman correlation coefficients on the plot
    textstr = '\n'.join((
        f'Pearson correlation: {pearson_corr:.2f}',
        f'Spearman correlation: {spearman_corr:.2f}',
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=props)

    plt.show()


def plot_results_ratesplit(df):
    df1 = df[(df['temp_increase_rate'] >= -2) & (df['temp_increase_rate'] <= 150)]

    fig, (rate1, rate2, rate3) = plt.subplots(3, 1, sharex=True)

    rate1_df = df1[df1['RATE'] == 21]
    rate2_df = df1[df1['RATE'] == 23]
    rate3_df = df1[df1['RATE'] == 26]

    rate1.scatter(rate1_df['start_time'].dt.hour, rate1_df['temp_increase_rate'], alpha=0.4, s=17.5,
                  c=rate1_df['external_temperature'], cmap='rainbow')
    rate1.grid()
    rate1.set_ylabel('Plan 21')

    sc2 = rate2.scatter(rate2_df['start_time'].dt.hour, rate2_df['temp_increase_rate'], alpha=0.4, s=17.5,
                        c=rate2_df['external_temperature'], cmap='rainbow')
    rate2.grid()
    rate2.set_ylabel('Plan 23')

    rate3.scatter(rate3_df['start_time'].dt.hour, rate3_df['temp_increase_rate'], alpha=0.4, s=17.5,
                  c=rate3_df['external_temperature'], cmap='rainbow')
    rate3.grid()
    rate3.set_ylabel('Plan 26')

    fig.suptitle('AC off for > 1 h events split by payment plan')
    fig.supxlabel('Time of day [h]')
    fig.supylabel('Temperature increase rate [deg C/h]')
    fig.colorbar(sc2, ax=(rate1, rate2, rate3), label='External temperature [deg. C]')

    plt.show()


def plot_temp_vs_time(df):
    plt.figure(figsize=(12, 8))
    filtered_df = df[df['streak_length'] < 48]
    sc1 = plt.scatter(filtered_df['external_temperature'], filtered_df['internal_temperature_end'],
                      c=filtered_df['streak_length'], cmap='gist_stern')
    cbar1 = plt.colorbar(sc1)
    cbar1.set_label('AC off duration [x15 min]')
    plt.title('Internal Temperature vs. External Temperature with Increase Rate')
    plt.xlabel('External Temperature (°C)')
    plt.ylabel('Internal Temperature (°C)')
    plt.grid(True)

    plt.show()


def plot_avg_temp_increase(df):
    # Group by 'external_temperature' and calculate mean 'temp_increase_rate'
    avg_temp_increase = df.groupby('external_temperature')['temp_increase_rate'].mean()
    std_err = np.sum(df.groupby('external_temperature')['temp_increase_rate'].std()) / np.sqrt(
        len(df.index))

    avg_temp_increase = avg_temp_increase.dropna()
    # Compute linear regression (line of best fit)
    m, b = np.polyfit(avg_temp_increase.index, avg_temp_increase.values, 1)

    # Calculate residuals (vertical distances from each data point to the line)
    residuals = avg_temp_increase.values - (m * avg_temp_increase.index + b)

    # Define outlier threshold (e.g., 2 * standard deviation of residuals)
    outlier_threshold = 3 * np.std(residuals)

    # Filter out data points beyond the outlier threshold
    filtered_indices = np.abs(residuals) <= outlier_threshold
    avg_temp_increase_no_outliers = avg_temp_increase[filtered_indices]
    residuals_no_outliers = avg_temp_increase_no_outliers.values - (m * avg_temp_increase_no_outliers.index + b)

    # Calculate SS_res (sum of squares of residuals)
    ss_res = np.sum(residuals_no_outliers ** 2)

    # Calculate SS_tot (total sum of squares)
    ss_tot = np.sum((avg_temp_increase_no_outliers.values - np.mean(avg_temp_increase_no_outliers.values)) ** 2)

    r_squared = 1 - (ss_res / ss_tot)

    # Compute correlations (optional)
    pearson_corr, pearson_pval = pearsonr(avg_temp_increase_no_outliers.index, avg_temp_increase_no_outliers.values)
    spearman_corr, spearman_pval = spearmanr(avg_temp_increase_no_outliers.index, avg_temp_increase_no_outliers.values)

    # Plotting the average temperature increase against external temperature
    plt.figure(figsize=(10, 6))
    plt.scatter(avg_temp_increase_no_outliers.index, avg_temp_increase_no_outliers.values, marker='o', linestyle='-',
                color='b')

    # Plot the regression line
    plt.plot(avg_temp_increase_no_outliers.index, m * avg_temp_increase_no_outliers.index + b, color='r',
             linestyle='--')

    # Customize plot
    plt.suptitle('Average Temperature Increase vs. External Temperature')
    plt.title('In case of AC load = 0 for more than one hour, only between 8AM and 10PM.', fontsize=10)
    plt.xlabel('External Temperature (°C)')
    plt.ylabel('Average Temperature Increase (°C/hour)')
    plt.grid(True)

    textstr = '\n'.join((
        f'Pearson correlation: {pearson_corr:.2f}',
        f'Spearman correlation: {spearman_corr:.2f}',
        f'Standard Error: {std_err:.2f}',
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=props)

    # Show plot
    plt.show()


def plot_rates_vs_time(df, rate_counts):
    rate_count = pd.DataFrame(rate_counts)
    peak_prices = pd.DataFrame([(23, 0.1333), (21, 0.3620), (26, 0.2585)], columns=('RATE', 'peak_price'))
    # num_customers = df.groupby('RATE')['customer_id'].value_counts()
    num_streaks = pd.DataFrame(df['RATE'].value_counts())
    print(num_streaks)
    rate_count['streaks'] = num_streaks['count']
    rate_count['average_hours'] = rate_count['streaks'] / rate_count['count']

    print(rate_count)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Scatter plot with a colormap based on index
    sc = ax.scatter(peak_prices['peak_price'], rate_count['average_hours'], c=rate_counts, cmap='Reds')

    for i in range(3):
        ax.annotate(peak_prices['RATE'].iloc[i],
                    (peak_prices['peak_price'].iloc[i], rate_count['average_hours'].iloc[i]), xytext=(0, 5),
                    textcoords='offset pixels')

    # Add labels and title
    ax.set_xlabel('Peak Price per kWh')
    ax.set_ylabel('Average AC off time [h]')
    ax.set_title('Pricing Plan vs average AC off time (>1h)')

    cbar1 = plt.colorbar(sc)
    cbar1.set_label('Number of customers with given plan')

    plt.show()


def plot_area_hours(df, rate_counts):
    rate_count = pd.DataFrame(rate_counts)

    df['hour'] = df['start_time'].dt.hour

    df = df[df['RATE'] != 22]

    # Group by RATE and hour and calculate counts
    grouped = df.groupby(['RATE', 'hour']).size().reset_index(name='count')
    print(grouped)

    # Normalize counts by total count for each RATE
    grouped['normalized_count'] = grouped.apply(
        lambda row: row['count'] / rate_count.loc[row['RATE'], 'count'], axis=1
    )

    # Pivot the data to get hours as rows and RATEs as columns
    pivot_df = grouped.pivot(index='hour', columns='RATE', values='normalized_count').fillna(0)

    # Reorder the pivot table columns to match the desired order
    rate_order = [23, 21, 26]
    pivot_df = pivot_df[rate_order]

    # Define colors for the plot
    colors = {
        23: 'blue',  # RATE 23 in blue
        # 22: '#FF9999',  # Light red
        21: '#FF6666',  # Medium red
        26: '#CC0000',  # Dark red
    }

    # Plot the area plot
    pivot_df.plot(kind='line', stacked=False, color=[colors[rate] for rate in rate_order])
    plt.title('AC off time > 1 hour, during July 2023 heatwave')
    plt.xlabel('Hour of day')
    plt.ylabel('Average AC off time [h]')
    plt.show()


def farenheit_to_celsius(x):
    return (x- 32) * 5/9


def celsius_to_fahrenheit(x):
    return x * 9/5 + 32


def plot_max_temp_vs_streak_time(df):

    temperature_df = pd.read_csv('../CSV_Load_files/2023_SRP_R.csv')

    temperature_df['date_time'] = pd.to_datetime(temperature_df['date_time'], errors='coerce')

    # Step 1: Sort by 'RATE' and 'start_time'
    df = df.sort_values(by=['customer_id', 'start_time']).reset_index(drop=True)

    df = df[(df['start_time'].dt.to_period('M').astype(str).str.startswith('2023-07')
             # | df['start_time'].dt.to_period('M').astype(str).str.startswith('2023-07')
             )]

    df['is_new_streak'] = (
            df['start_time'] != df.groupby('customer_id')['end_time'].shift() + pd.Timedelta(minutes=15)).astype(
        int)

    df['streak_id'] = df.groupby('customer_id')['is_new_streak'].cumsum()

    streaks = df.groupby(['customer_id', 'streak_id']).agg(
        streak_length=('start_time', lambda x: (df.loc[x.index, 'end_time'].max() - df.loc[
            x.index, 'end_time'].min()).total_seconds() / 3600 + 1),  # Length in minutes
        max_temperature=('max_temperature', 'max'),
        start_internal_temp=('start_time', lambda x: df.loc[x.idxmin(), 'internal_temperature']),
        # Internal temperature at the start of the streak
        streak_start_time=('start_time', 'min'),  # First 'start_time' of the streak
        # End time of the streak
        streak_end_time=('end_time', 'max'),
    ).reset_index()

    streaks['temp_delta'] = streaks['max_temperature'] - streaks['start_internal_temp']

    # Step 3: Calculate average temperature for each streak
    avg_temperatures = []

    for _, row in streaks.iterrows():
        # Extract customer_id and year from streak
        customer_id = row['customer_id']
        streak_year = row['streak_start_time'].year

        # Determine the relevant temperature column
        temp_column_name = f"TEMP_{streak_year}_{customer_id}"

        # Check if the column exists in temperature_df
        if temp_column_name not in temperature_df.columns:
            avg_temperatures.append(None)  # If no relevant data, append None
            continue

        # Filter temperature data for the specific streak duration
        mask = (temperature_df['date_time'] >= row['streak_start_time']) & (
                    temperature_df['date_time'] <= row['streak_end_time'])

        # Calculate average temperature using the relevant column
        avg_temp = temperature_df.loc[mask, temp_column_name].mean()
        avg_temp = farenheit_to_celsius(avg_temp)
        avg_temperatures.append(avg_temp)

    # Add the calculated average temperatures to the streaks DataFrame
    streaks['avg_temperature'] = avg_temperatures  # Add to streaks DataFrame

    # streaks = streaks.loc[streaks['streak_length'] >= 1].reset_index(drop=True)
    # streaks.replace("nan", np.nan, inplace=True)
    streaks.dropna(inplace=True)

    hot_id = streaks[(streaks['temp_delta'] > 4) & (streaks['max_temperature'] > 31)].reset_index(drop=True)
    print('hot IDs:', hot_id)
    hot_id.to_csv('hot_ids.csv', index=False)

    hourly_avg = streaks.groupby(streaks['streak_length'])['max_temperature'].mean().reset_index()
    number_of_streaks = streaks.groupby(streaks['streak_length'])['streak_length'].value_counts().reset_index()
    # number_of_streaks.columns = ['streak_length', 'count']
    print(number_of_streaks)
    hourly_avg.columns = ['hour', 'avg_max_temperature']
    hourly_avg_delta = streaks.groupby(streaks['streak_length'])['temp_delta'].mean().reset_index()
    hourly_avg_delta.columns = ['hour', 'avg_delta']

    # Calculate averages for plotting
    hourly_avg = streaks.groupby(streaks['streak_length'])['max_temperature'].mean().reset_index()
    hourly_avg.columns = ['hour', 'avg_max_temperature']

    number_of_streaks = streaks.groupby(streaks['streak_length'])['streak_length'].value_counts().reset_index()
    print(number_of_streaks)

    hourly_avg_delta = streaks.groupby(streaks['streak_length'])['temp_delta'].mean().reset_index()
    hourly_avg_delta.columns = ['hour', 'avg_delta']

    hourly_avg_temp = streaks.groupby(streaks['streak_length'])['avg_temperature'].mean().reset_index()
    hourly_avg_temp.columns = ['hour', 'avg_temperature']

    streaks.to_csv('Streak-maxtemp.csv')

    def celsius_to_fahrenheit(x):
        return x * 9/5 + 32

    def celsius_to_fahrenheit_delta(x):
        return x * 9/5

    fig=plt.figure(figsize=(5, 4))
    fig.suptitle('June 2023', size=13)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.scatter(streaks['streak_length'], streaks['temp_delta'], alpha=0.4, c='blue')
    ax1.plot(hourly_avg_delta['hour'], hourly_avg_delta['avg_delta'], 'r--', label='Average maximum temperature delta')
    ax1.set_xlabel('Continuous AC off time [hours]')
    ax1.grid(axis="y")
    ax1.set_ylabel('Maximum temperature delta [°C]')
    # ax1.set_title('Maximum temperature delta vs length of AC off time')
    ax1.legend()

    ax11 = ax1.twinx()
    ax11.set_ylabel('Maximum temperature delta [°F]', color='black')
    ax11.set_ylim(celsius_to_fahrenheit_delta(ax1.get_ylim()[0]), celsius_to_fahrenheit_delta(ax1.get_ylim()[1]))
    # Ensure the tick marks correspond correctly between the two scales
    ax11.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))

    # ax2 = fig.add_subplot(3,1,2)
    # ax2.scatter(streaks['streak_length'], streaks['max_temperature'], alpha=0.4, c='blue')
    # ax2.plot(hourly_avg['hour'], hourly_avg['avg_max_temperature'], 'r--', label='Average maximum temperature')
    # ax2.set_xlabel('Continuous AC off time [hours]')
    # ax2.grid(axis="y")
    # ax2.set_ylabel('Maximum temperature [°C]')
    # ax2.set_title('Maximum temperature vs length of AC off time')
    # ax2.legend()
    #
    # ax22 = ax2.twinx()
    # ax22.set_ylabel('Maximum temperature [°F]', color='black')
    # ax22.set_ylim(celsius_to_fahrenheit(ax2.get_ylim()[0]), celsius_to_fahrenheit(ax2.get_ylim()[1]))
    # ax22.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))

    # ax3 = fig.add_subplot(1, 1, 1)
    # ax3.scatter(streaks['streak_length'], streaks['avg_temperature'], alpha=0.4, c='blue')
    # ax3.plot(hourly_avg_temp['hour'], hourly_avg_temp['avg_temperature'], 'r--', label='Mean average temperature')
    # ax3.set_xlabel('Continuous AC off time [hours]')
    # ax3.grid(axis="y")
    # ax3.set_ylabel('Average internal temperature [°C]')
    # # ax3.set_title('Average temperature vs length of AC off time')
    # ax3.legend()
    #
    # ax33 = ax3.twinx()
    # ax33.set_ylabel('Average temperature [°F]', color='black')
    # ax33.set_ylim(celsius_to_fahrenheit(ax3.get_ylim()[0]), celsius_to_fahrenheit(ax3.get_ylim()[1]))
    # ax33.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))

    ax1.text(-0.1, 1, 'a)', transform=ax1.transAxes, fontsize=12, verticalalignment='top', weight='bold')
    # ax2.text(0, 1.1, 'b)', transform=ax2.transAxes, fontsize=12, verticalalignment='top', weight='bold')
    # ax3.text(-0.1, 1, 'd)', transform=ax3.transAxes, fontsize=12, verticalalignment='top', weight='bold')

    fig.tight_layout()
    plt.savefig('max temp delta june 23.png')
    plt.show()

    # plt.bar(number_of_streaks['streak_length'], number_of_streaks['count'], 0.2, color='blue')
    # plt.title('Number of occurrences of time periods of AC being off between 2 PM and 6 PM')
    # plt.xlabel('Hours of AC being off')
    # plt.ylabel('Number of occurrences')
    # plt.grid(axis='y')
    # plt.show()


def plot_temp_reach_histogram():
    """
    Creates a histogram of the percentage of streaks that reached a temperature of 30 degrees or more per streak duration.

    Parameters:
    - streaks (pd.DataFrame): DataFrame containing columns 'streak_length' and 'max_temperature'.
    """
    # Step 1: Filter streaks where max_temperature >= 30 degrees
    streaks = pd.read_csv('Streak-maxtemp.csv')
    print('Sample size: ', streaks['customer_id'].value_counts())

    streaks['streak_start_time'] = pd.to_datetime(streaks['streak_start_time'])

    # streaks = streaks[(streaks['streak_start_time'].dt.to_period('M').astype(str).str.startswith('2023-06')
             # | df['start_time'].dt.to_period('M').astype(str).str.startswith('2023-08')
             # )]
    streaks_june = streaks[streaks['streak_start_time'].dt.to_period('M').astype(str).str.startswith('2023-06')]
    streaks_july = streaks[streaks['streak_start_time'].dt.to_period('M').astype(str).str.startswith('2023-07')]

    streaks_june['streak_length_hour'] = streaks['streak_length'].round()
    streaks_july['streak_length_hour'] = streaks['streak_length'].round()

    # JUNE
    streaks_over_30_june = streaks_june[streaks_june['avg_temperature'] >= 30]

    streaks_over_30_june['streak_length_hour'] = streaks_over_30_june['streak_length'].round()

    total_streaks_per_hour_june = streaks_june['streak_length_hour'].value_counts().sort_index()
    streaks_over_30_per_hour_june = streaks_over_30_june['streak_length_hour'].value_counts().sort_index()

    percentage_over_30 = (streaks_over_30_per_hour_june / total_streaks_per_hour_june) * 100
    percentage_over_30 = percentage_over_30.reindex(total_streaks_per_hour_june.index, fill_value=0)

    # JULY
    streaks_over_30_july = streaks_july[streaks_july['avg_temperature'] >= 30]

    streaks_over_30_july['streak_length_hour'] = streaks_over_30_july['streak_length'].round()

    total_streaks_per_hour_july = streaks_july['streak_length_hour'].value_counts().sort_index()
    streaks_over_30_per_hour_july = streaks_over_30_july['streak_length_hour'].value_counts().sort_index()

    percentage_over_30_july = (streaks_over_30_per_hour_july/ total_streaks_per_hour_july) * 100
    percentage_over_30_july = percentage_over_30_july.reindex(total_streaks_per_hour_july.index, fill_value=0)

    bar_width = 0.35
    offset = bar_width / 2

    # Step 5: Plot the histogram
    plt.figure(figsize=(9, 4))
    plt.bar(percentage_over_30.index - offset, percentage_over_30.values, color='skyblue', edgecolor='black', width=0.25, label='June')
    plt.bar(percentage_over_30_july.index + offset, percentage_over_30_july.values, color='red', edgecolor='black', width=0.25, label='July')
    plt.suptitle('June and July 2023, 8:00-21:00')
    plt.xlabel('AC off duration [hours]')
    plt.ylabel('Instances with avg. temp. ≥ 30°C/86°F [%]')
    plt.title('AC off periods resulting in average internal temperature over 30°C/86°F by duration')
    plt.legend()

    # Adding percentage annotations above each bar
    for i in range(len(percentage_over_30)):
        plt.text(percentage_over_30.index[i], percentage_over_30.values[i] + 1,
                 f'{percentage_over_30.values[i]:.1f}%', ha='right', fontsize=9)

    for i in range(len(percentage_over_30_july)):
        plt.text(percentage_over_30_july.index[i], percentage_over_30_july.values[i] + 1,
                 f'{percentage_over_30_july.values[i]:.1f}%', ha='left', fontsize=9)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


plot_max_temp_vs_streak_time(results_df)

# plot_temp_reach_histogram()

# plot_area_hours(results_df, rate_counts)

# plot_avg_temp_increase(results_df)
#


# plot_results(results_df)

# plot_results_ratesplit(results_df)

# plot_temp_vs_time(results_df)

# plot_rate_vs_time(results_df)

# plot_rates_vs_time(results_df, rate_counts)
