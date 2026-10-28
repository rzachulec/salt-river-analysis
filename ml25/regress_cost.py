import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../aggregate_data_f_cost_weekdays.csv', )

def plotMeanInternalTemps():
    for rate, group in df.groupby('RATE'):
        num_customers = group['CustomerID'].nunique()
        print(rate)
        plt.figure()
        scatter = plt.scatter(
            group['dayAC'], group['cost'],
            c=group['mean_internal_temp'],
            cmap='coolwarm', alpha=0.5
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label('mean_internal_temp')
        plt.title(f'dayAC vs cost for RATE={rate} (hue: mean_internal_temp)\nNumber of Customers: {num_customers}')
        plt.show()
    
def plotMeanTempDifference():
    for rate, group in df.groupby('RATE'):
        num_customers = group['CustomerID'].nunique()
        group = group.copy()
        group['temp_difference'] = group['mean_internal_temp'] - group['mean_external_temp']
        print(rate)
        plt.figure()
        scatter = plt.scatter(
            group['dayAC'], group['cost'],
            c=group['temp_difference'],
            cmap='coolwarm', alpha=0.5
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label('temp_difference')
        plt.title(f'dayAC vs cost for RATE={rate} (hue: temp_difference)\nNumber of Customers: {num_customers}')
        plt.show()
        # plt.savefig(f'./img/dayAC_vs_cost_{rate}.png', bbox_inches='tight')
        # plt.close()
        
def plotMeanCostVsDifference():
    for rate, group in df.groupby('RATE'):
        num_customers = group['CustomerID'].nunique()
        group = group.copy()
        group['temp_difference'] = group['mean_internal_temp'] - group['mean_external_temp']
        print(rate)
        plt.figure()
        scatter = plt.scatter(
            group['cost'], group['temp_difference'],
            c=group['dayAC'],
            cmap='viridis', alpha=0.5
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label('dayAC')
        plt.title(f'cost vs temp differnce for RATE={rate} (hue: dayAC)\nNumber of Customers: {num_customers}')
        plt.show()
        # plt.savefig(f'./img/cost_vs_temp_difference_{rate}.png', bbox_inches='tight')
        # plt.close()
        
plotMeanTempDifference()
plotMeanCostVsDifference()
    
# sns.scatterplot(data=df, x='dayAC', y='cost', hue='RATE', palette='Set1')
# plt.title('dayAC vs cost grouped by RATE')
# plt.show()