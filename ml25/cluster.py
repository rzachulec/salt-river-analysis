import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("aggregate_data_f_cost.csv")

agg_df = df.groupby("CustomerID").agg({
    "mean_internal_temp": "mean",
    "std_internal_temp": "mean",
    "mean_ac_use": "mean",
    "std_ac_use": "mean", 
    "avg_daytime_temp": "mean",
    "avg_night_temp": "mean",
    "temp_range": "mean",
    "fraction_above_27": "mean",
    "fraction_below_24": "mean",
    "AC_usage_correlation": "mean",
    "HH_Age": "first",
    "HH_Income_Desc": "first",
    "RATE": "first",
    "HH_NumberOfPersons": "first",
    "LVBL_SQ_FT": "first",
    "dayAC": "mean",
    "cost": "mean",
}).reset_index()

agg_df["AC_correlation"] = agg_df.apply(
    lambda row: df[df["CustomerID"] == row["CustomerID"]][["dayAC", "cost"]].corr(method="pearson").iloc[0, 1],
    axis=1
)

features = [
    "mean_internal_temp", "std_internal_temp", "mean_ac_use", 
    "std_ac_use", "avg_daytime_temp", "avg_night_temp",
    "temp_range", "fraction_above_27", "fraction_below_24", "AC_usage_correlation",
    "HH_Age", "HH_NumberOfPersons", "LVBL_SQ_FT", "RATE", "HH_Income_Desc", "dayAC", "cost", "AC_correlation"
]

features = [
    "mean_internal_temp", "mean_ac_use", 
    "RATE",
    "dayAC", "cost", "AC_correlation"
]

X = agg_df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=0, init='random')
agg_df["Cluster"] = kmeans.fit_predict(X_scaled)

clusters = agg_df.groupby("Cluster")[features].mean()
clusters["Cluster"] = [0, 1, 2]
print(clusters)
clusters.to_csv("clusters.csv")

# sns.pairplot(clusters, vars=["mean_internal_temp", "mean_ac_use", "avg_daytime_temp", "HH_Income_Desc"], hue="Cluster")

sns.pairplot(agg_df, vars=["AC_correlation", "RATE", "cost", "mean_internal_temp"], hue="Cluster", corner=True)
# sns.pairplot(agg_df, vars=["mean_internal_temp", "RATE", "HH_Income_Desc"], hue="Cluster", corner=True)
plt.show()