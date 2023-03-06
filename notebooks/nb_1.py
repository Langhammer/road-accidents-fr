# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: roafr_env
#     language: python
#     name: python3
# ---

# %% [markdown]
# <img src="../images/headers/nb_1.svg"  width="1080" height="220">
#
# The purpose of this notebook is to
# 1. Import the data
# 2. Clean the data
# 3. Export the data

# %% [markdown]
# # Table of Contents
# - [Import Dataset](#import-dataset)
# - [Clean Accident Dataset](#clean-accident-dataset)
# - [Clean Vehicles Dataset](#clean-vehicles-dataset)
# - [Clean Persons Dataset](#clean-persons-dataset)
# - [Merge Datasets](#merge-datasets)
# - [Feature Engineering](#feature-engineering)
# - [Export Data](#export-data)

# %%
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pywaffle import Waffle
import missingno as msno

from roaf import data, visualization

# %%
# %matplotlib inline
plt.style.use("dark_background")
plt.set_cmap("Dark2")
sns.set_palette("Dark2")

# %% [markdown]
# # Parameters

# %% tags=["parameters"]
PLOT_DIR = "../images/"
N_PLOT_SAMPLE = None
PLOT_FILE_FORMATS = None

# %% [markdown]
# # Import Dataset

# %%
dfd = data.read_csv_of_year(range(2019, 2022))

# %% [markdown]
# # Clean Accident Dataset
# The accident dataset will be merged from the datasets *characteristics* and *locations,* as both
# have the same key *(accident_id).*

# %%
accidents = pd.merge(
    left=dfd["characteristics"],
    right=dfd["locations"],
    how="outer",
    left_on="Num_Acc",
    right_on="Num_Acc",
    validate="one_to_one",
    indicator=True,
)

# Check
accidents["_merge"].value_counts()

# %% [markdown]
# ## Translation of French Variable Names

# %%
accidents.rename(
    columns={
        "Num_Acc": "accident_id",
        "an": "year",  # Characteristics Dataset
        "mois": "month",
        "jour": "day",
        "hrmn": "hhmm",
        "lum": "daylight",
        "agg": "built_up_area",
        "int": "intersection_category",
        "atm": "weather",
        "col": "collision_category",
        "com": "municipality",
        "adr": "address",
        "gps": "gps_origin",
        "lat": "latitude",
        "long": "longitude",
        "dep": "department",
        "catr": "road_admin_category",  # Locations Dataset
        "voie": "road_number",
        "v1": "road_numerical_id",
        "v2": "road_alphanumerical_id",
        "circ": "traffic_regime",
        "nbv": "n_lanes",
        "pr": "landmark",
        "pr1": "dist_to_landmark",
        "vosp": "reserved_lane",
        "prof": "slope",
        "plan": "plane_layout",
        "lartpc": "median_strip_width",
        "larrout": "affected_road_width",
        "surf": "surface_condition",
        "infra": "infrastructure",
        "situ": "location",
        "env1": "near_school",
        "vma": "max_speed",
    },
    inplace=True,
)

# %% [markdown]
# ## Display of a sample

# %%
accidents.sample(20).T

# %% [markdown]
# From displaying a small sample of accidents it is visible that the address is not standardized
# (as mentioned in the documentation). Given that GPS data is available, the address will probably
# not be of any value.
#
# It is also visible that there are parentheses in *landmark* and *dist_to_landmark.*

# %% [markdown]
# ## Analysis of missing values
# In order to get an understanding of the missing values, the missingno package will be used.
# This package provides a couple of useful plots that visualize the number of missing values as
# well as the relationship between missing values of different variables.
#
# In most cases, the missing values have to be identified and converted, as they are encoded with -1.

# %%
# Convert Variables with -1 and empty strings for missing values
accidents.replace(to_replace=[-1, ""], value=None, inplace=True)

# %%
msno.matrix(accidents)
visualization.savefig(
    basename="msno_matrix_accidents", filepath=PLOT_DIR, formats=PLOT_FILE_FORMATS
)

# %%
msno.heatmap(accidents, figsize=(6, 6), fontsize=10)
plt.title("Correlations of Missing Values")
visualization.savefig(
    basename="msno_heat_accidents", filepath=PLOT_DIR, formats=PLOT_FILE_FORMATS
)


# %%
def na_stats(df, years=None):
    """Compute the number and ratio of missing values for the specified years.
    If no years are specified, the stats for all years will be computed.
    """
    if (years is None) & ("year" in df.columns):
        years = range(df["year"].min(), df["year"].max() + 1)

    if isinstance(years, int):
        years = [years]

    if years is not None:
        df = df[df["year"].isin(years)]

    na_stats_df = pd.DataFrame(df.isna().sum(), columns=["na_counts"])
    inverse_n_rows = 1 / len(df)
    na_stats_df["na_ratio"] = na_stats_df["na_counts"].apply(
        lambda x: x * inverse_n_rows
    )
    na_stats_df.rename_axis("variable", inplace=True)
    na_stats_df = na_stats_df[na_stats_df["na_counts"] != 0]
    na_stats_df.sort_values(by="na_counts", ascending=False, inplace=True)
    return na_stats_df


def plot_na_ratio(df=None, na_stats_df=None, years=None):
    """Plot the ratio of missing values for the specified years."""
    if (na_stats_df is not None) & (df is not None):
        raise ValueError("Only one argument of df and na_stats can be used.")
    if (na_stats_df is None) & (df is not None):
        na_stats_df = na_stats(df, years=years)

    sns.barplot(data=na_stats_df, x="na_ratio", y=na_stats_df.index)
    plt.title("Ratio of Missing Values")
    plt.ylabel("Variable Name")
    plt.xlabel("Ratio of Missing Values")
    return na_stats_df


# %%
plot_na_ratio(df=accidents, years=range(2019, 2022))

# %% [markdown]
# The plots show that the affected_road_with variable seems to come into existence only after one
# year, while a small number of values exists before.
#
# The road_numerical_id has a lot of values missing also in the first third of the dataset.
# It is possible that the definition of this variable changed at that point and that the values
# are not completely comparable before and after the change.
#
# The alphanumerical id of the road is missing in nearly all cases. It will probably not be
# beneficial to keep this variable.
#
# The correlation plot shows that similar variables are more likely to share missing values.
# The variables *plane layout* and *slope* are correlated the most, and both describe the
# topology of the accident. I assume that missing values mean that the accident happened on a flat
# and straight road.

# %%
# The columns median_strip_width, affected_road_width and road_numerical_id are missing a
# lot of values, so they will be dropped.
accidents.drop(
    columns=["median_strip_width", "affected_road_width", "road_numerical_id"],
    inplace=True,
)

# %% [markdown]
# ## Time and Date-Related Variables

# %%
# Fix inconsistent year format
accidents["year"].replace(
    {
        5: 2005,
        6: 2006,
        7: 2007,
        8: 2008,
        9: 2009,
        10: 2010,
        11: 2011,
        12: 2012,
        13: 2013,
        14: 2014,
        15: 2015,
        16: 2016,
        17: 2017,
        18: 2018,
    },
    inplace=True,
)

# Fix inconsistent time format
accidents["hhmm"] = accidents["hhmm"].apply(lambda s: str(s).replace(":", ""))

accidents["hour"] = accidents["hhmm"].apply(lambda hhmm: hhmm[:-2])
accidents["hour"] = accidents["hour"].replace("", np.nan).fillna(method="bfill")
accidents["minute"] = accidents["hhmm"].apply(lambda hhmm: hhmm[-2:])

accidents["date"] = pd.to_datetime(
    {
        "year": accidents["year"],
        "month": accidents["month"],
        "day": accidents["day"],
        "hour": accidents["hour"],
        "minute": accidents["minute"],
    }
)

# New variable: weekday, integer from 0 to 6 representing the weekdays from monday to sunday.
accidents["day_of_week"] = accidents["date"].apply(lambda x: x.day_of_week)

# New binary variable: is_weekend, 0 for monday to friday and 1 for saturday and sunday
accidents["is_weekend"] = (accidents["day_of_week"] > 4).astype("int")

# The hhmm variable will be used in its integer representation for plotting in nb 2
accidents["hhmm"] = accidents["hhmm"].astype("int")


# %% [markdown]
# ## Department Variable
# The department is numerically encoded, but they don't exactly match the *Code officiel géographique*
# (COG).


# %%
def department_converter(dep):
    """
    Takes in a department code as int and returns a string
    e.g. 750 will be '75' for Paris and 201 will be '2B'
    """
    if dep == 201:
        return "2A"
    elif dep == 202:
        return "2B"
    elif dep > 970:
        return str(dep)
    else:
        return str(dep).strip("0")


accidents.loc[(np.less(accidents["year"], 2019)), "department"] = accidents[
    (np.less(accidents["year"], 2019))
]["department"].apply(department_converter)

# %% [markdown]
# ## GPS-Data

# %%
# Replace commas with periods in GPS Data
accidents["latitude"] = (
    accidents["latitude"].apply(lambda x: x.replace(",", ".")).astype("float")
)
accidents["longitude"] = (
    accidents["longitude"].apply(lambda x: x.replace(",", ".")).astype("float")
)

# %%
# Now I can check if there are some values equal to zero, which can be an indicator for missing
# values
for this_var in ["latitude", "longitude"]:
    print(this_var, (accidents[this_var] == 0.0).sum())

# %%
# Convert to Web Mercator Projection
accidents = data.df_geotransform(df=accidents)

# %%
# Longitude
visualization.plot_continuous_variable_overview(
    accidents, "longitude", filter_percentile=0.1
)

# Saving (exporting svg can take a lot time)
visualization.savefig(
    basename="overview_longitude", filepath=PLOT_DIR, formats=PLOT_FILE_FORMATS
)

# Latitude
visualization.plot_continuous_variable_overview(
    accidents, "latitude", filter_percentile=0.1
)

# Saving (exporting svg can take a lot time)
visualization.savefig(
    basename="overview_latitude", filepath=PLOT_DIR, formats=PLOT_FILE_FORMATS
)

# %% [markdown]
# Plotting the distribution of latitude and longitude we can see that there is one very dense
# regions and some outlier regions. This shows that the dataset does not only contain the data from
# mainland France, but also from the overseas.

# %% [markdown]
# ## Other variables

# %%
PLOT_FILE_FORMATS = ["png"]

# %%
weather_dict = {
    1: "Normal",
    2: "Light rain",
    3: "Heavy rain",
    4: "Snow/Hail",
    5: "Fog/smoke",
    6: "Strong wind/storm",
    7: "Dazzling",
    8: "Overcast",
    9: "Other",
}
sns.countplot(y=accidents["weather"].replace(weather_dict))
plt.title("Weather Conditions")
visualization.savefig(
    basename="hist_weather", filepath=PLOT_DIR, formats=PLOT_FILE_FORMATS
)

# %%
accidents["weather"] = accidents["weather"].fillna(accidents["weather"].mode()[0])
accidents["weather"].replace({-1, 0}, inplace=True)
accidents["weather"] = accidents["weather"].astype("int")

# %%
accidents["collision_category"] = accidents["collision_category"].fillna(
    accidents["collision_category"].mode()[0]
)

# %%
accidents["built_up_area"].replace({1: 0, 2: 1}, inplace=True)

# %% [markdown]
# ## Max  Speed

# %%
plt.figure(figsize=(6, 6))
sns.countplot(data=accidents, y="max_speed")

# %% [markdown]
# Plotting the maximum speed shows that there are some irrational values.

# %%
accidents["max_speed"].replace(
    {300: 30, 500: 50, 700: 70, 800: 80, 900: 90}, inplace=True
)

# %% [markdown]
# # Clean Vehicles Dataset
# Another dataset is available which contains more additional information, but will not be used here.
# The reason is, that it contains only information about accidents with hurt or killed people, which
# means that (1) there are a lot of missing values and (2) the presence of this data could easily
# lead to leaking in machine learning.

# %%
dfd["vehicles"].rename(
    columns={
        "Num_Acc": "accident_id",
        "id_vehicule": "unique_vehicle_id",
        "num_veh": "vehicle_id",
        "senc": "direction",
        "catv": "vehicle_category",
        "obs": "immobile_obstacle",
        "obsm": "mobile_obstacle",
        "choc": "impact_point",
        "manv": "last_operation",
        "motor": "motor_type",
        "occutc": "n_occupants",
    },
    inplace=True,
)

# %%
dfd["vehicles"].sample(20).T

# %% [markdown]
# ## Analysis of Missing Values

# %%
# Missing values are assigned the vale -1 in the original dataset
dfd["vehicles"].replace({-1: None}, inplace=True)
na_stats(dfd["vehicles"])

# %%
msno.matrix(dfd["vehicles"])
visualization.savefig(
    basename="msno_matrix_vehicles", filepath=PLOT_DIR, formats=PLOT_FILE_FORMATS
)

# %%
# Every single vehicle Id ends with '01', so we can get rid of it
print(dfd["vehicles"]["vehicle_id"].apply(lambda x: x[-2:]).value_counts())
dfd["vehicles"]["vehicle_id"] = dfd["vehicles"]["vehicle_id"].apply(lambda x: x[:-2])

# %%
# Let's see how many values are not alphabetically encoded 8after removing the numeric part)
vehicle_id_az = dfd["vehicles"]["vehicle_id"].apply(
    lambda x: re.search(pattern="[^A-Z]", string=x)
)
for i_match, this_match in enumerate(vehicle_id_az):
    try:
        vehicle_id_az[i_match] = this_match.group(0)
    except AttributeError:
        pass
vehicle_id_az.value_counts()

# %%
dfd["vehicles"]["vehicle_id"] = dfd["vehicles"]["vehicle_id"].replace(
    dict.fromkeys(["[", "]", "\\"], np.nan)
)
dfd["vehicles"].isna().sum()

# %% [markdown]
# As a possible improvement the missing values fo vehicle_id could be inferred from the other
# vehicle ids. # This is not done here, because the variable is possibly not necessary because of
# the existence of unique_vehicle_id.

# %% [markdown]
# # Clean Persons Dataset

# %%
dfd["persons"].rename(
    columns={
        "Num_Acc": "accident_id",
        "id_vehicule": "unique_vehicle_id",
        "num_veh": "vehicle_id",
        "place": "seat",
        "catu": "role",
        "grav": "severity",
        "sexe": "sex",
        "an_nais": "year_of_birth",
        "trajet": "objective",
        "secu1": "safety_1",
        "secu2": "safety_2",
        "secu3": "safety_3",
        "locp": "pedestrian_location",
        "actp": "pedestrian_action",
        "etatp": "pedestrian_company",
    },
    inplace=True,
)

# %%
msno.matrix(dfd["persons"])
visualization.savefig(basename="msno_matrix_persons", filepath=PLOT_DIR)

# %%
# Missing values are assigned the vale -1 in the original dataset
dfd["persons"].replace({-1: np.nan}, inplace=True)
na_stats(dfd["persons"])

# %%
dfd["persons"].dtypes

# %% [markdown]
# ## Severity
# The severity is the most important variable, as it will be the target variable for
# machine learning.
#
# In the original database, the severity is encoded as
# *(OLD ORDER)*
# 1 - Unharmed
# 2 - Killed
# 3 - Hospitalized
# 4 - Minor injury
#
# The "hospitalized" indicator is no longer labelled by the public statistics authority as of 2019
# [(source)](https://www.data.gouv.fr/en/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2021/#description). # pylint: disable=C0301
# Therefore, the 'injured' categories will be merged.
#
# For the machine learning classifier provided by the XGBoost package, the classes have to start
# from 0.
#
# The severity categories will also be reordered to make the order more logical:
# *(NEW ORDER)*
# 0 - Unharmed
# 1 - Injured
# 2 - Killed

# %%
dfd["persons"]["severity"].replace({1: 0, 3: 1, 4: 1}, inplace=True)
dfd["persons"]["severity"].fillna(0, inplace=True)
dfd["persons"]["severity"] = dfd["persons"]["severity"].astype("int")

# %%
persons_sample = dfd["persons"].sample(frac=0.001, random_state=0)
persons_features = (
    persons_sample.select_dtypes(include=np.number)
    .drop(columns=["severity", "accident_id"])
    .columns
)
sns.pairplot(
    data=persons_sample,
    vars=persons_features,
    hue="severity",
    diag_kind="hist",
    palette="Dark2",
)

# %%
fig = plt.figure(
    FigureClass=Waffle,
    rows=5,
    columns=10,
    values=dfd["persons"]["role"].value_counts().sort_index(),
    labels=["driver", "passenger", "pedestrian"],
    legend={"loc": "lower left", "bbox_to_anchor": (0.16, -0.15), "ncol": 3},
    title={"label": "Role of Involved Persons"},
)

# %%
# The variable pedestrian_company has a high number of missing values.
# This is mostly due to the fact that the number of pedestrians is relatively low
dfd["persons"]["pedestrian_company"][dfd["persons"]["role"] == 3].value_counts(
    dropna=False
)

# %% [markdown]
# Most of the pedestrians were alone and the missing values are much lower if we only take the
# pedestrians into account. I can therefore assume that the missing values are mostly '1',
# if the person is a passenger. The rest will be set to zero.

# %%
dfd["persons"].loc[dfd["persons"]["role"] == 3, "pedestrian_company"] = dfd["persons"][
    "pedestrian_company"
][dfd["persons"]["role"] == 3].fillna(1)
dfd["persons"].loc[dfd["persons"]["role"] != 3, "pedestrian_company"] = dfd["persons"][
    "pedestrian_company"
][dfd["persons"]["role"] != 3].fillna(0)

# %% [markdown]
# # Merge Datasets

# %% [markdown]
# First, the 'vehicles' and 'persons' dataframes have to be merged on 'unique_vehicle_id'.
# Then, this new dataframe will be merged with the 'accidents' dataframe on 'accident_id'
#
# Note:
# In the original dataset, pedestrians are associated with a vehicle involved in the accident,
# probably the one that hit them.
# I will keep this association for now, but there surely is an alternative way to handle this.

# %%
persons_vehicles_merged = pd.merge(
    left=dfd["vehicles"],
    right=dfd["persons"],
    on="unique_vehicle_id",
    suffixes=(None, "_y"),
    validate="one_to_many",
)

# %%
df_complete = pd.merge(
    left=accidents,
    right=persons_vehicles_merged,
    on="accident_id",
    suffixes=(None, "_y"),
    validate="one_to_many",
)

# %% [markdown]
# # Feature Engineering
# Some features can only be created now that all data is in one dataframe.

# %% [markdown]
# ## Age

# %%
df_complete.loc[:, "age"] = df_complete["year"] - df_complete["year_of_birth"]

# %% [markdown]
# # Export Data

# %%
df_complete.to_parquet("../data/processed/df_by_user.parquet")
