# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: roafr
#     language: python
#     name: python3
# ---

# %% [markdown]
# Notebook 2: Visualization
# =========================
# In this notebook, the data will be visualized and analyzed

# %% [markdown]
# # Import Modules and Data

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import utils

# %matplotlib inline
plt.style.use("dark_background");

# %%
df = utils.df_from_pickle('../data/df.p')

# %% [markdown]
# # Time Series

# %%
plot_data.isna()

# %%
plot_df = pd.DataFrame({'is_weekend': df['is_weekend'], 'time':df['date'].apply(lambda x: int(x.time()))})
plot_df['time']

# %%
weights

# %%
day_time_ticks = (0,300,600,900,1200,1500,1800,2100,2400)
day_time_tick_labels = ('0:00', '03:00','06:00','09:00','12:00','15:00',
                   '18:00','21:00','24:00')
fig= plt.figure();
plot_df = df[['is_weekend', 'hhmm']].astype('int')
weights = plot_df['is_weekend'].apply(lambda x: 0.5 if x==1 else 0.2)
sns.histplot(data=plot_df,
            x='hhmm',
            hue='is_weekend',
            weights=weights,
            stat='frequency',
            bins=24,
            binrange=(0,2400),
            common_norm=False,
            palette='Dark2');
plt.xticks(ticks=day_time_ticks, 
           labels=day_time_tick_labels);
plt.xlabel('Time of Day')
plt.xlim((0,2400))
plt.legend(['Weekends', 'Weekdays']);
plt.title('Distribution of Accidents by Daytime')

# %% [markdown]
# # Age and Sex

# %%
ax = sns.violinplot(data=df, 
                x='role', 
                y='age', 
                hue='sex', 
                split=True, 
                palette='Dark2', 
                inner=None);
plt.title('Age of People Involved in Road Accidents')
plt.ylim((0,df['age'].max()))
ax.legend(handles=ax.legend_.legendHandles, labels=['Male', 'Female']);
ax.set_xticklabels(['driver', 'passenger', 'pedestrian']);

