# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -LanguageId
#     notebook_metadata_filter: -kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
# ---

# %% [markdown]
# Notebook 3: Conventional Machine Learning
# =========================================

# %% [markdown]
# # Import Section

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
import multiprocessing
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.random import sample_without_replacement
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler

import utils

# %%
# %matplotlib inline
plt.style.use("dark_background");
plt.set_cmap('Dark2');
sns.set_palette('Dark2');

# %%
df = utils.df_from_pickle('../data/df.p')

# %% [markdown]
# # Data Preprocessing for Machine Learning

# %%
df_ml = df.select_dtypes(include=np.number) \
          .drop(columns=['accident_id', 'accident_id_y']) \
          .dropna(axis=1, how='any')

# %%
df_ml = pd.get_dummies(data=df_ml, columns=['daylight', 'built-up_area', 'intersection_category', 'weather', 'collision_category', 
                                         'road_admin_category', 'traffic_regime', 'reserved_lane', 'plane_layout', 'surface_condition',
                                          'infrastructure', 'location', 'is_weekend', 'role'])

# %%
features = df_ml.drop(columns='severity')
feature_columns = features.columns
target = df_ml['severity']
random_under_sampler = RandomUnderSampler()
features, target = random_under_sampler.fit_resample(X=features, y=target)

max_sample_size = 1_000
sample_size = len(target)
if sample_size > max_sample_size:
    sample_idx = sample_without_replacement(n_population=sample_size, 
                                            n_samples=max_sample_size, 
                                            random_state=0)
    features = features.iloc[sample_idx]
    target = target.iloc[sample_idx]

# %%
scaler = StandardScaler()
features = scaler.fit_transform(features)

# %%
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, )

# %% [markdown]
# # XGBoost

# %%
X_train = pd.DataFrame(data=X_train)
X_train.columns = feature_columns

# %%
xgb_clf = XGBClassifier(n_jobs=multiprocessing.cpu_count()//2)

param_grid = {'max_depth': [2,4,6],
              'n_estimators': [100,200],
              'learning_rate': [0.05,0.1]}

grid = GridSearchCV(estimator=xgb_clf,
                    param_grid=param_grid,
                    cv=4,
                    n_jobs=2,
                    verbose=1)

grid.fit(X_train, y_train)

# %%
grid.best_params_

# %%
best_xgb = grid.best_estimator_
y_pred = best_xgb.predict(X_test)
print(classification_report(y_true=y_test, y_pred=y_pred))

# %%
best_xgb.get_booster()

# %%
from xgboost import plot_importance
p = plot_importance(best_xgb, max_num_features=15, height=0.8, grid='off')
p.grid(False)
