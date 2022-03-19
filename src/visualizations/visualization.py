#!/usr/bin/env python
# coding: utf-8

# https://towardsdatascience.com/predicting-house-prices-with-linear-regression-machine-learning-from-scratch-part-ii-47a0238aeac1 .

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

cwd = os.getcwd()
raw_data_dirpath = '../../data/raw'
visualization_image_path = 'visualizationImages/'

# Load Housing Data into dataframe
df = pd.read_csv(os.path.join(cwd, raw_data_dirpath, 'train.csv'), index_col='Id')
df.shape

# Look at the df columns
print(df.columns)

# Adding target_label for independent feature column name
target_label = 'SalePrice'

# Look at the quick description of dataframe
print(df.describe())

# Look at the df head
print(df.head())

# Look at the df tail
print(df.tail())

# Look at the datatype of each column
print(df.dtypes)

# Find Categorical and numerical feature columns
categorical_features = [col for col in df.columns if df[col].dtype == 'object']
numerical_features = list(set(df.columns) - set(categorical_features))
print(len(categorical_features))
print(len(numerical_features))

# Data description of categorical features
print(df[categorical_features].describe())

# Update cwd to visualizationImage dir to store plots
os.chdir(visualization_image_path)

# Value distribution of target variable i.e. SalePrice
# Most of the density lies between 100k and 250k, but there appears to be a lot of outliers on the pricier side.
fig = plt.figure(figsize=(18, 6))
hist_plot = sns.histplot(df['SalePrice'], kde=True)
fig = hist_plot.get_figure()
fig.savefig('SalePriceDistribution.png')


# Visulize of how SalePrice varies w.r.t. GrLivArea
# You might’ve expected that larger living area should mean a higher price. This chart shows you’re generally correct. But what are those 2–3 “cheap” houses offering huge living area?
fig = plt.figure(figsize=(12, 6))
regplot = sns.regplot(x=df['GrLivArea'], y=df['SalePrice'])
fig = regplot.get_figure()
fig.savefig('SalePriceVsGrLivArea_RegPlot.png')

# One column you might not think about exploring is the “TotalBsmtSF” — Total square feet of the basement area, but let’s do it anyway:
# Intriguing, isn’t it? The basement area seems like it might have a lot of predictive power for our model.
fig = plt.figure(figsize=(12, 6))
regplot = sns.regplot(x=df['TotalBsmtSF'], y=df['SalePrice'])
fig = regplot.get_figure()
fig.savefig('SalePriceVsTotalBsmtSF_RegPlot.png')

# Visualize how SalePrice data distributed w.r.t OverallQual
# Everything seems fine for this one, except that when you look to the right things start getting much more nuanced. Will that “confuse” our model?
fig = plt.figure(figsize=(12, 6))
boxplot = sns.boxplot(x=df['OverallQual'], y=df['SalePrice'])
fig = boxplot.get_figure()
fig.savefig('SalePriceVsOverallQual_BoxPlot.png')

# Let’s have a more general view on the top 10 correlated features with dependent/independent features:
fig = plt.figure(figsize=(6, 6))
df_corr = df.corr().abs()
df_corr_sorted = df_corr.unstack().sort_values(ascending=False).drop_duplicates()
df_corr_sorted = pd.DataFrame(df_corr_sorted).reset_index()
# Adding dummy columns name for pivoting
df_corr_sorted.columns = ['A', 'B', 'C']
heatmap = sns.heatmap(df_corr_sorted[:10].pivot(values='C', index='A', columns='B'), annot=True, fmt='.2f')
fig = heatmap.get_figure()
fig.savefig('top10CorrelatedFeatures_HeatMap.png')

# Let’s have a more general view on the top 10 correlated features with independent feature i.e. SalePrice:
fig = plt.figure(figsize=(10, 10))
df_corr = df.corr().abs()
df_corr_sorted = df_corr.unstack().sort_values(ascending=False).drop_duplicates()
df_corr_sorted = pd.DataFrame(df_corr_sorted).reset_index()
# Adding dummy columns name for pivoting
df_corr_sorted.columns = ['A', 'B', 'C']
top_corr_features_with_target = df_corr_sorted[df_corr_sorted['B'] == 'SalePrice'][:10]['A']
heatmap = sns.heatmap(pd.DataFrame(df_corr[top_corr_features_with_target]).loc[
                top_corr_features_with_target, top_corr_features_with_target], annot=True, fmt='.2f')
fig = heatmap.get_figure()
fig.savefig('top10CorrelatedFeaturesWithSalePrice_HeatMap.png')

print('Visualization finished successfully!')
