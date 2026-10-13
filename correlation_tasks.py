"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

from nbformat.v4 import output_from_msg

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    df = pd.read_csv(data.csv_path, sep='\t', na_values='NA')
    features = ['FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height']

    male_df = df[df['Gender'] == 'Male']
    female_df = df[df['Gender'] == 'Female']

    men_count = int(len(male_df))
    women_count = int(len(female_df))

    men_mri_correlation = male_df[features].corrwith(male_df['MRI_Count']).to_dict()
    women_mri_correlation = female_df[features].corrwith(female_df['MRI_Count']).to_dict()

    strongest_mri_feature = None
    max_abs_corr = -1.0

    for feat, corr_val in men_mri_correlation.items():
        if pd.notna(corr_val) and abs(corr_val) > max_abs_corr:
            max_abs_corr = abs(corr_val)
            strongest_mri_feature = feat

    for feat, corr_val in women_mri_correlation.items():
        if pd.notna(corr_val) and abs(corr_val) > max_abs_corr:
            max_abs_corr = abs(corr_val)
            strongest_mri_feature = feat
    return BrainCorrelationSummary(
        men_count=men_count,
        women_count=women_count,
        women_mri_correlation=women_mri_correlation,
        men_mri_correlation=men_mri_correlation,
        strongest_mri_feature=strongest_mri_feature
    )


def plot_brain_visualizations(df):
    features = ['FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height', 'MRI_Count']

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    male_matrix = df[df['Gender'] == 'Male'][features].corr()
    sns.heatmap(male_matrix, annot=True, cmap='BrBG', ax=axes[0])
    axes[0].set_title("Корреляции у мужчин (Male)")

    female_matrix = df[df['Gender'] == 'Female'][features].corr()
    sns.heatmap(female_matrix, annot=True, cmap='BrBG', ax=axes[1])
    axes[1].set_title("Корреляции у женщин (Female)")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x='MRI_Count', y='FSIQ', hue='Gender')
    plt.title("Зависимость FSIQ от MRI_Count")
    plt.show()

full_dataframe = pd.read_csv('brainsize.txt', sep='\t', na_values='NA')
input_data = BrainDataInput(csv_path='brainsize.txt')
output_data = analyze_brain_correlations(input_data)
print(output_data)
plot_brain_visualizations(full_dataframe)
