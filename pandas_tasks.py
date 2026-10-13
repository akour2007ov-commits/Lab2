"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd
from babel.util import missing

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    df = pd.read_csv(data.csv_path)
    row_count = int(len(df))
    missing_by_column = df.isna().sum().to_dict()
    adults_over_30_count = int((df['Age'] > 30).sum())
    mean_age_by_pclass = df.groupby('Pclass')['Age'].mean().to_dict()
    survival_rate_by_pclass = df.groupby('Pclass')['Survived'].mean().to_dict()
    highest_fares = df['Fare'].nlargest(5).tolist()
    return TitanicSummary(
        row_count=row_count,
        missing_by_column=missing_by_column,
        adults_over_30_count=adults_over_30_count,
        mean_age_by_pclass=mean_age_by_pclass,
        survival_rate_by_pclass=survival_rate_by_pclass,
        highest_fares=highest_fares
    )
input_data = TitanicInput(csv_path='titanic.csv')
output_data = analyze_titanic(input_data)
print(output_data)