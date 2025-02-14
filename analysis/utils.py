import numpy as np
from datetime import datetime
import ast

def remove_outliers_irq(data: list) -> list:
    q1 = np.percentile(data, 25)  # 25th percentile (Q1)
    q3 = np.percentile(data, 75)  # 75th percentile (Q3)
    iqr = q3 - q1  # Interquartile range
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [x for x in data if lower_bound <= x <= upper_bound]


def remove_outliers_zscore(data: list) -> list:
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)  # Use ddof=1 for sample standard deviation
    z_scores = [(x - mean) / std_dev for x in data]
    threshold = 3
    return [x for x, z in zip(data, z_scores) if abs(z) <= threshold]


def remove_outliers_confidence_interval(data: list) -> list:
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)  # Use ddof=1 for sample standard deviation
    n = len(data)  # Number of data points
    std_error = std_dev / np.sqrt(n)
    confidence_level = 0.95
    t_value = t.ppf((1 + confidence_level) / 2, df=n - 1)  # Degrees of freedom = n - 1
    margin_of_error = t_value * std_error
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    return [x for x in data if lower_bound <= x <= upper_bound]


def convert_to_datetime(timestamp_str):
    timestamp_str = timestamp_str.strip()
    try:
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.strptime(timestamp_str, "%m/%d/%Y, %I:%M:%S %p")


def convert_fmp_to_datetime(timestamp_str):
    timestamp_str = timestamp_str.strip() 
    try:
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        return datetime.strptime(timestamp_str, "%m/%d/%Y, %I:%M:%S %p")



def parse_list_column(column):
    return column.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
