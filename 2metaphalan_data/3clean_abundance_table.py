#!/usr/bin/env python3

import pandas as pd
import re

# 读取原始丰度表格
abundance_table = pd.read_csv('2metaphalan_data/longitude_abundance_table.txt', sep='\t', skiprows=1)

# 获取列名
columns = abundance_table.columns.tolist()

# 创建新的列名映射
column_mapping = {}
for col in columns[1:]:  # 跳过第一列 'clade_name'
    # 提取样本ID（例如，从'ACC0147_V02_S165_metaphlan'中提取'ACC0147'）
    match = re.match(r'([A-Za-z0-9]+)_', col)
    if match:
        sample_id = match.group(1)
        column_mapping[col] = sample_id

# 重命名列
new_columns = {'clade_name': 'clade_name'}
new_columns.update(column_mapping)
abundance_table = abundance_table.rename(columns=new_columns)

# 保存到新文件
abundance_table.to_csv('2metaphalan_data/longitude_abundance_clean.txt', sep='\t', index=False)

print("Clean abundance table created: metaphalan_data/longitude_abundance_clean.txt")


# 读取原始丰度表格
abundance_table = pd.read_csv('2metaphalan_data/ENIGMA_abundance_table.txt', sep='\t', skiprows=1)

# 获取列名
columns = abundance_table.columns.tolist()

# 创建新的列名映射
column_mapping = {}
for col in columns[1:]:  # 跳过第一列 'clade_name'
    # 提取样本ID（例如，从'ACC0147_V02_S165_metaphlan'中提取'ACC0147'）
    match = re.match(r'([A-Za-z0-9]+)_', col)
    if match:
        sample_id = match.group(1)
        column_mapping[col] = sample_id

# 重命名列
new_columns = {'clade_name': 'clade_name'}
new_columns.update(column_mapping)
abundance_table = abundance_table.rename(columns=new_columns)

# 保存到新文件
abundance_table.to_csv('2metaphalan_data/ENIGMA_abundance_clean.txt', sep='\t', index=False)

print("Clean abundance table created: metaphalan_data/ENIGMA_abundance_clean.txt")


# 读取原始丰度表格
abundance_table = pd.read_csv('2metaphalan_data/IBDR_abundance_table.txt', sep='\t', skiprows=1)

# 获取列名
columns = abundance_table.columns.tolist()

# 创建新的列名映射
column_mapping = {}
for col in columns[1:]:  # 跳过第一列 'clade_name'
    # 提取样本ID（例如，从'ACC0147_V02_S165_metaphlan'中提取'ACC0147'）
    match = re.match(r'([A-Za-z0-9]+)_', col)
    if match:
        sample_id = match.group(1)
        column_mapping[col] = sample_id

# 重命名列
new_columns = {'clade_name': 'clade_name'}
new_columns.update(column_mapping)
abundance_table = abundance_table.rename(columns=new_columns)

# 保存到新文件
abundance_table.to_csv('2metaphalan_data/IBDR_abundance_clean.txt', sep='\t', index=False)

print("Clean abundance table created: metaphalan_data/IBDR_abundance_clean.txt")