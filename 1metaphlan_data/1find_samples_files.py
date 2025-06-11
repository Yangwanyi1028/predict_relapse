#!/usr/bin/env python3

import os
import glob

# 無 tailfix 的 ID
no_tailfix = [
    "ACC0147","ACC0163","ACC0048","ACC0102","ACC0034","ACC0138","ACC0096","ACC0049",
    "MiRES012","ACC0080","ACC0032","ACC0108","ACC0097","MiRES005","ACC0265","ACC0110",
    "ACC0192","ACC0013","ACC0114","ACC0136","MiRES011","ACC0169","MiRES003","ACC0135",
    "ACC0066","MiRES009","MiREL020","MIRES007","ACC0009","ACC0031","MiRES004"
]

# 有 tailfix 的 ID
tailfix = [
    "ACC0009_V03","ACC0013_V04","ACC0031_V03","ACC0032_V02","ACC0034_V02","ACC0048_V02",
    "ACC0049_V03","ACC0066_V02","ACC0072_D00","ACC0080_D00","ACC0085_V02","ACC0092_V02",
    "ACC0096_V03","ACC0097_V02","ACC0102_D00","ACC0107_D00","ACC0108_V04","ACC0110_D00",
    "ACC0114_V02","ACC0118_V02","ACC0135_V02","ACC0136_V02","ACC0138_D00","ACC0147_V02",
    "ACC0157_D00","ACC0163_V02","ACC0164_D00","ACC0167_D00","ACC0169_V02","ACC0191_D00",
    "ACC0192_D00","ACC0265_V03","MiREL020_D00","MiRES001_D00","MiRES002_D00","MiRES003_D00",
    "MiRES004_V02","MiRES005_D00","MIRES007_D00","MiRES008_D00","MiRES009_D00","MiRES011_D00",
    "MiRES012_D00","MiRES013_D00"
]
# 找出無 tailfix ID 在有 tailfix ID 中的對應
matched = {}
for nt in no_tailfix:
    matches = [t for t in tailfix if t.startswith(nt + "_")]
    if matches:
        matched[nt] = matches
    else:
        matched[nt] = []

# 輸出有對應的ID
for k, v in matched.items():
    if v:
        print(f"{k}: {v}")

# 輸出結果：獲取所有有對應的 tailfix 樣本ID
sample_ids = []
for v in matched.values():
    if v:
        sample_ids.extend(v)

# sample_ids 現在是所有有 tailfix 的樣本ID列表

# 要搜索的目录
search_dirs = ["1metaphlan_data/3_metaphlan4", "1metaphlan_data/ALL-metaphlan4", "1metaphlan_data/CD_Cohort2_174_118"]

# 结果文件路径列表
result_files = []

# 为每个样本ID查找对应的文件
for sample_id in sample_ids:
    found = False
    for directory in search_dirs:
        # 查找以样本ID开头的文件
        pattern = f"{directory}/{sample_id}*metaphlan"
        matching_files = glob.glob(pattern)
        # # 文件格式为 ID_Sxx_metaphlan，如 CD022_S15_metaphlan
        # pattern = f"{search_dir}/{sample_id}_S*_metaphlan"
        # matching_files = glob.glob(pattern)
        if matching_files:
            # 如果找到多个文件，选择最新的一个（按字母排序，通常V后面的数字大的是最新的）
            matching_files.sort(reverse=True)
            result_files.append(matching_files[0])
            found = True
            break
    
    if not found:
        print(f"Sample ID {sample_id} not found")

# 将结果写入文件
with open("1metaphlan_data/longitude_sample_files.txt", "w") as f:
    for file_path in result_files:
        f.write(f"{file_path}\n")

print(f"Found {len(result_files)} files, saved to metaphalan_data/longitude_sample_files.txt")



# 读取样本ID列表
with open("1metaphlan_data/IBDR_sample_ids.txt", "r") as f:
    sample_ids = [line.strip() for line in f if line.strip()]

# 要搜索的目录
search_dirs = ["1metaphlan_data/3_metaphlan4", "1metaphlan_data/ALL-metaphlan4", "1metaphlan_data/CD_Cohort2_174_118"]

# 结果文件路径列表
result_files = []

# 为每个样本ID查找对应的文件
for sample_id in sample_ids:
    found = False
    for directory in search_dirs:
        # 查找以样本ID开头的文件
        pattern = f"{directory}/{sample_id}*metaphlan"
        matching_files = glob.glob(pattern)
        # # 文件格式为 ID_Sxx_metaphlan，如 CD022_S15_metaphlan
        # pattern = f"{search_dir}/{sample_id}_S*_metaphlan"
        # matching_files = glob.glob(pattern)
        if matching_files:
            # 如果找到多个文件，选择最新的一个（按字母排序，通常V后面的数字大的是最新的）
            matching_files.sort(reverse=True)
            result_files.append(matching_files[0])
            found = True
            break
    
    if not found:
        print(f"Sample ID {sample_id} not found")

# 将结果写入文件
with open("1metaphlan_data/IBDR_sample_files.txt", "w") as f:
    for file_path in result_files:
        f.write(f"{file_path}\n")

print(f"Found {len(result_files)} files, saved to metaphalan_data/IBDR_sample_files.txt")




# 读取样本ID列表
with open("1metaphlan_data/ENIGMA_SCD_ids.txt", "r") as f:
    sample_ids = [line.strip() for line in f if line.strip()]

# 要搜索的目录
search_dirs = ["1metaphlan_data/3_metaphlan4", "1metaphlan_data/ALL-metaphlan4", "1metaphlan_data/CD_Cohort2_174_118"]

# 结果文件路径列表
result_files = []

# 为每个样本ID查找对应的文件
for sample_id in sample_ids:
    found = False
    for directory in search_dirs:
        # 查找以样本ID开头的文件
        pattern = f"{directory}/{sample_id}_*metaphlan"
        matching_files = glob.glob(pattern)
        
        if matching_files:
            # 如果找到多个文件，选择最新的一个（按字母排序，通常V后面的数字大的是最新的）
            matching_files.sort(reverse=True)
            result_files.append(matching_files[0])
            found = True
            break
    
    if not found:
        print(f"Sample ID {sample_id} not found")

# 将结果写入文件
with open("1metaphlan_data/ENIGMA_sample_files.txt", "w") as f:
    for file_path in result_files:
        f.write(f"{file_path}\n")

print(f"Found {len(result_files)} files, saved to metaphalan_data/ENIGMA_sample_files.txt")
