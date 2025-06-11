#!/bin/bash
# 运行整个MetaPhlAn数据处理流程

# 设置错误处理
set -e  # 如果任何命令失败，立即退出脚本
set -u  # 使用未定义的变量时报错

echo "开始运行MetaPhlAn数据处理流程..."

# 步骤1: 查找样本ID
echo "步骤1: 查找样本ID"
Rscript 1metaphlan_data/0find_sample_id.R
if [ $? -ne 0 ]; then
    echo "错误: 查找样本ID失败"
    exit 1
fi

# 步骤2: 查找样本文件
echo "步骤2: 查找样本文件"
python3 1metaphlan_data/1find_samples_files.py
if [ $? -ne 0 ]; then
    echo "错误: 查找样本文件失败"
    exit 1
fi

# 步骤3: 合并 abundance 表格
echo "步骤3: 合并 abundance 表格"
# 处理 longitude 数据
echo "  处理 longitude 数据..."
python3 1metaphlan_data/2merge_metaphlan_tables.py -l 1metaphlan_data/longitude_sample_files.txt -o 1metaphlan_data/longitude_abundance_table.txt
if [ $? -ne 0 ]; then
    echo "错误: 合并 longitude 数据失败"
    exit 1
fi

# 处理 IBDR 数据
echo "  处理 IBDR 数据..."
python3 1metaphlan_data/2merge_metaphlan_tables.py -l 1metaphlan_data/IBDR_sample_files.txt -o 1metaphlan_data/IBDR_abundance_table.txt
if [ $? -ne 0 ]; then
    echo "错误: 合并 IBDR 数据失败"
    exit 1
fi
# 处理 ENIGMA 数据
echo "  处理 ENIGMA 数据..."
python3 1metaphlan_data/2merge_metaphlan_tables.py -l 1metaphlan_data/ENIGMA_sample_files.txt -o 1metaphlan_data/ENIGMA_abundance_table.txt
if [ $? -ne 0 ]; then
    echo "错误: 合并 ENIGMA 数据失败"
    exit 1
fi

# 步骤4: 清理 abundance 表格
echo "步骤4: 清理 abundance 表格"
python3 1metaphlan_data/3clean_abundance_table.py
if [ $? -ne 0 ]; then
    echo "错误: 清理 abundance 表格失败"
    exit 1
fi

# 步骤5: 合并并校正丰度数据

echo "步骤5: 合并并校正丰度数据"
Rscript 1metaphlan_data/4combine_abundance.R
if [ $? -ne 0 ]; then
    echo "错误: 合并并校正丰度数据失败"
    exit 1
fi

echo "MetaPhlAn数据处理流程完成!"

# 显示结果文件
echo "生成的结果文件:"
ls -lh 1metaphlan_data/*abundance*.txt

exit 0