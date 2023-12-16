# -*- coding: utf-8 -*-
# @Time : 2023/11/26 22:33
# @Author : Shengxi Liang
# @File : random_data_generator.py
# @Software: PyCharm
import random

def generate_unique_data(num_points, existing_data, file_path):
    new_data = []
    with open(file_path, "w") as file:
        while len(new_data) < num_points:
            point = [round(random.uniform(-100, 100), 2), round(random.uniform(-100, 100), 2)]
            if point not in existing_data and point not in new_data:
                new_data.append(point)
                file.write("\t".join(map(lambda x: f"{x:.2f}", point)) + "\n")

# 已存在的数据点
existing_data = [
    [1.65, 4.28], [-3.45, 3.42], [4.84, -1.15], [-5.37, -3.36], [0.97, 2.92],
    [-3.57, 1.53], [0.45, -3.30], [-3.49, -1.72], [2.67, 1.59], [-3.16, 3.19]
]

# 指定文件路径
file_path = 'test_random.txt'  # 随机产生的数字存储

# 生成100个新的数据点并保存到文件
generate_unique_data(200, existing_data, file_path)
