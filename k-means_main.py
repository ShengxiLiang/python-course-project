# @Time : 2023/11/26 22:33
# @Author : Shengxi Liang
# @File : K-means_main.py
# @Software: PyCharm


import random
import math
import sys
import copy
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog, messagebox
from tkinter import filedialog
# 本程序使用Python自带库进行k-means聚类算法的实现，确保与Python解释器的兼容性。
# 为了更好地可视化聚类结果，使用matplotlib库，这是唯一的非自带库依赖。

def load_data(file_name, delimiter='\t'):
    """
    从文件中加载二维数据点。
    :param file_name: 包含数据的文件名。
    :param delimiter: 数据分隔符，默认为制表符。
    :return: 数据点列表，若文件不存在或格式错误则返回None。
    """
    data = []
    try:
        with open(file_name, 'r') as file:
            for line in file:
                # 忽略空行，并按delimiter分割
                if line.strip():
                    row = [float(num) for num in line.strip().split(delimiter)]
                    data.append(row)
    except FileNotFoundError:
        messagebox.showerror("File Error", f"Error: The file '{file_name}' was not found.")
        return None
    except ValueError:
        messagebox.showerror("Data Error", f"Error: Data format is incorrect in '{file_name}'.")
        return None
    return data

def standardize_data(data):
    """
    数据预处理，将数据标准化为均值为0，标准差为1。
    :param data: 原始数据列表。
    :return: 标准化后的数据列表。
    """
    if not data:
        return []
    mean = [sum(x) / len(x) for x in zip(*data)]
    std = [math.sqrt(sum((x - m) ** 2 for x in col) / len(col)) for m, col in zip(mean, zip(*data))]
    return [[(x - m) / s if s else 0 for x, m, s in zip(row, mean, std)] for row in data]

def euclidean_distance(point1, point2):
    """
    计算两点之间的欧式距离。
    :param point1: 第一个点的坐标。
    :param point2: 第二个点的坐标。
    :return: 两点之间的距离。
    """
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

def initialize_centroids(data, k):
    """
    随机初始化k个质心。
    :param data: 数据点列表。
    :param k: 质心数量。
    :return: 随机选择的k个质心列表。
    """
    return random.sample(data, k)

def assign_points_to_clusters(data, centroids):
    """
    将每个点分配到最近的质心。
    :param data: 数据点列表。
    :param centroids: 质心列表。
    :return: 按质心分配的聚类列表。
    """
    clusters = [[] for _ in centroids]
    for point in data:
        closest_centroid_idx = min(range(len(centroids)),
                                   key=lambda idx: euclidean_distance(point, centroids[idx]))
        clusters[closest_centroid_idx].append(point)
    return clusters

def compute_new_centroids(clusters):
    """
    重新计算每个簇的质心。
    :param clusters: 按质心分配的聚类列表。
    :return: 新的质心列表。
    """
    return [list(map(lambda x: sum(x) / len(x), zip(*cluster))) for cluster in clusters if cluster]

def compute_SSE(clusters, centroids):
    """
    计算聚类的误差平方和（SSE）。
    :param clusters: 按质心分配的聚类列表。
    :param centroids: 质心列表。
    :return: 聚类的SSE值。
    """
    sse = 0
    for idx, cluster in enumerate(clusters):
        sse += sum(euclidean_distance(point, centroids[idx]) ** 2 for point in cluster)
    return sse

def k_means(data, k, max_iterations=100, convergence_threshold=0.001):
    """
    K-means主算法，返回聚类结果、质心和SSE。
    :param data: 数据点列表。
    :param k: 质心数量。
    :param max_iterations: 最大迭代次数。
    :param convergence_threshold: 收敛阈值。
    :return: 聚类结果、质心和SSE。
    """
    centroids = initialize_centroids(data, k)
    prev_centroids = []
    iteration_count = 0  # 追踪迭代次数

    for iteration in range(max_iterations):
        clusters = assign_points_to_clusters(data, centroids)
        new_centroids = compute_new_centroids(clusters)
        centroid_shift = sum(euclidean_distance(c1, c2) for c1, c2 in zip(new_centroids, centroids))

        if centroid_shift < convergence_threshold:
            break
        centroids = new_centroids
        iteration_count += 1

    sse = compute_SSE(clusters, centroids)
    return clusters, centroids, sse, iteration_count

def plot_clusters(clusters, centroids, sse, iteration_count):
    """
    使用matplotlib绘制聚类结果。
    :param clusters: 按质心分配的聚类列表。
    :param centroids: 质心列表。
    :param sse: SSE值。
    :param iteration_count: 迭代次数。
    """
    colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple']
    plt.figure(figsize=(10, 6))
    for idx, cluster in enumerate(clusters):
        points = zip(*cluster)
        plt.scatter(*points, c=colors[idx % len(colors)], marker='o', label=f"Cluster {idx + 1} (n={len(cluster)})")
    for idx, centroid in enumerate(centroids):
        plt.scatter(*centroid, c='black', marker='x')

    plt.title(f"K-Means Clustering (SSE: {sse:.2f}, Iterations: {iteration_count})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()
def show_clustering_results(clusters, centroids):
    """
    显示聚类结果的自定义对话框。
    :param clusters: 按质心分配的聚类列表。
    :param centroids: 质心列表。
    """
    # 创建一个新的顶层窗口
    top = tk.Toplevel()
    top.title("Clustering Results")

    # 创建一个文本框和滚动条
    text = tk.Text(top, wrap="none")
    scroll_y = tk.Scrollbar(top, orient="vertical", command=text.yview)
    scroll_x = tk.Scrollbar(top, orient="horizontal", command=text.xview)
    text.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

    # 填充文本框
    for idx, cluster in enumerate(clusters):
        text.insert(tk.END, f"Cluster {idx + 1} (Centroid: {centroids[idx]}):\n")
        for point in cluster:
            text.insert(tk.END, f"  {point}\n")
        text.insert(tk.END, "\n")

    # 布局
    scroll_y.pack(side="right", fill="y")
    scroll_x.pack(side="bottom", fill="x")
    text.pack(side="left", fill="both", expand=True)
    # 窗口关闭事件处理程序
    def on_closing():
        top.destroy()
        sys.exit()  # 退出程序

    top.protocol("WM_DELETE_WINDOW", on_closing)


def main():
    """
    主函数，执行数据加载、聚类和结果展示。
    """
    # 创建一个Tk窗口实例
    root = tk.Tk()
    # 将主窗口隐藏，因为我们只需要文件对话框
    root.withdraw()

    # 配置文件对话框的参数，用于选择数据文件
    file_options = {
        "defaultextension": ".txt",  # 默认文件扩展名为.txt
        "filetypes": [("Text files", "*.txt"), ("All files", "*.*")],  # 可选择的文件类型
        "initialdir": ".",  # 文件对话框打开时的初始目录，默认是当前工作目录
        "title": "Choose a data file"  # 文件对话框的标题
    }

    # 弹出文件选择对话框，让用户选择数据文件，并返回选中的文件路径
    file_name = filedialog.askopenfilename(**file_options)
    # 如果用户没有选择文件就关闭对话框，则退出程序
    if not file_name:
        print("No file selected. Exiting program.")
        return

    # 加载用户选中的数据文件
    data = load_data(file_name)
    # 如果数据加载失败（文件不存在或数据格式错误），则退出程序
    if data is None:
        return

    # 加载数据成功后，深度复制一份数据用于后续处理
    original_data = copy.deepcopy(data)

    # 通过一个简单的对话框请求用户输入聚类的数量（k值）
    k = simpledialog.askinteger("Input", "Enter the number of clusters (k):", parent=root)
    # 如果用户没有输入k值，提示并退出程序
    if k is None:
        print("No input provided. Exiting program.")
        return
    # 如果用户输入的k值不合理（小于等于0或大于数据点的数量），弹出错误提示并退出
    if k <= 0 or k > len(data):
        messagebox.showerror("Error",
                             "Invalid input. Please enter a positive integer less than or equal to the number of data points.")
        return

    # 对数据进行标准化处理
    standardized_data = standardize_data(data)

    # 使用标准化后的数据进行k-means聚类
    std_clusters, std_centroids, sse, iteration_count = k_means(standardized_data, k)
    # 使用原始数据进行k-means聚类，用于展示聚类结果
    orig_clusters, orig_centroids, _, _ = k_means(original_data, k)

    # 使用matplotlib库绘制标准化数据的聚类结果
    plot_clusters(std_clusters, std_centroids, sse, iteration_count)

    # 使用原始数据的聚类结果，展示每个簇的信息
    show_clustering_results(orig_clusters, orig_centroids)

    # 准备显示所有聚类信息的字符串，包括每个簇的中心点和属于该簇的数据点
    cluster_info = "K-Means Clustering Results:\n\n"
    for idx, cluster in enumerate(std_clusters):
        cluster_info += f"Cluster {idx + 1} (Centroid: {std_centroids[idx]}):\n"
        for point in cluster:
            cluster_info += f"\t{point}\n"
        cluster_info += "\n"

    # 进入Tkinter的主事件循环，等待用户交互事件
    root.mainloop()


# 如果这个脚本是作为主程序运行，而不是被导入到其他Python脚本中，则执行main函数
if __name__ == '__main__':
    main()