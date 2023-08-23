import os
import random

from PIL import Image
import numpy as np
import shutil
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import pdb
def load_images(img_info_df):
    img_info = pd.read_csv(img_info_df)
    file_paths = img_info["Path"].tolist()
    file_names = [os.path.basename(i) for i in img_info["Path"].tolist()]
    images = [np.array(Image.open(i)) for i in img_info["Path"].tolist()]
    return images, file_names, file_paths
# 将图片数据转换为特征向量（使用像素值平均）
def image_to_feature_vector(image):
    return np.mean(image, axis=(0, 1)).flatten()



# HE 数据

# 结局变量信息
out_info = pd.read_excel(r"E:\zhouyehan\single_label_Proj\化免114.xlsx")
out_info["TRG"] = out_info["TRG"] - 1
out_info["MPR"] = out_info["MPR"] - 1
out_info["PCR"] = out_info["PCR"] - 1
out_info["编号"] = out_info["编号"].astype("int64")

HE_dir = ['HE-1', 'HE-10', 'HE-11', 'HE-2', 'HE-3', 'HE-4', 'HE-5', 'HE-6', 'HE-7', 'HE-8', 'HE-9']
# HE 数据
image_list = []
# i = HE_dir[0]
for i in HE_dir:
    # patch_info = pd.read_csv(f'E:\zhouyehan\single_label_Proj\patch\{i}\patch_info.csv',index_col=1)
    patch_info = parse_patch_info_one_hzy(patch_info_path=f'E:\zhouyehan\single_label_Proj\patch_1024\{i}\patch_info.csv',parse_var ="info", cutoff =0.8,if_eval=True)
    patch_info = patch_info.loc[(patch_info['parse'] != "_background_") & (patch_info['parse'].notnull())]
    patch_info["parse"] = patch_info["parse"].astype("int64")
    patch_info["TRG"] = [find_x_hzy(out_info,i,"TRG") for i in patch_info["parse"].tolist()]
    patch_info["MPR"] = [find_x_hzy(out_info, i, "MPR") for i in patch_info["parse"].tolist()]
    patch_info["PCR"] = [find_x_hzy(out_info, i, "PCR") for i in patch_info["parse"].tolist()]
    image_list.append(patch_info)


HE_df_all = pd.concat(image_list)
HE_df_all["Path"] = HE_df_all["filename"].apply(lambda x:os.path.join("E:\zhouyehan\single_label_Proj\patch_1024",x))
HE_df_all["img_name"] = HE_df_all["filename"].str.split("\\",expand = True)[1].tolist()
HE_df_all.to_csv(r'E:\zhouyehan\single_label_Proj\HE_df_all_1024.csv',header=True,index=False )

img_info = pd.read_csv(r'E:\zhouyehan\single_label_Proj\HE_df_all_1024.csv')

images, file_names, file_paths = load_images(r'E:\zhouyehan\single_label_Proj\HE_df_all_1024.csv')
image_vectors = [image_to_feature_vector(img) for img in images]
# 无监督聚类（使用K均值聚类）
n_clusters = 6  # 聚类数量
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(image_vectors)
# 可视化聚类结果（使用降维）
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(image_vectors)
plt.figure(figsize=(8, 6))
for i in range(n_clusters):
    cluster_points = reduced_features[cluster_labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}')
plt.title('Image Clustering Visualization')
plt.xlabel('')
plt.ylabel('')
plt.legend()
# plt.show()

# 创建文件夹以存放聚类结果
for i in range(n_clusters):
    cluster_folder = f'Cluster {i + 1}'
    if not os.path.exists(cluster_folder):
        os.makedirs(cluster_folder)

# 将图片移动到对应的聚类文件夹
for i, filename in enumerate(file_names):
    # source_path = os.path.join(image_folder, filename)
    source_path = file_paths[i]
    target_folder = f'Cluster {cluster_labels[i] + 1}'
    target_path = os.path.join(target_folder, filename)
    shutil.copy(source_path, target_path)


# 剔除 cluster 2 HE 数据
cluster2_directory = r'C:\Users\Administrator\bin\huangzongyao\torchkit\Cluster 2'
cluster2_files = os.listdir(cluster2_directory)

img_info_filter = img_info[~img_info["img_name"].isin(cluster2_files)]
img_info_filter.to_csv(r'E:\zhouyehan\single_label_Proj\HE_df_all_1024_filter.csv',header=True,index=False )


# HE test
out_info = pd.read_csv(r"E:\zhouyehan\single_label_Proj\TEST_patch\test_info.csv")
test_image_list = []
for i in out_info["Patient_id"].tolist():
    # patch_info = pd.read_csv(f'E:\zhouyehan\single_label_Proj\patch\{i}\patch_info.csv',index_col=1)
    patch_info = parse_patch_info_one_hzy(patch_info_path=f'E:\zhouyehan\single_label_Proj\TEST_patch\{i}\patch_info.csv',parse_var ="info", cutoff =0.8,if_eval=True)
    patch_info.insert(0,"Patient_id",i)
    patch_info = patch_info.loc[(patch_info['parse'] != "_background_") & (patch_info['parse'].notnull())]
    patch_info = pd.merge(patch_info,out_info,how = "left",on = "Patient_id")
    test_image_list.append(patch_info)

test_all_df = pd.concat(test_image_list)
test_all_df.insert(0,"Path",[os.path.join(r"E:\zhouyehan\single_label_Proj\TEST_patch",i) for i in test_all_df["filename"].tolist()])
test_all_df["img_name"] = test_all_df["filename"].str.split("\\",expand = True)[1].tolist()
test_all_df.to_csv(r'E:\zhouyehan\single_label_Proj\TEST_patch\test_df_all.csv')


## 聚类
images, file_names, file_paths = load_images(r'E:\zhouyehan\single_label_Proj\TEST_patch\test_df_all.csv')
image_vectors = [image_to_feature_vector(img) for img in images]
# 无监督聚类（使用K均值聚类）
n_clusters = 6  # 聚类数量
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(image_vectors)
# 可视化聚类结果（使用降维）
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(image_vectors)
plt.figure(figsize=(8, 6))
for i in range(n_clusters):
    cluster_points = reduced_features[cluster_labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}')
plt.title('Image Clustering Visualization')
plt.xlabel('')
plt.ylabel('')
plt.legend()
# plt.show()

# 创建文件夹以存放聚类结果
for i in range(n_clusters):
    cluster_folder = f'Cluster {i + 1}'
    if not os.path.exists(cluster_folder):
        os.makedirs(cluster_folder)

# 将图片移动到对应的聚类文件夹
for i, filename in enumerate(file_names):
    # source_path = os.path.join(image_folder, filename)
    source_path = file_paths[i]
    target_folder = f'Cluster {cluster_labels[i] + 1}'
    target_path = os.path.join(target_folder, filename)
    shutil.copy(source_path, target_path)





# CD数据
CD_dir = ['CD3-1', 'CD3-10', 'CD3-11', 'CD3-2', 'CD3-3', 'CD3-4', 'CD3-5', 'CD3-6', 'CD3-7', 'CD3-8', 'CD3-9']
# 结局变量信息
out_info = pd.read_excel(r"E:\zhouyehan\single_label_Proj\化免114.xlsx")
out_info["TRG"] = out_info["TRG"] - 1
out_info["MPR"] = out_info["MPR"] - 1
out_info["PCR"] = out_info["PCR"] - 1
out_info["编号"] = out_info["编号"].astype("int64")

image_list = []
for i in CD_dir:
    # patch_info = pd.read_csv(f'E:\zhouyehan\single_label_Proj\patch\{i}\patch_info.csv',index_col=1)
    patch_info = parse_patch_info_one_hzy(patch_info_path=f'E:\zhouyehan\single_label_Proj\patch_1024\{i}\patch_info.csv',parse_var ="info", cutoff =0.8,if_eval=True)
    patch_info = patch_info.loc[(patch_info['parse'] != "_background_") & (patch_info['parse'].notnull())]
    patch_info["parse"] = patch_info["parse"].astype("int64")
    patch_info["TRG"] = [find_x_hzy(out_info,i,"TRG") for i in patch_info["parse"].tolist()]
    patch_info["MPR"] = [find_x_hzy(out_info, i, "MPR") for i in patch_info["parse"].tolist()]
    patch_info["PCR"] = [find_x_hzy(out_info, i, "PCR") for i in patch_info["parse"].tolist()]
    image_list.append(patch_info)


CD_df_all = pd.concat(image_list)
CD_df_all["Path"] = CD_df_all["filename"].apply(lambda x:os.path.join("E:\zhouyehan\single_label_Proj\patch_1024",x))
CD_df_all["img_name"] = CD_df_all["filename"].str.split("\\",expand = True)[1].tolist()
CD_df_all.to_csv(r'E:\zhouyehan\single_label_Proj\CD_df_all_1024.csv',header=True,index=False )


img_info = pd.read_csv(r'E:\zhouyehan\single_label_Proj\CD_df_all_1024.csv')

images, file_names, file_paths = load_images(r'E:\zhouyehan\single_label_Proj\CD_df_all_1024.csv')
image_vectors = [image_to_feature_vector(img) for img in images]
# 无监督聚类（使用K均值聚类）
n_clusters = 7  # 聚类数量
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(image_vectors)
# 可视化聚类结果（使用降维）
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(image_vectors)
plt.figure(figsize=(8, 6))
for i in range(n_clusters):
    cluster_points = reduced_features[cluster_labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}')
plt.title('Image Clustering Visualization')
plt.xlabel('')
plt.ylabel('')
plt.legend()
# plt.show()

# 创建文件夹以存放聚类结果
for i in range(n_clusters):
    cluster_folder = f'Cluster {i + 1}'
    if not os.path.exists(cluster_folder):
        os.makedirs(cluster_folder)

# 将图片移动到对应的聚类文件夹
for i, filename in enumerate(file_names):
    # source_path = os.path.join(image_folder, filename)
    source_path = file_paths[i]
    target_folder = f'Cluster {cluster_labels[i] + 1}'
    target_path = os.path.join(target_folder, filename)
    shutil.copy(source_path, target_path)

# 剔除 cluster 7 HE 数据
cluster7_directory = r'C:\Users\Administrator\bin\huangzongyao\torchkit\Cluster 7'
cluster7_files = os.listdir(cluster7_directory)

img_info_filter = img_info[~img_info["img_name"].isin(cluster7_files)]

# 随机挑选10个人的数据作为test
img_info_filter["Patient_id"] = img_info_filter["parse"]
random.seed(1120)
img_info_filter_train_val = img_info_filter[~img_info_filter["Patient_id"].isin(random.sample(img_info_filter["Patient_id"].unique().tolist(),10))]
img_info_filter_test = img_info_filter[img_info_filter["Patient_id"].isin(random.sample(img_info_filter["Patient_id"].unique().tolist(),10))]
img_info_filter_train_val.to_csv(r'E:\zhouyehan\single_label_Proj\CD_df_all_1024_filter_train_val.csv',header=True,index=False )
img_info_filter_test.to_csv(r'E:\zhouyehan\single_label_Proj\CD_df_all_1024_filter_test.csv',header=True,index=False )






PD_dir = ['PD-L-1', 'PD-L-10', 'PD-L-11', 'PD-L-2', 'PD-L-3', 'PD-L-4', 'PD-L-5', 'PD-L-6', 'PD-L-7', 'PD-L1-8','PD-L-9']
# 结局变量信息
out_info = pd.read_excel(r"E:\zhouyehan\single_label_Proj\化免114.xlsx")
out_info["TRG"] = out_info["TRG"] - 1
out_info["MPR"] = out_info["MPR"] - 1
out_info["PCR"] = out_info["PCR"] - 1
out_info["编号"] = out_info["编号"].astype("int64")

image_list = []
for i in PD_dir:
    # patch_info = pd.read_csv(f'E:\zhouyehan\single_label_Proj\patch\{i}\patch_info.csv',index_col=1)
    patch_info = parse_patch_info_one_hzy(patch_info_path=f'E:\zhouyehan\single_label_Proj\patch_1024\{i}\patch_info.csv',parse_var ="info", cutoff =0.8,if_eval=True)
    patch_info = patch_info.loc[(patch_info['parse'] != "_background_") & (patch_info['parse'].notnull())]
    patch_info["parse"] = patch_info["parse"].astype("int64")
    patch_info["TRG"] = [find_x_hzy(out_info,i,"TRG") for i in patch_info["parse"].tolist()]
    patch_info["MPR"] = [find_x_hzy(out_info, i, "MPR") for i in patch_info["parse"].tolist()]
    patch_info["PCR"] = [find_x_hzy(out_info, i, "PCR") for i in patch_info["parse"].tolist()]
    image_list.append(patch_info)


PD_df_all = pd.concat(image_list)
PD_df_all["Path"] = PD_df_all["filename"].apply(lambda x:os.path.join("E:\zhouyehan\single_label_Proj\patch_1024",x))
PD_df_all["img_name"] = PD_df_all["filename"].str.split("\\",expand = True)[1].tolist()
PD_df_all.to_csv(r'E:\zhouyehan\single_label_Proj\PD_df_all_1024.csv',header=True,index=False )


img_info = pd.read_csv(r'E:\zhouyehan\single_label_Proj\PD_df_all_1024.csv')

images, file_names, file_paths = load_images(r'E:\zhouyehan\single_label_Proj\PD_df_all_1024.csv')
image_vectors = [image_to_feature_vector(img) for img in images]
# 无监督聚类（使用K均值聚类）
n_clusters = 8  # 聚类数量
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(image_vectors)
# 可视化聚类结果（使用降维）
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(image_vectors)
plt.figure(figsize=(8, 6))
for i in range(n_clusters):
    cluster_points = reduced_features[cluster_labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}')
plt.title('Image Clustering Visualization')
plt.xlabel('')
plt.ylabel('')
plt.legend()
# plt.show()

# 创建文件夹以存放聚类结果
for i in range(n_clusters):
    cluster_folder = f'Cluster {i + 1}'
    if not os.path.exists(cluster_folder):
        os.makedirs(cluster_folder)

# 将图片移动到对应的聚类文件夹
for i, filename in enumerate(file_names):
    # source_path = os.path.join(image_folder, filename)
    source_path = file_paths[i]
    target_folder = f'Cluster {cluster_labels[i] + 1}'
    target_path = os.path.join(target_folder, filename)
    shutil.copy(source_path, target_path)

# 剔除 cluster 4  数据
cluster4_directory = r'C:\Users\Administrator\bin\huangzongyao\torchkit\Cluster 4'
cluster4_files = os.listdir(cluster4_directory)

img_info_filter = img_info[~img_info["img_name"].isin(cluster4_files)]

# 随机挑选10个人的数据作为test
img_info_filter["Patient_id"] = img_info_filter["parse"]
random.seed(1120)
img_info_filter_train_val = img_info_filter[~img_info_filter["Patient_id"].isin(random.sample(img_info_filter["Patient_id"].unique().tolist(),10))]
img_info_filter_test = img_info_filter[img_info_filter["Patient_id"].isin(random.sample(img_info_filter["Patient_id"].unique().tolist(),10))]
img_info_filter_train_val.to_csv(r'E:\zhouyehan\single_label_Proj\PD_df_all_1024_filter_train_val.csv',header=True,index=False )
img_info_filter_test.to_csv(r'E:\zhouyehan\single_label_Proj\PD_df_all_1024_filter_test.csv',header=True,index=False )


