


# 1 Download, Preprocess of Xarray.

不管你怎么排版，怎么乱搞，只有目的是生产以下文件，都必须以1为开头

1 编号所有文件的目的只有一个：

- 得到一个或一组xarray文件，满足以下要求：
  - 经纬度相同1km，以年为分辨率，
  - 变量名xxx：(int32类型)， 撂荒地开始的年
  - 变量名xxx：(int32类型)， 撂荒地的周期
  - 变量名xxx：(Bool类型)， 包含2022年是否为撂荒地
  - 变量名xxx：(int类型)， 土地利用类型


- 得到另一个或一组xarray文件，满足以下要求：
  - 经纬度相同1km，
  - 只需要包含xx年，xx年，xx年即可
  - 变量名xxx：(int32类型)， 风速xx
  - 变量名xxx：(int32类型)， 风速xx
  - 变量名xxx：(int32类型)， 风速xx



检验结果的代码：

ds_liaohuang=xr.open_dataset('xxx.nc')
ds_feature=xr.open_dataset('xxx.nc')

Option:

能够得到一个

ds_all= any function or logic, 所有的特征和撂荒


# 2. CSV 训练文件的生成和训练

## 2.1 process_csv_for_aligning
process_csv_for_aligning 是生成对齐文件的代码，这个文件的唯一目的是：

获得一个叫做 aligned_coordiantes.csv 文件，for example:

| Column | Description |
|--------|-------------|
| sol_id | Unique identifier for each solar panel location |
| GID_0 | Country code (e.g. CAN for Canada) |
| panels | Number of solar panels (1.0 = single panel) |
| p_area | Panel area in square kilometers |
| l_area | Land area in square kilometers |
| water | Binary flag indicating presence of water (0 = no, 1 = yes) |
| urban | Binary flag indicating urban area (0 = no, 1 = yes) |
| power | Power output in megawatts |
| longitude | Aligned longitude coordinate |
| latitude | Aligned latitude coordinate |
| Year | Year of the data |

(注意，这个文件包含所有年份的光伏站点， 包括**2017和2018**年！！！！！！)





## 2.2 process_csv_for_embedding 是生成训练文件的代码,


这个文件只有一个目的，生成目标
保存为2.training_embedding.csv 文件，包含

| Column | Description |
|--------|-------------|
| sol_id | Unique identifier for each solar panel location |
| longitude | Aligned longitude coordinate |
| latitude | Aligned latitude coordinate |
| power | Power output in megawatts |
| Wind_speed_100m | Wind speed at 100 meters above ground level |
| Temperature | Temperature in Celsius |
| Precipitation | Precipitation in millimeters |
| Land_cover | Land cover type |



这个代码的逻辑是：
根据ds_all文件，aligned_coordiantes.csv文件， 生成training_data.csv文件

首先，根据aligned_coordiantes.csv文件，的坐标和年份，提取ds_all满足条件的像素，拼接到新的dataframe中，

dataframe 保存为2.training_embedding.csv




## 2.3 process_csv_for_prediction 是生成预测文件的代码

   
这个代码的逻辑是：

根据ds_all文件，
给定年份参数，

得到一个dataframe，包含所有的普通像素

| Column | Description |
|--------|-------------|
| sol_id | Unique identifier for each solar panel location |
| longitude | Aligned longitude coordinate |
| latitude | Aligned latitude coordinate |
| Wind_speed_100m | Wind speed at 100 meters above ground level |
| Temperature | Temperature in Celsius |
| Precipitation | Precipitation in millimeters |
| Land_cover | Land cover type |

保存为 2.data_prediction.csv

## 3.1 预测

df_predict=pd.read_csv('2.data_prediction.csv')

df_training=pd.read_csv('2.training_embedding.csv')




1. 首先标准化所有特征

2. 然后，给定特征列表，如：

['Wind_speed_100m', 'Temperature', 'Precipitation', 'Land_cover']


根据距离相似度函数，预测df_predict的power, 转换为ds, 绘图后保存为


3.prediction_{year}.nc







