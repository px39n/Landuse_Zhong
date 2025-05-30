


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



## 4.1 Emission_Reduction_potential.ipynb
净碳减排效益评估（Net Carbon Benefit）**


对每块土地，定义其部署光伏的净减排效益，考虑以下三个维度：

$$
{\text { Net } \text { Benefit }_i=\text { Emission Reduction }_i-\text { Opportunity }_{\text {Loss }}^i}
$$

- Emission Reduction $n_i$ ：该地部署光伏带来的年均碳减排量（基于电网碳强度和发电潜力计算）
- Opportunity Loss $_i$ ：该地若用于其他用途（如碳汇，农业，保育）的平均或最大减排潜力损失
- 这个差值代表真实的＂减排净值＂，更贴合气候政策优化目标

**1 Section， 你需要计算一个表格的column，叫做Emission Reduction**
- Emission Reduction $n_i$ ：该地部署光伏带来的年均碳减排量（基于电网碳强度和发电潜力计算）
**Input** is 3.data_prediction.csv'


**Output** is a column of the table

**2 Section， 你需要计算一个表格的column，叫做Opportunity Loss**
- Opportunity Loss $_i$ ：该地若用于其他用途（如碳汇，农业，保育）的平均或最大减排潜力损失
**Output** is a column of the table

**3 Section， 你需要计算一个表格的column，叫做Net Carbon Benefit**
- Net Carbon Benefit $_i$ ：该地部署光伏的净减排效益

使用公式
$$
{\text { Net } \text { Benefit }_i=\text { Emission Reduction }_i-\text { Opportunity }_{\text {Loss }}^i}
$$
的变体，你自己思考

然后最后必须只能得到一个东西csv，叫做**4.data_prediction_net_benefit.csv**





## **5．环境代理成本评估（Proxy Environmental Cost）**
构建一种新的＂环境代价指标＂，量化部署光伏的不可见损失，例如：
- 生物多样性丧失（使用物种多样性指数或生态热区重叠度）
- 土地利用冲突（与农业，森林或自然保护区的重叠系数）
- 景观完整性破坏，社会接受度等

将这些维度转换为统一单位（如美元／吨 $\mathrm{CO}_2$ 或加权分值），形成一种代理成本（proxy cost）或综合损失函数，用于与减排效益一同纳入多目标优化。


这是你最后一个任务， 需要生成一个csv文件，叫做**6.data_prediction_proxy_cost.csv**




## 6.1 EDA of Data.ipynb， 这个是探索性数据分析

分成三个section (严格按照markdown 一级标题来分区)

Section 1： 空间分布
画出来你的2016年和2017年的撂荒地空间分布

Section 2： 特征分布
这里的目的是，画出来你的特征的分布， 包括风速，温度，降水量，土地利用类型


（这一部分，可以使用我编写的库， pip install FeatureInsight
然后使用如下代码，
```python
from FeatureInsight import struct_Investigation,univar_dis,bivar_dis
Structure summary=struct_Investigation(df)
summary.print() 
summary.sort('Unique Count')

univar_dis(df,df.columns)


```


## 6.2 Figure1_Policy_Suitability_Map.ipynb， 这个是生成图1的代码

分成三个section (严格按照markdown 一级标题来分区)

Section 1. 光伏潜力预测图
用气泡，或者化成连续的样子。

## 6.3 Figure2_Emission_Reduction_Potential.ipynb， 这个是生成图2的代码

分成三个section (严格按照markdown 一级标题来分区)

Section 1-3. 净碳效益评估图
用气泡，或者化成连续的样子。分别绘制


Emission Reduction

Opportunity Loss

Net Carbon Benefit


