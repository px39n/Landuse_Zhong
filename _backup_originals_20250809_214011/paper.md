



Abstract: 

# Introduction 

To cap cumulative carbon emissions at approximately 335 PgC from 2023 onward and meet global climate targets, the large‑scale deployment of low‑carbon photovoltaic (PV) systems is indispensable. Projections indicate that by 2050, solar PV will become the world’s leading electricity source—supplying 25 % of global demand with some 8,519 GW of installed capacity (IRENA, 2019a). As the dominant clean‑energy market, the United States (US) plans to expand its solar share to roughly 30 % by mid‑century. However, the rapid rollout of utility‑scale PV in rural America threatens productive cropland and may impose adverse environmental impacts. This tension has spurred growing interest in natural climate solutions (NCS) and allocating PV on marginal lands, including abandoned cropland, open-pit mining sites, and urban rooftops. Despite numerous promising pathways, the real feasibility of large-scale PV deployment at the national scale remains largely unknown, thereby clouding policymakers' confidence in advancing energy transitions. To fill this gap, we quantify both the land-specific PV deployment potential from 2020-2050 and the decision-making trade-offs, including environmental suitability, carbon abatement ability, and economic feasibility.

Solar energy is essential for climate mitigation. Unlike nuclear, bioenergy, or geothermal energy, photovoltaic (PV) farm siting is highly sensitive to climatic conditions. Recent research shows strong compatibility between agricultural land and PV deployment, as farming activities already exist in regions with favorable solar radiation, temperature, and wind conditions. While US croplands expand over 0.4 million hectares annually, substantial abandonment also occurs in areas like the Mid-Atlantic Coast, aligning with the accelerating abandonment trends observed across the Global North. Similar to agrivoltaic co-location and perovskite photovoltaic deployment approaches, redeveloping abandoned cropland for PV installation represents a promising emissions reduction pathway that could provide unique spatial opportunities. However, large-scale PV deployment requires local energy infrastructure and socio-economic support, which vary significantly across regions. Previous research has inadequately captured these national-specific spatiotemporal heterogeneities. Recent advances in machine learning and high-dimensional statistical data embedding offer opportunities to extract multi-dimensional characteristics from historical PV installation data and generalize these insights to specific policy target regions.

Although Natural Climate Solutions (NCS) are widely recognized as the most cost-effective approach, forest ecosystems require decades for long-term carbon sequestration and have high land demand. Field experiments from the United States, Israel etc., have demonstrated that despite PV deployment leading to a significantly decrease in surface albedo and consequently increase local temperature, the atmospheric carbon mitigation efficiency is ~50× than afforestation. Furthermore, as the US levelized cost of PV electricity generation continues to decline to $0.03 by 2030, large-scale PV deployment will become increasingly economically viable. However, current assessments of solar deployment potential remain limited in scope, primarily focusing on electricity generation potential while neglecting the broader socio-economic and policy implications. The absence of comprehensive cross-sectoral comparative evaluations (e.g., agriculture, forestry) may significantly underestimate the policy potential of large-scale PV deployment and create uncertainty for policymakers in advancing energy transitions.

Here, we propose an extensible land-specific carbon abatement decision-making framework. First, we identify the spatiotemporal distribution of abandoned cropland and existing solar farm locations across US using time-series land-use imagery and OpenStreetMap (OSM) geographic data. Subsequently, we train an environmentally-similarity-based machine learning model that integrates multi-source spatiotemporal data including climatic, geographic-physical, and socio-economic factors to assess land suitability for PV deployment. Second, we estimate the probability distribution of cropland redevelopment strategies based on statistical measurement principles, calculate the net emissions reduction expectations of Natural Climate Solutions (NCS), and further validate the emissions reduction potential of photovoltaics. Third, utilizing multiple policy scenarios from the AR6 assessment, we estimate the net cash flows and cumulative cost changes of PV deployment compared to other emissions reduction strategies under different future climate policy scenarios from 2020 to 2050. Finally, we integrate these three dimensions to generate a PV deployment priority decision map (Fig.1), which could serve as a decision-making tool for US and other countries.

# Result 

## US Abandoned Cropland Similarity for PV Opportunities

根据全球协调一致的长时序ESA土地分类数据和OSM数据方法（详见方法），交代观测到的总撂荒地面积以及光伏总体的数量。

结合机器学习我们学习了（汇报模型性能），并且分为绝对相似性、相对相似性

对比以往的研究，候选为光伏部署的撂荒用地分区统计（主要根据州来判断，交代总体平均值与标准误），并且说持续的duraiton\发生年份的众数or空间分布。

At local scales, 那些最适合部署光伏的撂荒耕地

（与当前的光伏分布是否存在差异？）

## Net Carbon Mitigation Expectation across Multi-sector Land Use

生物多样性与气候变化不一定guranteed to benefit 


## Economic Feasibility of PV Deployment under Future Policy Scenarios

# Discussion 


# Method 

## Environmental similarities between deployed PV farms and abandoned cropland

Mapping of Abandoned Cropland and PV Farm. In this article, we focus on the abandoned cropland in US for PV deployment decision-making. This choice is based on the fact that abandoned cropland, as pre-cultivated areas, represents natural ideal sites for large-scale PV deployment. Furthermore, the widespread abandonment of cropland across the Global North has attracted increasing attention from land managers. According to the FAO definition, cropland abandonment refers to cropland that is no longer actively cultivated and has been converted to other ecosystem types without direct human intervention for at least 5 consecutive years. Following this definition, we performed moving window detection on the ESA-CCI land cover maps from 1992-2022, which are spatiotemporally harmonized and widely used for long-term land change monitoring. For further reducing false detections due to crop rotations and temporal resolution limitations that introduce uncertainty in abandonment detection, we standardized the detection algorithm operations: 1) We aggregated the original 37 land cover classes into 9 major classes into 1km × 1km, including 5 ecosystem types (forest, wetlands, grassland, arid ecosystem, and shrublands), 1 cropland class (including all six cropland subclasses), 2 non-vegetation areas (water bodies, bare areas, and permanent ice), and 1 human settlements area. 2) A 5-year moving window was used for cropland abandonment detection, with areas converted to human settlements and reclaimed cropland (pixels that were once detected as abandoned but reverted to cropland for 2 successive years) excluded from the analysis. 3) Each abandoned cropland pixel was labeled with the abandonment start year, duration period, reclamation status, and abandonment status in the target year for auxiliary analysis and validation.

Current time-series datasets for large-scale PV deployment remain scarce, making it difficult to identify spatiotemporal variations in environmental preferences for PV construction. Building upon the work of Dunnett et al., we further extended the PV dataset to include time-series information, enabling policymakers to access customized data for policy-relevant regions and time periods globally. This study utilizes the OpenStreetMap (OSM) online access interface to obtain point and polygon data for solar farms in the United States for 2015 and 2020, applying a recommended 400-meter neighborhood radius for the density-based spatial clustering with noise (DBSCAN) algorithm. This approach was chosen because we focus on high-level solar farm clusters, and within the 300-500 meter search radius, the gradient decline in unclustered PV vector counts is most rapid.

Modeling environmental similarity on abandoned cropland. This step involves a series of socio-economic and physical environmental predictors to more precisely generate probability surfaces for the likely development of renewable infrastructure. Ongoing research in advanced machine learning for capturing high-dimensional PV deployment features has primarily focused on learning positive sample characteristics while neglecting the environmental features of strong negative samples. However, in real-world PV deployment scenarios, many factors act as absolute constraints (such as water bodies, glaciers, nature reserves, disaster-prone areas, and densely populated urban residential areas), which may reduce model generalization capabilities. Therefore, we developed a two-stage machine learning model based on environmental similarity. The model incorporates features including physical geographic environment (land cover, DEM, slope, gross dry matter productivity), climatic factors (near-surface air temperature, wind speed, shortwave radiation), and socio-economic factors (population density, distance to human settlements, road density, GDP per capita). First, we fit a kernel density estimation (KDE) model to the environmental features of historical PV farms to obtain their underlying density distribution. Then, we compute the density of each candidate abandoned cropland under this KDE model as an environmental-similarity score, and select the bottom 5–10% of locations as strong negative samples. Second, after completing the positive-negative sample assembly, we employed a two-stage modeling approach based on the DeepFM deep learning model to capture multi-level, high-order, and complex interactions among different features, while also addressing the spatial sparsity issue of different features in abandoned cropland. This approach outputs and predicts the probability of PV deployment for candidate abandoned cropland sites.

## Net Carbon Abatement Expectation

Assessing PV carbon mitigation potential. The carbon emissions mitigated by PV power were calculated by replacing grid power using the baseline emission factors of the national grid. PV power generation is determined by PV power generation potential ($PV_{POT}$) and the installed capacity. The $PV_{POT}$, which describes the performance of PV cells constrast to the nominal power capacity, depends on primarily on actual environmental conditions. Following previous studies, $PV_{POT}$ can be calculated as follows:

$$PV_{POT}=P_{R}\cdot\frac{I}{I_{STC}}$$
$$P_{R}=1+γ\cdot(T_{cell} - T_{STC})$$

where $I$ represents surface downwelling shortwave radiation and $P_{R}$ is the performance correlation ratio determined by temperature. The standard state referred to in this paper includes the shortwave flux on the PV cell ($I_{STC}: 1000 W m^{-2}$), the temperature of PV panel ($T_{STC}: 25 ℃$) , the temperature coefficient ($γ:-0.005 ℃^{-1}$). $T_{Cell}$ represent the temperature of PV cell and can be approximated as follows: 
$$T_{cell}=a_{1}+a_{2}\cdot T+a_{3}\cdot I+a_{4}\cdot W$$

where $a_{1},a_{2},a_{3}$ and $a_{4}$ are taken as 4.3 ℃, 0.943 , 0.028 $℃ (W m^{-2})^{-1}$ and $-1.528 ℃(m s^{-1})^{-1}$, respectively. Those parameters are proven to be independent on location and cell technology, and have been widely used for estimating PV performance. To convert PV electricity generation into mitigation benefits, we followed the Harimonize Approach to Greenhouse Gas Accounting in accordance with the United Nations Framework Convention on Climate Change (UNFCCC). This method reflects the emission intensity of the marginal electricity grid displaced by renewable generation, typically approximated using the Combined Margin (CM). The CM for the grid is comprised of an Operating Margin (OM) and a Build Margin (BM). In principle, the OM consists of existing power plants that would be affected and have the highest variable operating cost. The BM represents the cohort of the future power plants whose construction could be affected by PV project, based on an average of future emission intensities of new electricity generation. 

$$P_{carbon}=P_{POT} \cdot  P_{Installed}\cdot EF_{CM}$$
$$EF_{CM}= EF_{OM} \cdot W_{OM}+ EF_{BM}\cdot W_{BM}$$

where $EF_{OM}$ is the OM factor and $EF_{BM}$ is BM factor. 


Allocating reuse probability based on spatial-temporal joint distribution. Research from Crawford reveals the conversion instability of abandoned cropland, with an average duration of only 14.22 years. This means that during our 30-year study period, more than half of the abandoned cropland may revert to cultivated land. Additionally, for the original land use trajectories of abandoned cropland, there are different conversion pathways such as forest restoration and non-woody vegetation recovery. This study combines existing research on the empirical function of cropland reclamation time and historical spatial change trajectories of abandoned cropland to construct a spatiotemporal joint distribution model for abandoned cropland redevelopment strategies. Specifically, on the one hand, in the spatial dimension, we identified the land type with the highest frequency of occurrence for each abandoned cropland pixel during its duration based on measure theory. Following 10×10 comprehensive planning units, we generated density variables based on the proportion of different land use types, and used inverse distance weighting to generate probabilities for reclamation, forest restoration, and vegetation recovery strategies. On the other hand, in the temporal dimension, we established a model based on existing research on the remaining reclamation proportion over time t, following the formula below.

Calculating carbon expectation across land sectors. 我们主要聚焦在复垦、复林、复草三种不同土地部门活动带来的减排作用，综合上一步得出的再开发概率分布计算出碳期望。






## Cost-benefit Analysis across Climate Scenarios

### 