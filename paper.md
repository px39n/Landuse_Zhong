



Abstract: The rapid expansion of photovoltaic facilities has intensified cross-sector spatial competition, creating climate decision uncertainties. Reallocating marginal land offers a low-cost policy solution. We conducted a spatially explicit assessment of cross-sector carbon mitigation potential and developed an integrated environment-emission-economic feasibility (3EF) decision framework for large-scale PV deployment on abandoned cropland. Using xx ha of U.S. abandoned cropland (1992-2020), we found substantial electricity generation substitution opportunities, projecting xx kW/yr average generation from 2020-2050, cumulatively equivalent to xx Gt C—one-third of global mitigation targets. Statistical analysis revealed abandoned cropland primarily reallocates to forestry (xx ha, >60% probability). Land-based multi-sector strategies sequester xx Mg C over 30 years, representing 1/50th of energy sector mitigation. Additionally, 基于未来气候政策情景的全生命周期净现金流分析shows by 2050, 利用撂荒地大规模部署光伏实现累积收益xx USD，平均投资回收期xx 年。Furthermore, Global abandoned cropland deployment could achieve net-zero emissions through grid substitution, providing critical decision insights for clean energy transition.




为了解决recent大规模扩张的光伏场所引发的能源、农业、森林等跨部门空间竞争带来的气候决策不确定性，重新分配emerging marginal land成为一种低成本且具有可行性的promisong policy tools。Here，我们进行了空间显式基于土地的跨部门碳减排潜力评估研究，并提供了整合environment, emmission, economic feasibility (3EF) 大规模部署PV的撂荒地climate mitigation decision地图。研究发现，利用美国存在的xx ha撂荒地（from 1992-2020）大规模部署光伏存在巨大的electricity generation替代性减排机会，预计从2020-2050年期间，光伏平均能够带来xxkw/yr的发电，累积相当于xx Gt C，这相当于全球减排目标的1/3。我们基于统计测度理论估计了能源部门政策未干预下撂荒地用于农业、森林、植被恢复策略的概率分布，发现美国的撂荒地主要倾向于再分配给森林部门（xx ha, >60%概率），这些区域同时也是（xx简单结合本地州的情况）。相较于current carbon storage, 基于土地的多部门气候减排策略在30年年共计expected固定xx Mg C，但总体为能源部门碳减排的1/50。此外，结合未来气候政策情景评估部署PV全生命周期的经济可行性，发现到2050年利用撂荒地部署PV光伏的累积NPV为xx USD，累积成本xx USD，平均投资回收期为xx 年，这是传统报酬相对较高的农业部门xx倍（森林部门与植被恢复主要表现为支出）。在政策展望上，为了迈向全球2050碳中和目标，可能仅需要将全球的撂荒地用于光伏部署，就可以通过电网的替代性减排满足全球气候减排就足以实现全球净零排放，这一进步指导了清洁能源转型on-the-ground decision insight。


# Introduction 

To cap cumulative carbon emissions at approximately 335 PgC from 2023 onward and meet global climate targets, the large‑scale deployment of low‑carbon photovoltaic (PV) systems is indispensable. Projections indicate that by 2050, solar PV will become the world’s leading electricity source—supplying 25 % of global demand with some 8,519 GW of installed capacity (IRENA, 2019a). As the dominant clean‑energy market, the United States (US) plans to expand its solar share to roughly 30 % by mid‑century. However, the rapid rollout of utility‑scale PV in rural America threatens productive cropland and may impose adverse environmental impacts. This tension has spurred growing interest in natural and land-based climate solutions, like allocating PV or afforestation on marginal lands, including abandoned cropland, open-pit mining sites, and urban-rural settlements. Despite numerous promising pathways, the real feasibility of large-scale PV deployment at the national scale remains largely unknown, thereby clouding policymakers' confidence in advancing energy transitions. To fill this gap, we quantify both the land-specific PV deployment potential from 2020-2050 and the decision-making trade-offs, including environmental suitability, carbon abatement ability, and economic feasibility.

Solar energy is essential for climate mitigation. Unlike nuclear, bioenergy, or geothermal energy, photovoltaic (PV) farm siting is highly sensitive to climatic conditions. Recent research shows strong compatibility between agricultural land and PV deployment, as farming activities already exist in regions with favorable solar radiation, temperature, and wind conditions. While US croplands expand over 0.4 million hectares annually, substantial abandonment also occurs in areas like the Mid-Atlantic Coast, aligning with the accelerating abandonment trends observed across the Global North. Similar to agrivoltaic co-location and perovskite photovoltaic deployment approaches, redeveloping abandoned cropland for PV installation represents a promising emissions reduction pathway that could provide unique spatial opportunities. However, large-scale PV deployment requires local energy infrastructure and socio-economic support, which vary significantly across regions. Previous research has inadequately captured these national-specific spatiotemporal heterogeneities. Recent advances in machine learning and high-dimensional statistical data embedding offer opportunities to extract multi-dimensional characteristics from historical PV installation data and generalize these insights to specific policy target regions.

Although Natural and Land-based Climate Solutions (NLCS) are widely recognized as the most cost-effective approach, forest ecosystems require decades for long-term carbon sequestration and have high land demand. Field experiments from the United States, Israel etc., have demonstrated that despite PV deployment leading to a significantly decrease in surface albedo and consequently increase local temperature, the atmospheric carbon mitigation efficiency is ~50× than afforestation. Furthermore, as the US levelized cost of PV electricity generation continues to decline to $0.03 by 2030, large-scale PV deployment will become increasingly economically viable. However, current assessments of solar deployment potential remain limited in scope, primarily focusing on electricity generation potential while neglecting the broader socio-economic and policy implications. The absence of comprehensive cross-sectoral comparative evaluations (e.g., agriculture, forestry) may significantly underestimate the policy potential of large-scale PV deployment and create uncertainty for policymakers in advancing energy transitions.

Here, we propose an extensible land-specific carbon abatement decision-making framework. First, we identify the spatiotemporal distribution of abandoned cropland and existing solar farm locations across US using time-series land-use imagery and OpenStreetMap (OSM) geographic data. Subsequently, we train an environmentally-similarity-based machine learning model that integrates multi-source spatiotemporal data including climatic, geographic-physical, and socio-economic factors to assess land suitability for PV deployment. Second, we estimate the probability distribution of cropland redevelopment strategies based on statistical measurement principles, calculate the net emissions reduction expectations of Natural Climate Solutions (NCS), and further validate the emissions reduction potential of photovoltaics. Third, utilizing multiple policy scenarios from the AR6 assessment, we estimate the net cash flows and cumulative cost changes of PV deployment compared to other emissions reduction strategies under different future climate policy scenarios from 2020 to 2050. Finally, we integrate these three dimensions to generate a PV deployment priority decision map (Fig.1), which could serve as a decision-making tool for US and other countries.

# Result 

## Land Priorities for PV deploying 



## US Abandoned Cropland Similarity for PV Opportunities

根据全球协调一致的长时序ESA土地分类数据和OSM数据方法（详见方法），我们观测到的总撂荒地面积以及光伏总体的数量。对比以往的研究，这些候选为光伏部署的撂荒用地分区统计（主要根据州来判断，交代总体平均值与标准误），并且说持续的duraiton\发生年份的众数or空间分布。

结合机器学习我们学习了（汇报模型性能），并且进一步分为绝对相似性、相对相似性，并且汇报均值、标准误以及分布形态。

At local scales, 那些最适合部署光伏的撂荒耕地

（与当前的光伏分布是否存在差异？与现状撂荒地的差异主要在哪里？具体来看各个因子的贡献、以及这些因子的贡献与农业适宜性的对话）

## Net Carbon Mitigation Expectation across Multi-sector Land Use

气候变化不一定guranteed to benefit when cropland is abandoned, 这决定了撂荒地in pratice的再利用方式存在比较大的不确定性。本研究基于统计测度理论，按照10×10pixel构建了撂荒地土地利用的时空偏好概率分布，重点对比了复林、复垦以及植被恢复三种不同策略从2020-2050年的碳减排期望（Fig.2）。 从减排能力来看，2020-2050之间能源转型驱动的光伏碳减排能力要远高于土地部门驱动的碳减排效益，能够贡献多少 MgC/ha，

从传统土地部门策略来看，美国撂荒地在区域层面仍然偏好于森林用途，xx%概率；农业，xx%概率；植被，xx%概率。具体到county-level,阐述这些概率怎么分配



## Economic Feasibility of PV Deployment under Future Policy Scenarios

未来气候情景下不同国家政策情景



# Discussion 

本文的研究发现...。事实上，综合最近的有关将光伏、风电等关键基础设施布局到marginal spcace已经成为研究与政策重点。对比城市屋顶、废弃矿坑，我们提出的撂荒地转型是更激进但可能也是更具吸引力的战略。不止在美国，从全球来看，撂荒耕地达到了101Mha，这达到了全球耕地总体的6.5%。


近年来有关能源转型驱动下的土地利用预测研究



本研究显式地绘制了空间决策优先级地图。但是现实政策制定中受限于财政预算、国际贸易、市场需求等具有强时间变化性的变量，

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