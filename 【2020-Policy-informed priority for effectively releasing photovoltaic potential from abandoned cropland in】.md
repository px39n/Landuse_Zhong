# **Policy-informed priority for effectively releasing photovoltaic potential from abandoned cropland in the United States**

Pengyu Zhong1,3, Wenze Yue<sup>1</sup> , Yang Chen<sup>X</sup> , Tianyu Wang<sup>2</sup> , Sisi Meng<sup>4</sup>

**Abstract:** Abandoned lands provide critical spatial opportunities for deploying photovoltaic (PV) systems and implementing land-based natural climate solutions (LNCS), yet translating this potential into actionable policy strategies remains poorly understood. Here, we develop a policy-informed prioritization framework that integrates machine learning with multi-criteria decision analysis to evaluate environmental suitability, emission mitigation, and economic viability (3E objectives). In the United States, 4.7 Mha of abandoned cropland identified from 1992–2020 exhibit strong environmental similarity to existing PV sites, with a mean deployment probability of 84.0 ± 0.2%. Deploying PV on these lands could offset averagely 62.83 ± 17.05 Gt CO<sup>2</sup> over 2020–2050 (~16× LNCS) and their economic breakeven levels vary widely across policy scenarios (–2726 k USD ha⁻¹ to +2,424 k USD ha⁻¹). Only 10% of total abandoned croplands could meet 30% of US 2050 solar targets, and 33 states could satisfy all sectoral electricity demand under the mean future scenario. However, single-criterion deployment priority based on historical priors demonstrates substantial mismatch to the future policy requirements. The cumulative net economic gaps ranging from -206 billion USD to 1462 billion USD. By introducing an adaptive 3E-synergy index, our framework improves co-benefits efficiency by 38.4% across 48 states. Bayesian hierarchical analysis further identifies solar radiation as the only consistently stable driver under decision uncertainty. These findings highlight the latent potential of abandoned croplands as a nexus where renewable energy transformation and climate-neutral goals converge.

<sup>1</sup> Department of Land management, Zhejiang University, Hangzhou, China

<sup>2</sup> Department of Geography, The University of Hong Kong, Hong Kong, China

<sup>3</sup> Keough School of Global Affairs, University of Notre Dame, Notre Dame, USA

<sup>4</sup> Cornell SC Johnson College of Business, Cornell University, USA

## **1 Introduction**

Low-carbon photovoltaic (PV) systems play a critical role in climate neutrality1–3 . By 2050, global PV capacity is projected to be the world's largest electricity source, meeting 25% of total demand and providing 21% of the CO<sup>2</sup> emission reductions (~4.9 Gt annually)4,5 . However, the rapid rollout of PV has raised concerns about prime cropland conversion and ecological degradation. Globally, croplands now comprise 43% of global PV land use<sup>6</sup> and even in the United States (US), several states have imposed moratoriums on solar development in croplands<sup>7</sup> . This tension has catalyzed a paradigm shift toward marginal land utilization, where PV and landbased natural climate solutions (LNCS) such as ecosystem restoration can be strategically deployed. Recent studies on PV deployment across dessert areas<sup>8</sup> , openpit mining sites<sup>9</sup> , urban-rural settlements or rooftops<sup>10</sup>, and carbon-intensive regions<sup>11</sup> have identified cumulative PV installation potential exceeding 30,000 GW, covering 219-336,000 km² land area. These projections substantially exceed market-based forecasts several times such as those from the IEA (~15,468 GW) <sup>12</sup> and IREA (~8,519 GW) 13 . While promising in power generation, insufficient research pays attention to translating such potential into feasible policy action. The guidance on where to prioritize PV deployment and how different strategies deliver benefits heterogeneously remains largely unknown. With the US aiming to raise its solar share to 30% (~920 GW) by mid-century<sup>14</sup>, there is an urgent need for spatially explicit frameworks that balance environmental, economic, and emission trade-offs to inform policy decisions.

Unlike nuclear, bioenergy, or geothermal energy, the power output of PV is highly sensitive to environmental conditions and varies markedly across space<sup>15</sup> . Croplands are recognized as ideal recipient environments for PV because of favorable solar radiation and infrastructure support. Yet amid escalating debates on PV and food security, substantial cropland abandonment emerges worldwide, increasingly commanding policy and research attention across nations<sup>16</sup>. This stems from their rich policy implications for mutually reinforcing synergies between social and biophysical potentials17,18 . For instance, ref. <sup>19</sup> has revealed that abandoned croplands represent an important opportunity for biodiversity recovery and provide forest habitats, albeit reducing landscape heterogeneity to some extent. Similarly, ref. <sup>20</sup> assessed the production and carbon sequestration potential provided by abandoned croplands under diverse reuse pathways, including recultivation and forest restoration. These former croplands represent ideal sites for both LNCS and PV deployment owing to their suitable climatic conditions and water availability, and flat terrain. Moreover, amid the surging demand for off-grid energy driven by the expansion of large-scale data centers, abandoned croplands also offer strategic spaces for decentralized power generation and investment that foster rural revitalization and labor market. However, while some field experiments indicate that PV outperforms afforestation in atmospheric carbon mitigation by approximately 50-fold21,22, LNCS remain widely recognized as the most cost-effective mitigation pathway22,23 . Yet, the interactions among different climate solutions, as well as large scale spatial assessments that jointly quantify their emission mitigation potential and economic feasibility, which are two key determinants of climate strategies, remain largely unexplored. In response to the ongoing encroachment on prime cropland, US recent efforts have simultaneously advanced PV installation technologies such as perovskite solar cells<sup>24</sup> and agrivoltaics co-location<sup>25</sup>, while expanding toward strategic spatial prioritization to rank areas for renewable energy deployment<sup>26</sup> . Building on this shift, the rapid development of machine learning has markedly enhanced our understanding of interactions between renewable energy projects and socio-environmental systems. These models enable high-dimensional feature embedding that not only enhances deployment prediction accuracy but also reveals latent structures underlying these complex relationships<sup>27</sup> . Despite the proliferation of machine learning models, several limitations hinder their translation into actionable policy. First, conventional models primarily depend on historical environmental profiles and expert-driven priors, which

fail to capture forward-looking policy objectives that incorporate the subjective and evolving preferences of decision-makers<sup>28</sup> . Second, rare negative labels poses a major challenge for classification models, particularly supervised learning approaches applied in policy making<sup>29</sup> . These models are prone to false positives (Type I errors, i.e., misclassifying unsuitable sites as suitable), where the resulting societal risks can be substantial. Third, black-box models fall short in explicitly disentangle or quantify the interactions and trade-offs among multiple objectives<sup>30</sup> . In climate decisionmaking, where environmental suitability, emission mitigation potential, and economic feasibility are key determinants shaped by varying preferences, this opacity reduces policymakers' confidence in prioritizing PV deployment. These challenges echo with broader concerns regarding the transparency and generalizability of data-driven decisions for climate change.

To address these challenges, machine learning models incorporating prior assumptions and negative sampling strategies offer promising alternatives. In particular, generative models grounded in distributional learning can jointly capture multidimensional data structures and quantify inter-observational similarity, thereby leveraging high-dimensional environmental features to generate more robust and interpretable spatial prioritization outcomes for PV deployment. At the same time, integrating multi-criteria into spatial prioritization allows policymakers at national and local levels to compare the feasibility of alternative strategies and realize socioenvironmental co-benefits. This approach has been applied in watershed management<sup>31</sup>, conservation planning<sup>32</sup>, and ecological restoration<sup>33</sup> to balance multiple objectives. In our case, we focus on emerging abandoned cropland in the US as areas of interest for PV deployment, addressing the following research questions: (1) What environmental structures support large-scale PV deployment in the US, and whether abandoned croplands suitable for PV deployment? (2) Does PV deployment significantly outperform LNCS in emissions mitigation ability and economic viability? And whether these strategies are complementarity? (3) How to quantify the interplay among multi-criteria decisions and translate the theoretical potential of marginal land for PV deployment into policy action?

Here, we develop a comprehensive assessment framework integrating two-stage machine learning with multi-criteria decision analysis across environmental suitability, emission mitigation, and economic viability (3E). And we introduce the 3E-synergy index as a key metric for PV deployment prioritization at both nationalstate level. First, long time-series land use imagery and PV site inventories are used to generate training samples and feature embeddings. A two-stage model then applies Gaussian Mixture Models to generative environmentally dissimilar negatives. And a Transformer-ResNet model followed by aims to estimate the environmental suitability of deploying PV on abandoned croplands, which is defined as the probability surface generated from historical profiles. Second, we quantify the emission mitigation ability and economic viability from PV power generation over the 2020-2050 and compare them with those of LNCS opportunity. Third, by integrating cost-benefit data from IPCC AR6 multi-policy scenarios, we assess the economic viability of PV versus LNCS. Finally, we synthesize a data-driven 3E-synergy index integrating environmental, emission, and economic dimensions, and validate its effectiveness through hypothesis testing. We further identify the key drivers of 3E-synergy (intraand inter-state) using hierarchical Bayesian inference, thereby supporting PV deployment decision-making from national to local levels. Beyond abandoned farmland, our integrated assessment framework holds application potential for transferring to different countries to formulate decisions on priority zones for spatial deployment of various facilities, contributing to more robust advancement of the energy transition.

# **2 Result**

### *2.1 Environmental suitability for deploying PV on abandoned cropland*

Using long-term harmonized ESA-CCI land-use imagery from 1992 to 2020, we identified 4,703 kha of abandoned cropland across the US, providing substantial marginal space for PV deployment, which is equivalent to 20.9 times the current footprint of ground-mounted large scale US PV facilities (225 kha totally) 34 . Consistent with earlier findings35,36, these sites are primarily concentrated in Texas (515 kha, 11.0%), Illinois (295 kha, 6.3%), and California (272 kha, 5.79%). To evaluate their environmental suitability for PV deployment from historical priors, we developed a two-stage machine learning model incorporating 15 environmental predictors describing infrastructural, socioeconomic, geophysical, and climatic environmental conditions<sup>37</sup> .

The first-stage analysis resolves the environmental structure of existing PV by fitting a 23-component Gaussian Mixture Model (GMM), with model selection guided by the Bayesian Information Criterion (BIC) and log-likelihood (Fig. 1a). A subsequent Ward hierarchical clustering groups these components into three dominant environmental archetypes: socioeconomically active regions with mature infrastructure (38.9%, cluster 1), biomass-rich mountainous systems along the Rocky Mountain corridor (8.2%, cluster 2), and sparsely populated, geophysically flat landscapes (52.9%, cluster 3), the latter of which is primarily concentrated across the Midwest (Fig 1e). Building on this environmental structure, we find that existing PV and abandoned croplands exhibit highly similar log-density distributions (Fig. 1d), indicating strong overlap in their underlying environmental niches. Notably, component 20 alone accounts for nearly half of all abandoned sites (46.2%), underscoring that pre-cultivated environmental settings remain intrinsically suitable for PV deployment.

Leveraging PV environmental distribution, the second stage employs a Transformer-ResNet classifier trained on positive PV samples and GMM-generated negatives using a 1:1 sampling ratio. The trained model is then applied to estimate site-level suitability across abandoned croplands. Compared with the MLP benchmark (F1 = 0.87), our mixture architecture yields improved performance (F1 = 0.90). Nationally, abandoned croplands exhibit high suitability for PV deployment, with a mean predicted probability of 84.1 ± 0.3%. High-suitability regions, largely concentrated on the Rust Belt and East Coast. However, substantial regional variation exists, suggesting the need for state-specific PV deployment strategies. For instance, despite California being the second-largest PV market in the United States, its abandoned croplands exhibit a mean suitability probability of 73.6%, ranking 39th out of the 48 mainland states.

![](paper_b40afd_images/_page_7_Figure_0.jpeg)

**Fig.1 Environmental suitability for deploying PV on abandoned croplands and their environmental profiles in US. a-c**, Environmental structure of PV site selection, illustrating how heterogeneous local features summarized into interpretable environmental clusters. **a,** GMM-based decomposition into 23 environmental components of PV sites. **b,** Feature contributions to each component grouped by thematic drivers. **c,** Environmental components condensed into three dominant clusters with Ward hierarchical. **d-e,** Environmental similarity between PV and abandoned croplands. Distributional similarity of GMM log-density **(d)** are further aggregated in ~4,500 km² equal-area hexagons **(e)**, with pie charts showing dominant PV environmental clusters. **f-g,** PV deployment probability surface on abandoned croplands Transformer-ResNet **(f)**, visualized by decile-based bins**(g)**.

![](paper_b40afd_images/_page_8_Figure_1.jpeg)

**Fig.2** Carbon emission mitigation ability of PV and LNCS from reusing abandoned cropland. **a–d,** Overlay of PV and LNCS CO₂ mitigation ability per hectare from 2020–2050.The 5×5 bivariate quantile color matrix **(a)** illustrates the potential tradeoffs between the two strategies. The multi-scenario line plot **(b)** illustrates the temporal evolution of the emission mitigation potential of PV electricity substitution under alternative climate policy assumptions and full land availability. The distributional curve **(c)** and bar chart **(d)** present the aggregated mitigation capacity of PV and LNCS, where total mitigation outcomes are calculated by combining PV generation benefits adjusted by policy-dependent power grid carbon intensity with weighted LNCS contributions across different deployment strategies. **d-f,** Probability

of LNCS implementation and expected CO₂ mitigation outcomes arrogated by US states, including agriculture **(e)**, forest **(f)**, and non-woody vegetation **(g)**.

PV deployment and land-based natural climate solutions (LNCS) represent two major mitigation pathways on abandoned croplands. Here, emission mitigation ability is quantified as the net CO<sup>2</sup> reduction from PV power substituting grid electricity minus the potential emission reduction from LNCS derived through adjacent land use preference.

PV emission mitigation potential is primarily governed by solar irradiance and grid carbon intensity. Average US solar irradiance is approximately 1,602 kWh m-2 yr-1 , concentrated in the West Coast and US-Mexico border regions. Western states like Arizona (2,100 kWh m-2 yr⁻¹) and New Mexico (2,075 kWh m-2 yr-1 ) exhibit the highest solar potential while northeastern such as New York (1,369 kWh m-2 yr-1 ) and Pennsylvania (1,418 kWh m-2 yr-1 ) show relatively lower values. Under these conditions, US abandoned croplands could provide a cumulative installed capacity of 7,996 GW, generating 10.04 petawatt-hours per year (PWh yr-1 ), equivalent to 8.23 times the US household electricity consumption (1.31 PWh yr-1 ) 38 .

Across ten demand and grid marginal scenarios from the IEA and EIA, deploying PV on abandoned cropland could mitigate an average of 62.83 ± 17.05 Gt CO<sup>2</sup> over 2020–2050. This mitigation potential far exceeds that of LNCS, with PV delivering roughly 16 times the CO<sup>2</sup> reduction achieved by LNCS strategies (3.91 Gt CO₂). Accounting for the opportunity cost of foregone LNCS mitigation, the net PV mitigation potential is 58.92 ± 17.05 Gt CO₂ over 2020–2050, corresponding to 23.5% of the remaining carbon budget (250 Gt CO₂) associated with a 50% chance of limiting warming to 1.5°C.

Site-level analyses reveal pronounced spatial heterogeneity in mitigation outcomes. Only 6.2% of abandoned cropland sites simultaneously rank within the top 30% for both PV generation capacity and LNCS potential, with the mean carbon storage 428.51 MgC ha-1 . This pattern indicates that PV deployment and LNCS

implementation are not mutually exclusive, emphasizing the necessity of pixel-level prioritization.

#### *2.3 Economic viability under future climate policy scenarios*

![](paper_b40afd_images/_page_10_Figure_2.jpeg)

**Fig.3 Lifespan economic viability of PV deploying under future climate polies by assembling AR6-IPCC dataset. a,** PV economic waterfall matrix across global coordinated policy pathways (P1–P3) and varying policy implementation intensities (Ta–Tc). Economic benefit flows are decomposed into revenues (blue), expenditures (red), opportunity costs (yellow), and net profits (green). **b,** Spatial distribution of lifespan economic viability per hectare after state-level demand adjustment and is visualized by decile-based bins.

Although economic constraints play a crucial role in policy implementation, they have received limited attention in previous studies. In our study, economic viability is defined as the net economic benefit derived from PV power generation revenue deducting expenditure costs and LNCS opportunities over the 2020–2050 period. We evaluated economic viability using policy scenarios from the Sixth Assessment Report (AR6) dataset, assembling a policy matrix to produce a overall economic benefit map with detailed LNCS opportunity cost estimates.

Our results show that PV profitability is highly sensitive to both policy timing and implementation pathways (Fig. 3a). Under scenarios with no globally coordinated action (P1), nearly all US PV sites fail to generate positive net economic benefits,

averaging -1,646 kUSD ha-1 over the PV lifespan. This economic burden is substantially improved under Immediate Action (P2) and Delayed Action (P3) scenarios, yielding mean economic outcomes of -270 kUSD ha-1 and +2,174 kUSD ha-1 , respectively.

Regarding specific policy initiatives (i.e. Nationally Determined Contributions, emission permit transfers, and additional sectoral policies), we find that different policy initiatives siginificantly affect PV cost-benefit break-even line. For instance, the global emission permit trading imposes direct policy costs accorss US, rangin from -24 kUSD ha-1 to -2,119 kUSD ha-1 (P2a to P2c). Whereas under preceding NDC-based scenarios, the variation ranges from -1731 kUSD ha-1 to 439 kUSD ha-1 (P3a to P3b).

Beyond aggregate outcomes, we find that economic benefits propagate unevenly across geographic space. By compressing multi-scenario and LNCS cost-benefit information into an overall mean metric, our analysis shows that PV sites with higher net economic benefits are primarily concentrated along the Pacific Coast, South Atlantic, and arid regions of East South Central (Fig. 3b). Conversely, regions characterized by high installed capacity but limited power generation potential may incur substantial economic losses during large-scale PV deployment. At the state level, the lowest economic decile experiences an average cumulative loss of -116.62 billion USD , whereas the highest decile achieves a mean cumulative gain of +298.18 billion USD for the highest decile.

Moreover, relative to the pronounced economic uncertainty arising from spatial heterogeneity in PV performance, our research demonstrates that LNCS strategies could provide a cost-effective and fiscally stable pathway for achieving climate neutrality. The overall mean economic return of PV deployment ranging from -1,266 kUSD ha-1 to +1,664 kUSD ha-1 . Only 14 of the 48 contiguous US states achieving positive net returns. In contrast, LNCS implementation costs less tha 1 kUSD ha-1 after accounting for reasonable forestry and agricultural product revenues and

#### *2.4 3E-synergy priorities for PV deploying*

![](paper_b40afd_images/_page_12_Figure_2.jpeg)

**Fig.4 3E outcomes of single-criteria deploying PV on croplands and 3E-synery index combination. a.** Priority map (3E-synergy index) of abandoned cropland for PV deployment. Unshaded regions indicate states where electricity demand across all sectors under the mean 2050 demand scenario can be fully met abandoned croplands. **b-d,** Policy performance under the 3E-synergy solution compared with single-criteria strategies, including **(b)** environmental sustainability (probability, unitless), **(c)** carbon emission mitigation ability(kt CO<sup>2</sup> ha-1 ), and **(d)** economic viability (kUSD ha-1 ). **e-g**, Distribution of Top 50% priority areas under single-criteria strategies and cumulative cross-target performance. Cumulative curves are computed by multiplying raster

densities by pixel area and integrating across the prioritized area, referring to **(e)** total environmental scores (unitless 1e6), **(f)** emission mitigation (Gt CO2), and **(g)** economic expectation (billion USD in 2020; negative-NPV region shaded; dashed red line indicates the break-even level). Notably, deploying PV on only the top 10% of abandoned cropland would be sufficient to fulfill the targeted U.S. energy transition.

To translate PV potential into policy, we establish a policy-informed prioritization framework and a corresponding policy performance metric (See Methods). The result shows that PV deployment priorities based on different criteria lead to distinct cumulative outcomes (Fig. 5). In particular, when focusing solely on optimal environmental suitability, carbon mitigation ability, or economic viability, the performance of other dimensions declines showing pronounced trade-offs (Fig. 4e-g). If all projected US PV facilities by 2050 (~740 GW) are installed on abandoned cropland, less than ten percent of total abandoned area (4,376 kha) would be required. Towards this aim, however, single-criterion prioritization results in substantial cumulative shortfalls, reaching 36.6% for environmental suitability and 27.6% for emission mitigation ability. The economic dimension exhibits the largest divergence with accumulative renvenue ranging from -107 billion to +714 billion USD. The results of performance evaluation show that trade-offs are most pronounced in the economic dimension (Fig. 4b-d), where unit-area benefits range from -10 kUSD ha-1 under suitability-based priority to 226.84 kUSD ha-1 under maximizing economic efficiency. Consistent with this dispersion, the coefficient of variation for economic viability reaches 0.95, substantially exceeding those of environmental suitability (0.09) and emission mitigation ability (0.06).

To alleviate the inherent trade-offs among environmental suitability, emission mitigation ability, and economic viability, we developed a self-adaptive 3E-synergy index for prioritizing PV deployment on abandoned croplands (Fig. 4a). A higher 3Esynergy value identifies sites that achieve consistently strong performance across all three dimensions with low outcome variance, thereby delineating priority zones for coordinated development (Fig. 5). The spatial distribution of 3E synergy exhibits

pronounced regional differentiation. High-synergy clusters are concentrated in the southern and southwestern US, followed by a transitional belt across the central and southeastern states, while lower synergy values dominate northern agricultural regions and the Great Lakes area. Compared with single-criterion priority schemes, the 3Esynergy priority improves environmental suitability by 6.9%, emission mitigation ability by 2.8%, and economic viability by 629%, demonstrating its effectiveness in simultaneously enhancing policy performance across multiple dimensions. Additionally, accounting for state-level electricity demand constraints under future scenarios, we find that 33 out of the 48 US states could fully meet their all-sectoral electricity demand through PV deployment on abandoned croplands in 2050, based on the average demand scenarios from NREL<sup>39</sup>. Under the 3E-synergy solution, achieving this target requires 46% of total abandoned cropland area.

![](paper_b40afd_images/_page_14_Figure_1.jpeg)

**Fig.5 Policy performance across multiple criteria under policy-informed priority framework.** The continuous response surface is fitted from 5,000 uniformly sampled preference weight combinations, representing expected policy outcomes under alternative priority schemes. The triangular color bar encodes the relative composition of 3E priorities. Circles indicate single-criterion priority and 3E-synergy (red) solution performance for environmental (purple), emission (green), economic (orange) objectives.

#### *2.5 Local policy action towards 3E-synergy*

**Table 1** The quantity of abandoned cropland available across US states and its areaweighted 3E characteristics, along with overall improvement from deploying PV based on the 3E-synergy prioritization strategy. \* represents whether abandoned cropland could fully satisfy the total sectoral electricity demand across residential, transportation, commercial, productive sectors.

| State      | Abandoned   | PV        | Environmental    | Emission   | Economic  | Energy   | 3E-synergy   |
|------------|-------------|-----------|------------------|------------|-----------|----------|--------------|
|            | land (k ha) | installed | suitability (per | mitigation | viability | demand   | and          |
|            |             | capacity  | ha)              | ability (t | (k        | (TWh)    | improvement  |
|            |             | (GW)      |                  | per/ha)    | USD/ha)   |          | (%)          |
| Texas      | 515.96      | 877.14    | 0.72             | 3689.42    | 995.83    | 572.86*  | 0.91(+49.75) |
| California | 272.49      | 463.23    | 0.74             | 4167.68    | 1663.92   | 529.72*  | 0.94(+36.39) |
| Georgia    | 244.59      | 415.80    | 0.99             | 3559.99    | 514.84    | 197.19*  | 0.90(+15.53) |
| Indiana    | 223.56      | 380.06    | 0.95             | 3086.24    | -440.81   | 131.56*  | 0.79(+4.23)  |
| New York   | 57.10       | 97.07     | 0.99             | 2973.04    | -371.57   | 255.29   | 0.80(+0.21)  |
| US         | 4703.36     | 7995.71   | 0.84             | 3446.23    | 72.21     | 5583.35* | 0.84(+38.36) |

To transform national strategy into local action, state-level decision matrices give mor flexible references for local governments assessing the reuse of abandoned land. Here, we illustrate with five states, namely California, Indiana, Texas, New York, and Georgia, which span diverse 3E profiles and PV capacity potentials (Table 1 and Extended Table 1).

The 3E-synergy solution yields an overall improvement of 38.4% relative to single-criteria decision, statistically significant across 48 states (*P* = 0.0004; 95% CI = 18.4-58.3%). States with larger performance gains exhibit imbalance among dimensions, where one shortfall can be compensated by the synergy effect. California is one such case. In contrast, states that already maintain either strong or weak coherence across all dimensions show limited improvement, as observed in Texas.

We applied a Mundlak decomposition within a hierarchical Bayesian framework to disentangle intra- and inter-state drivers of the 3E-synergy index. The results reveal scale dependent heterogeneity and within-between divergence. At the inter-state level, structural characteristics such as total economic scale show a positive but uncertain association with 3E-synergy, while population exerts a consistently negative effect,

reflecting intensified land use competition across states. At the intra-state level, higher 3E synergy is consistently associated with more remote abandoned croplands, characterized by greater distances to urban centers, power infrastructure, and road networks, indicating a clear reversal between state-scale structural conditions and local site suitability. Among all predictors, solar radiation emerges as the only scalerobust driver with strong positive effects at both levels, underscoring its dominant physical constraint on synergistic PV deployment.

# **3 Discussion**

The transformation of socio-ecological systems has generated extensive abandoned croplands, which hold mutually compatible potential for implementing land-based natural climate solutions (LNCS) and deploying renewable energy facilities. Considering that many US farmers are shifting from cultivating crops to cultivating electricity, our findings confirm that deploying PV on abandoned cropland represents a viable alternative pathway when evaluated in terms of environmental suitability, emission mitigation, and economic viability. However, this feasibility exhibits pronounced spatial heterogeneity, and single-criterion deployment priorities lead to significant trade-offs in policy performance. We propose the 3E-synergy index as a policy tool to promote co-benefits in PV decisions. The framework integrates machine learning with multi-criteria decision analysis, forming a complete process for assessment, computation, and diagnostic evaluation. Our approach combines the strengths of generative and discriminative models to couple environmental distribution learning with robust decision boundaries.

**Mismatch of historical priors and future policy objectives.** Historical priors derived from past PV deployment do not necessarily align with future climate and economic policy objectives. Although abandoned croplands generally exhibit high environmental suitability for PV deployment, our results reveal a substantial mismatch between expert-informed historical priors and forward-looking policy

deployment is guided solely by historically driven suitability, the expected economic performance becomes negative, with an average loss of -9.93 kUSD ha-1 and a potential cumulative loss of -206 billion USD (Fig. 4g). In contrast, under an economically optimal decision framework, cumulative net returns could reach up to 1462 billion USD. This outcome reflects the fact that historical PV deployment patterns primarily encode past policy priorities, whereas future policy objectives are scenario-dependent, dynamic, and subject to continual adjustment. These findings highlight the importance of explicitly incorporating policy objectives into PV prioritization frameworks rather than relying on historical deployment signals alone. To address limitations in policy-oriented modeling, particularly the lack of reliable negative samples in pilot applications, we developed a task-specific two-stage framework. Low-density regions within the environmental prior space were used to generate negative samples, allowing the model to learn more realistic exclusion patterns while accepting the loss of some potential opportunities. Our results further indicate that lightweight Transformer-ResNet architectures improve convergence for tabular data. Nevertheless, the framework does not distinguish among PV technologies with different environmental requirements and does not capture technological progress. Future studies should incorporate temporal dynamics and technology evolution to better support context-specific PV deployment strategies. **Human-driven climate solutions are more efficient but more uncertain than LNCS.** We systematically compared the emission mitigation potential and economic viability of photovoltaic deployment and land-based natural climate solutions. Under the mean scenario, PV exhibits an emission mitigation potential approximately sixteen times greater than that of LNCS. This estimate is lower than some extrapolations from empirical case studies, primarily because PV mitigation is achieved indirectly through electricity substitution. As fossil fuel shares decline under future energy transitions, the marginal mitigation benefit of PV is expected to

goals. This mismatch is most pronounced in the economic dimension. When PV

decrease. Accordingly, cumulative PV mitigation ranges from 87.78 Gt under stated energy policies to 29.87 Gt under net zero emissions scenarios (Supplementary Table 5). Despite this uncertainty, the results suggest that human-driven climate mitigation remains most efficiently addressed through human-developed technologies rather than relying solely on natural solutions.

Assuming full utilization of abandoned croplands, our analysis indicates that deploying PV on only about 10% of U.S. abandoned cropland could achieve a 30% solar share by 2050. Many states with large land endowments and agricultural legacies could further meet their total future electricity demand across all sectors using these lands. At the global scale, abandoned cropland, accounting for roughly 6.6% of total cropland, represents a unique opportunity to support climate neutrality. Nevertheless, this analysis focuses on CO₂ mitigation and does not capture the broader ecosystem services provided by LNCS, which may lead to an underestimation of their overall climate benefits.

Economic conditions play a central role in shaping both the pace and feasibility of climate policy implementation. By integrating multiple AR6 policy scenarios, we explicitly account for heterogeneity in the timing, ambition, and stringency of global climate action. Our results indicate that national-level policy stringency is the primary determinant of the economic break-even performance of PV deployment, whereas LNCS exhibit comparatively stable and robust long-term economic returns. This divergence highlights the complementary functions of technology-based mitigation and nature-based strategies within integrated climate policy portfolios. We acknowledge that economic estimates are derived primarily at the national scale. Although we partially address spatial heterogeneity by adjusting PV outcomes according to state-level projections of future electricity demand, finer-scale economic variation remains unresolved. Future research should incorporate subnational electricity pricing structures and grid policy instruments, such as net energy metering and regional tariff schemes, to more accurately characterize spatial heterogeneity in

PV economic performance.

**Synergy enables efficiency.** The policy-informed prioritization framework and the adaptive, dynamically weighted synergy index developed in this study both demonstrate strong potential for large-scale application. On the one hand, the policyinformed priority framework integrates macro-level policy preferences with quantilebased kernel functions, allowing decision-makers with heterogeneous objectives to select alternative spatial implementation pathways (Supplementary Note 4). By deriving an analytical solution, we construct a joint policy-performance convex surface, which reveals that output trade-offs among competing policy objectives are inherently nonlinear rather than proportional. This priority design not only clarifies how abstract policy preferences are operationalized within spatial ranking processes, but also establishes a unified benchmark for comparing cumulative outcomes across different policy strategies. Nevertheless, our analysis primarily emphasizes the efficiency of policy co-benefit generation. Considerations related to equity-oriented outcomes (tail-focused kernels) and robustness-oriented strategies (Gaussian kernels) are treated as complementary decision references rather than core optimization targets. In addition, heterogeneous state-level policy constraints are not explicitly modeled. Incorporating such institutional and regulatory heterogeneity through additional tuning factors represents an important direction for future research on policy-oriented spatial prioritization.

On the other hand, the proposed 3E-synergy index is designed to explicitly mediate trade-offs among competing objectives. Operating directly on pixel-level outcomes, the index compresses multidimensional trade-off information into a single diagnostic measure that highlights both the current level of co-benefits and the directions along which performance can be further enhanced (Extended Fig. X). By explicitly accounting for trade-off structures across environmental, emission, and economic dimensions, the index enables the identification of spatial configurations with strengthened synergistic gains. Taken together, the integration of policy-informed

prioritization and adaptive synergy assessment provides a general analytical framework for cross-regional and cross-scenario decision-making in energy deployment and land-use management.

To further investigate the drivers of such synergistic outcomes, we conducted multiscale Bayesian regression analyses. The results indicate that under decision uncertainty, solar radiation emerges as the only consistently robust driver, exhibiting stable explanatory power both within and between states. In contrast, the effects of other covariates display substantial instability, with 95% highest density intervals frequently overlapping zero. At the intra-state level, we observe scale-reversal effects for factors such as road accessibility and different classes of road network density. This pattern underscores the importance of differentiated state-level policy designs, rather than uniform national strategies. More broadly, these findings suggest that the redevelopment of abandoned cropland may involve wider trade-offs across alternative land-use pathways. As the dominant drivers of land-use change in the United States increasingly shift toward natural drivers<sup>40</sup>, it becomes necessary to incorporate a broader portfolio of potential redevelopment options into future spatial decisionmaking frameworks.

## **Methods**

#### *Two-stage modeling of environmental suitability for PV deploying*

We developed a two-stage machine learning framework integrating a Gaussian Mixture Model (GMM) and a Transformer–ResNet classifier to generate the probability surface of PV deployment on abandoned cropland. The modeling pipeline consisted of three modules: (1) feature engineering, which coordinates data labeling and harmonizes multi-dimensional feature embeddings. (2) Environmental similarity measurement, where a GMM captures the joint distribution of PV site environmental characteristics and reconstructs a negative-sample pool from low probability density regions. (3) Probability prediction, where a Transformer–ResNet classifier learns the

decision boundaries of PV deployment.

**Feature Engineering of abandoned cropland and PV farm.** We first identified abandoned cropland samples in the US for 2020 using the globally consistent, longterm ESA-CCI land cover dataset (1992–2022). Following the FAO definition, cropland abandonment refers to cropland that is no longer actively cultivated and has transitioned to other ecosystem types without direct human intervention for at least five consecutive years. We standardized the moving window detection for reducing false detections due to crop rotations and temporal resolution limitations: (1) We aggregated the original 37 land cover classes into 9 major classes including 5 ecosystem types (forest, wetlands, grassland, arid ecosystem, and shrublands), 1 cropland class (including all six cropland subclasses), 2 non-vegetation areas (water bodies, bare areas, and permanent ice), and 1 human settlements area. (2) A 5-year moving window was used for cropland abandonment detection, with areas converted to human settlements and reclaimed cropland (pixels that were once detected as abandoned but reverted to cropland for 2 successive years) excluded from the analysis.

As for PV labels, we combined two complementary data sources to strengthen the ground-truth foundation of our spatial modeling. One dataset was derived from harmonized global solar farm dataset acquired from OSM<sup>41</sup> and another is a global PV inventory generated by convolutional neural network models<sup>2</sup> . All labels and features were standardized to a 1/120° (~1 km at equator) resolution, and more detailed data sources are available in the Supplementary Materials. Our model incorporated 15 features including physical geographic environment (land cover, digital elevation, slope, gross dry matter productivity), climatic factors (near-surface air temperature, wind speed, shortwave radiation), and socio-economic factors (population density<sup>42</sup> , gridded and subnational level of GDP per capita<sup>43</sup> , distance to human settlements<sup>44</sup> and power system<sup>45</sup> , and road density in tertiary, primary and secondary level<sup>46</sup>). A unified preprocessing pipeline was implemented for feature

standardization, and categorical land cover data were further transformed through one-hot encoding to convert discrete types into numerical features suitable for model computation.

**First stage of GMM and negative label generation.** In the first stage, the GMM was employed to quantify environmental similarity between abandoned cropland and existing PV sites based on their joint statistical distributions. Compared with nonparametric kernel density estimation (KDE) and other discriminative models, which suffers from the curse of dimensionality struggle to provide the comprehensive environmental information, GMM provides a more robust and interpretable density estimation by decomposing the feature space into a finite number of Gaussian components. This decomposition establishes a bidirectional mapping between the environmental distribution and the PV site feature space, enabling probabilistic inference across heterogeneous landscapes. To ensure numerical stability during model training, we performed a grid search with five-fold cross-validation to optimize hyperparameters, including the number of mixture components, covariance type, regularization coefficient, and initialization counts. The Bayesian Information Criterion (BIC) was used to balance model complexity and maximize log-likelihood, thereby improving generalization capacity.

Negative samples were drawn from the 5th percentile of the GMM log-probability distribution, corresponding to low-density regions in the environmental feature space. These samples were reconstructed into the original feature domain through nearestneighbor mapping with small Gaussian perturbations, ensuring an equal number of positive and negative samples for the subsequent discriminative model training. **Second stage of Transformer-Resnet training.** The second stage translates the learned environmental similarity into probabilistic decision boundaries for PV deployment. We developed a Transformer–ResNet architecture to capture both global feature interactions and local residual variations within the feature space. The Transformer component encodes long-range dependencies and contextual

relationships among environmental features, while the residual connections in ResNet facilitate gradient propagation and improve convergence stability. The model was trained using a binary cross-entropy loss function and optimized with the Adam algorithm to accelerate gradient convergence. The Transformer encoder was composed of two attention heads and residual feed-forward layers, followed by dense fusion layers to integrate multi-scale representations.

In model diagnostics, the hybrid architecture demonstrated superior predictive performance on the test dataset (Accuracy = 0.89, Precision = 0.88, Recall = 0.92, F1 = 0.90). Learning curve analysis confirmed strong generalization ability, with less than 2 % difference between training and validation performance. For benchmarking, a baseline Multilayer Perceptron (MLP) model was trained for comparison. Detailed hyperparameter configurations and training protocols are provided in the Supplementary Materials.

#### *Assessment of net expected carbon abatement*

For PV mitigation estimation, we utilized high resolution climatological datasets from CHELSA<sup>47</sup> to correct the nominal module power capacity under standard test conditions and to calculate actual PV power generation over a 30-year lifecycle. Carbon substitution effects were quantified using emission factors from the IFI Dataset, further adjusted to 2050 according to the IEA projections of power-sector carbon intensity. For LNCS opportunity costs, spatial aggregation and inversedistance weighting were applied to land-cover change observations to generate probability vectors for each abandoned cropland site. These probability vectors incorporate neighborhood land-use preferences, representing the likelihood of conversion to agriculture, afforestation, or vegetation restoration. The LNCS emission reduction potential was derived by multiplying these probability vectors by the carbon sequestration potential of each strategy, as estimated from ensembled carbon pool datasets.

**Calculating PV carbon mitigation potential.** The carbon emissions mitigated by PV power were calculated by replacing grid power using the baseline emission factors of the national grid. PV power generation is determined by PV power generation potential () and the installed capacity (). The , which describes the performance of PV cells constrast to the nominal power capacity, depends on primarily on actual environmental conditions. Following previous studies<sup>48</sup> , can be calculated as follows:

$$PV_{POT} = P_R \cdot \frac{I}{I_{STC}}$$
 
$$P_R = 1 + \gamma \cdot (T_{cell} - T_{STC})$$

where represents surface downwelling shortwave radiation and is the performance correlation ratio determined by temperature. The standard state referred to in this paper includes the shortwave flux on the PV cell (: 1000 −2 ), the temperature of PV panel (: 25 ℃) , the temperature coefficient (: -0.005 ℃−1 ). represent the temperature of PV cell and can be approximated as follows:

$$T_{cell} = a_1 + a_2 \cdot T + a_3 \cdot I + a_4 \cdot W$$

where 1, 2, <sup>3</sup> and <sup>4</sup> were taken as 4.3 ℃, 0.943 (unitless), 0.028 ℃ ( −2 ) −1 and -1.528 ℃ ( −1 ) −1 , respectively. Those parameters are proven to be independent on location and cell technology, and have been widely used for estimating PV performance. To convert PV electricity generation into mitigation benefits, we followed the Harimonize Approach to Greenhouse Gas Accounting in accordance with the United Nations Framework Convention on Climate Change (UNFCCC)<sup>49</sup>. This method reflects the carbon dioxide emission intensity of the marginal electricity grid displaced by renewable generation, typically approximated using the Combined Margin (CM). The CM for the grid is comprised of an Operating Margin (OM) and a Build Margin (BM). In principle, the OM consists of existing power plants that would be affected and have the highest variable operating cost. The BM represents the cohort of the future power plants whose construction could be affected by PV project, based on an average of future emission intensities of new

electricity generation.

$$PV_{carbon} = PV_{POT} \cdot PV_{Installed} \cdot EF_{CM} \cdot \eta_{sys} \cdot Y \cdot CF$$
 $EF_{CM} = EF_{OM} \cdot W_{OM} + EF_{BM} \cdot W_{BM}$ 
 $PV_{Installed} = A_{panel} \cdot \eta_{stc} \cdot PF$ 

where is the carbon abatement. *EFOM* is the OM factor and is BM factor. is module rated power density which was set as 0.17 −2 . denotes as system-level energy conversion factor typically ranging from 0.7-0.9, and we took 0.8 in this study<sup>50</sup> . <sup>Y</sup> denotes the annual operating hours of PV default set to 8760 *h*, and is the conversion factor ~0.27 (molar mass C/ molar mass CO2). Considering the anticipated reduction in power carbon intensity by 2050, the PV carbon mitigation factor was dynamically calibrated using IEA projections to enhance the robustness of the mitigation assessment<sup>51</sup> . *PF* can be used to measure the fraction of land utilization rate of PV, ranging from 0.7-1.0 according to empirical research<sup>52</sup> . **Generating probability map of LNCS.** Abandoned cropland admits multiple postabandonment conversion pathways, so the likelihood of implementing LNCS is inherently spatially heterogeneous across sites. To explicitly estimate the variation in opportunities under policy interventions, we introduced statistical expectation formulation and allocated LNCS probability for each site, grounded in measure theory and historical land-cover trajectories. First, for each abandoned site, we identified the land cover type with the highest frequency of occurrence during its duration and assigned it as the most preferred land-use state. Second, we aggregated pixels into larger regional units using 10 × 10 grids, and constructed density vectors for each unit according to the proportional counts of different land-use types, which aligned with canonical LNCS plants of forest, cultivated crops, and non-woody vegetation. Third, K-d tree was constructed for searching proximate abandoned cropland, and inverse distance weighting (IDW) was applied to assign reuse probability across forest, agriculture, and grassland land sectors.

$$\mathbb{E}_{i}[C] = PV_{carbon} - \sum_{k \in \{F,A,N\}} p_{i,k} \cdot C_{i,k}, \qquad k \in \{F,A,N\}$$

where [] denotes the net expected carbon abatement effect. , and , are the probability and the cumulative carbon sink potential of abandoned site assigned to strategy from LNCS (i.e. afforestation, agriculture, and non-woody vegetation). **Calculating carbon sink potential across LNCS.** We integrated several globally high-resolution datasets of aboveground biomass (AGB), belowground biomass (BGB), and soil organic carbon (SOC) into expectation calculation framework for carbon storage. Considering the biological limitation of natural carbon sequestration, the net accumulation of carbon storage was assessed over a 30-year horizon, capped by the maximum potential carbon storage of Walker et al. <sup>53</sup> under current biogeophysical conditions and the absence of human disturbance. For estimating growth across different carbon pools, we used the gridded annual carbon sequestration rates<sup>54</sup> of natural forest regrowth as a multiplier for AGB, further combined with the root-mass fraction (RMF) map from Ma et al.<sup>55</sup>, the SOC potential maps under future scenarios from FAO<sup>56</sup>, and the dead wood and litter carbon indicator (DWCI) to finetune carbon storage across LNCS.

$$C_{i,k}(t) = min\left(C_{i,k,0} + \sum_{\tau=1}^{t} \Delta C_{i,k,\tau}, C_{i,k,max}\right)$$

where ,() denotes the cumulative carbon storage of abandoned site under LNCS of ; ,,0 is the current carbon stock; Δ,, represents the annual net carbon increment during period; and ,, is the biophysical potential derived from existing estimates.

(1) **Afforestation.** The afforestation accounts for four major carbon pools. For each abandoned site, we estimated total biomass stocks growth using the natural forest AGB accumulation map combined with the RMF map to derive BGB. The potential SOC density in the upper 2 m of soil was defined as the historical stock under no land-use conversion, and the unrealized SOC potential was used to represent the

incremental contribution of afforestation over the 30 years. The dead wood and litter pools represent undecomposed carbon leaving on the ground surface. This pool was calculated as fraction of AGB and the ratio was provided by Harris et al<sup>57</sup> .

- (2) **Agriculture.** We selected the suitability data of perennial and annual crops from Global Agro-Ecological Zoning database (GAEZ-V4)<sup>58</sup>. Following the IPCC guidelines for land-use practices, we assumed that abandoned sites highly suitable for perennial woody crops could only accumulate aboveground biomass for a finite period or ultimately reach a steady state, due to periodic pruning, harvesting, and other removals. By contrast, abandoned sites assigned to annual crop were assumed to exhibit no net biomass accumulation, since yearly gains in biomass stocks are offset by harvest and mortality within the same year. Thereby the principal variation in carbon storage occurs in SOC as a function of agricultural management. Taking historical SOC as the baseline from Walker et al., we calculated net changes by applying the SOC density variation rates under FAO's agricultural management with business as usual.
- (3) **Non-woody vegetation.** We considered savanna, shrub, grassland, and other nonforest land types as the main targets for implementing non-woody restoration. Carbon pools of non-woody plants are characterized by extensive root systems, with biomass largely concentrated in BGB, which primarily drives carbon input to soils<sup>59</sup> . Accordingly, shrub- and grass-specific root-mass fractions (RMFs) were applied to estimate BGB, while unrealized SOC potential was included to capture changes in soil carbon stocks under this pathway. DWCI was treated consistently with the above pathways.

#### *Cost-benefit analysis across climate policy scenarios*

Upon the LNCS probability framework, we extended the expected value analysis to incorporate cost-benefit of PV deployment and computed opportunity of LNCS. For PV economic benefit, the data on electricity generation revenues and costs were obtained from IPCC AR6 Scenarios Database<sup>60</sup> , which is recognized as an 'ensemble of opportunity' providing transparency, pluralism, scientific credibility, and policy relevance scheme.

**Calculating carbon sink potential across LNCS.** We selected 11 policy scenarios with 3 tiers (P1, P2, P3) from IPCC AR6 assessment to capture the overall mean trend of future economic viability of PV under diverse model assumptions and constraints<sup>61</sup> . The policy categories in AR6 were identified using text pattern matching on the scenario metadata and calibrated on the best-known scenarios from model intercomparisons, with further validation against the related literature, reported trajectories, and exchanges with modellers. The policy scenarios give insights into how societal choices may steer the system, and P1 tier represents no globally coordinated policy; P2 are globally coordinated climate policies with immediate action, P3 are globally coordinated climate policies with delayed. Additionally, P4 are cost-benefit scenarios and were not analyzed separately but incorporated into the mean trend due to insufficient data.

$$\mathbb{E}_{i}^{(s,y)}[\text{NPV}] = \text{NPV}_{i}^{\text{PV}}(s,y) - \sum_{k \in \{F,A,N\}} p_{i,k}^{(s)} \text{NPV}_{i}^{k}(s,y)$$

where (,) [NPV] denotes the expected NPV of abandoned site under policy scenarios in target year. NPV PV(, ) is the NPV of PV deploying at site , defined as the difference between cumulative revenues and cumulative costs. And , () NPV (, ) denotes the weighted NPV of LNCS.

Each abandoned site considered for PV deployment encompassed 265 alternative pathways, and three major datasets including one-time investment costs, operating costs incurred over the PV life cycle, and the price of secondary energy were extracted at 10-year intervals from 2020 to 2050 to calculate the net present value (NPV). All prices were standardized to constant 2020 US dollars and a discount rate of 5% was applied. We adopted a unified but simplified set of assumptions combined with a relatively diversified range of pathways to capture the average profitability of PV under different climate policy scenarios<sup>62</sup> .

As for the opportunity cost of LNCS, we quantified each NPV separately. For agricultural sector, we acquired domestic price projections to 2050 for 184 crop types from FAO<sup>63</sup> and aligned crop categories and Representative Concentration Pathways (RCPs) with the gridded potential yield data from GAEZ-V4<sup>58</sup>. According to ref.<sup>64</sup> , the profitability of individual farms can vary substantially, we simply assumed a 20% marginal profit based on US farm finances<sup>65</sup> and set an uniform initial investment mainly accounting for irrigation investment and land value apraisal<sup>66</sup> . For afforestation, we calculated the NPV by assuming a one-time harvest of the potential AGB in 2050 and its conversion into wood products, applying fractions of biomass remaining in each product type and corresponding profit ratios. Biomass was allocated across three end uses (i.e., bioenergy, paper and pulp, and solid wood), and we applied mean implementation costs of planting native and exotic trees<sup>23</sup> . Lastly, for non-woody vegetation, given that the global pasture market is much smaller than that of agricultural and woody products (e.g., accounting for only ~0.5% of global woody trade), we did not assume direct economic returns. Instead, we approximated its feasibility primarily using the implementation cost of natural regeneration.

#### *Policy-informed priority and 3E-synergistic PV deployment*

We developed a policy-informed priority framework that combines macro decision preferences with quantile attention. This allows 3E objectives to jointly shape the ordering of potential PV sites. For achieving co-benefits, we introduced an adaptive 3E-synergy index, which could capture the present co-benefits level and the directions in which they can be strengthened. Finally, for translating into actionable policy across all US states, we then compared synergy informed strategies with single criterion baselines at both national and state scales by using hypothesis testing and Bayesian regression analysis.

**Policy-informed priority and performance evaluation.** Under different policy preferences and priority order, the ultimate outcomes remain identical, yet the efficiency of generating benefits can vary substantially across space and time. In this study, policy preferences are encoded by a macro preference vector, and the quantile attention is represented by prior kernel functions. The policy-informed prioritization problem is then formulated as a joint multi-objective optimization:

$$\max_{\pi} [E_1(\pi), \cdots E_k(\pi)]$$

$$E_k(\pi) = \int_0^1 g_k(\pi(u)) w(u) du$$

() represents the quantile-weighted efficiency of generating cumulative benefits for objective under a given prioritization sequence . (()) denotes the marginal gain of objective at quantile . The kernel () is the quantile attention which allows different policy logics like decreasing kernels emphasize early gains (efficiency-oriented strategies), increasing kernels emphasize tail outcomes (equity-oriented strategies), and Gaussian kernels emphasize robustness around intermediate quantiles. This formulation generalizes conventional single-criterion or static weighted prioritization by explicitly modeling how policy preferences shape benefit generation along the implementation sequence rather than only the final outcomes. Detailed kernel specifications and the numerical solution of priority sequences are provided in Supplementary Note 4.

**Policy performance evaluation.** Under different priority solutions, the ultimate outcomes remain identical, yet the efficiency of generating benefits can vary substantially across space and time. To verify the feasibility of our solution in achieving co-benefits, we computed the mean cumulative benefit by integrating the gain curve over the full quantile range as the following equation:

$$\mathbb{E}[f(U)] = \int_0^1 f(u)du, \qquad U \sim \text{Uniform}(0,1)$$

where [()] denotes the expected outcome of given priority and () is the accumulative gain under the ranked PV priority deloyment (i.e. prioritize 3E-synergy and single-criteria of environment suitability, emission abatement ability and economic viability). Higher values indicates that under a uniform observation of implementation stages, a given prioritization scheme achieves superior performance

efficiency.

**3E-synergistic index calculation.** Using weighted coupling coordination degree (WCCD) as a proxy for measuring synergy level of each abandoned sites, we performed a sequential least squares programming (SLSQP) algorithm to search for PV deployment priority with greater policy feasibility.

$$T = \sum_{i=1}^{3} w_i U_i, \qquad C = \left(\frac{\prod_{i=1}^{3} U_i}{\left(\frac{1}{3} \sum_{i=1}^{3} U_i\right)^3}\right)^{\frac{1}{3}}$$

$$3E - synergy \ index = max \ CCD = \sqrt{C_i(U_i) \cdot T_i(w_i, U_i)}$$

$$subject \ to \ \sum_{i=1}^{3} w_i = 1, w_i \ge 0$$

where the coupling degree captures the equilibrium among the subsystems, while the coordination degree reflects their integrated performance under different decision preferences and is initially defined as the arithmetic mean of the normalized objectives . In this study, we maximize ( ,) to identify the potential synergistic deployment scheme (3E-synergy).

**Statistical inference for policy improvements.** To evaluate whether synergistic prioritization yields significant improvements, we calculated the percentage of policy improvements followed by:

$$I_{s,d,k} = \frac{WCCD_{s,d} - S_{s,d,k}}{\left|S_{s,d,k}\right|}$$

,, quantifies how much the synergistic solution improves the policy efficiency under dimension () of strategy () in state (), relative to the best-performing under single-objective strategy ,,.

**Mundlak decomposition and hierarchical Bayesian regression.** We further identified the drivers of 3E-synergy across national and subnational scales using a Mundlak decomposition hierarchical Bayesian regression framework. This approach decomposes each predictor into within-region (intra-state) and between-region (interstate) components, allowing us to distinguish local variation within states from

structural differences across states. First, Box–Cox transformations were applied to environmental and socio-economic predictors to improve distributional properties and estimation stability. Second, we implemented the Mundlak decomposition to simultaneously capture cross-state heterogeneity and within-state local variability, supporting spatial policy design across both national and local governance contexts. Finally, Bayesian regularization and hierarchical shrinkage priors were employed to mitigate potential singularity, multicollinearity, and coefficient instability that may arise in conventional OLS estimation.

$$\begin{aligned} x_{i,s} &= \left(x_{i,s} - \bar{x}_s\right) + \bar{x}_s \\ 3E - synery \ index_{i,s} &\sim Beta(\mu_{i,s}, \phi) \\ logit(\mu_{i,s}) &= \alpha_s + \beta_{within} \cdot x_{i,s}^{within} + \beta_{between} \cdot x_s^{between} \end{aligned}$$

 $x_{i,s} - \bar{x}_s$  represents the within-state deviation, capturing how the environmental or socioeconomic characteristic at pixel i differs from the state-level mean  $\bar{x}_s$ .  $3E - synery index_{i,s}$  denotes the 3E performance combination at pixel i in state s, constrained to the interval (0,1) and therefore modeled using a Beta likelihood.  $\beta_{within}$  describes how local deviations from the state-average value affect 3E-synergy. In contrast,  $\beta_{between}$  describes how differences in state-average levels shift overall synergy, reflecting structural or institutional advantages between states.  $\alpha_s$  is state-level varying intercept, captures persistent state-level institutional, climatic, and structural differences.

## **Extended Data Figure**

![](paper_b40afd_images/_page_33_Figure_1.jpeg)

**Extended Data Fig.1** Policy-informed priority framework for 3E-synergy. The framework comprises two modules of policy objective assessment and policy priorityzone decision-making. In the objective assessment module, the environmental suitability is based on expert knowledge embeddings, while PV mitigation ability and evaluate economic viability are assessed over 2020–2050 accounting for the opportunity costs of LNCS. In the decision-making module, macro-level policy preferences are integrated with quantile attention to generate policy performance frontiers. The 3E-synergy solution provides pixel-level trade-off signals and delineates a robust decision domain across the three objectives.

![](paper_b40afd_images/_page_34_Figure_0.jpeg)

**Extended Data Fig.2** Policy performance frontiers of policy-informed priorities under different quantile attention kernels and macro-preference combinations. Circled hyperplanes denote policy-performance benchmarks emphasizing investment efficiency (red, decreasing kernel), robust returns (purple, gaussian kernel), and regional equity (green, increasing kernel), respectively. Cumulative curve shows

![](paper_b40afd_images/_page_35_Figure_0.jpeg)

**Extended Data Fig.3** Cumulative 3E benefits across US states and target states.

**Extended Table 1** Abandoned land 3E patterns and policy guidance for deploying PV across US states.

| State       | Abandone  | PV        | Environme   | Emission   | Economic     | Energy  | 3E        |
|-------------|-----------|-----------|-------------|------------|--------------|---------|-----------|
|             | d land (k | installed | ntal        | mitigation | viability (k | demand  | synergy   |
|             | ha)       | capacity  | suitability | ability (t | USD/ha)      | (TWh)   | and       |
|             |           | (GW)      | (per ha)    | per/ha)    |              |         | improveme |
|             |           |           |             |            |              |         | nt (%)    |
| Texas       | 515.96    | 877.14    | 0.72        | 3689.42    | 995.83       | 572.86* | 0.91      |
|             |           |           |             |            |              |         | (+49.75)  |
| Illinois    | 295.4     |           |             | 3162.07    | -42.82       | 240.87* | 0.82      |
|             |           | 502.17    | 0.93        |            |              |         | (+8.78)   |
| California  | 272.49    | 463.23    | 0.74        | 4167.68    |              | 529.72* | 0.94      |
|             |           |           |             |            | 1663.92      |         | (+36.39)  |
| Michigan    | 246.01    | 418.22    |             | 2937.64    | -441.21      |         | 0.79      |
|             |           |           | 0.98        |            |              | 182.43* | (+10.19)  |
| Georgia     | 244.59    | 415.8     | 0.99        |            | 514.84       | 197.19* | 0.90      |
|             |           |           |             | 3559.99    |              |         | (+15.53)  |
| Indiana     | 223.56    | 380.06    | 0.95        |            | -440.81      | 131.56* | 0.79      |
|             |           |           |             | 3086.24    |              |         | (+4.23)   |
| Wisconsin   |           |           |             |            |              |         | 0.77      |
|             | 209.3     | 355.8     | 0.95        | 2999.38    | -620.92      | 120.01* | (+3.77)   |
|             |           |           |             |            |              |         | 0.71      |
| Montana     | 171.38    | 291.34    | 0.42        | 3479.28    | -755.14      | 21.01*  | (+1.75)   |
| North       |           |           |             |            |              | 204.88* | 0.87      |
| Carolina    | 167.7     | 285.09    | 0.99        | 3401.52    | 351.57       |         | (+8.06)   |
|             | 156.2     |           | 0.93        | 3057.05    |              |         | 0.80      |
| Ohio        |           | 265.54    |             |            | -151.05      | 221.58* | (+1.87)   |
|             | 152.66    | 259.51    | 0.86        | 3201.49    |              | 354.77* | 0.88      |
| Florida     |           |           |             |            | 652.38       |         | (+19.96)  |
|             |           | 213.95    | 0.98        | 3394.89    | -140.22      | 98.95*  | 0.85      |
| Louisiana   | 125.85    |           |             |            |              |         | (+4.32)   |
|             |           |           |             |            |              |         | 0.93      |
| Colorado    | 110.67    | 188.14    | 0.7         | 4191.62    | 700.25       | 81.13*  | (+50.96)  |
|             |           |           |             |            |              |         | 0.81      |
| Idaho       | 108.03    | 183.66    | 0.39        | 3831.16    | -418.9       | 27.40*  | (+31.72)  |
| Utah        |           |           |             |            |              |         | 0.89      |
|             | 107.85    | 183.35    | 0.51        | 4172.86    | 331.5        | 49.87*  | (+67.58)  |
|             |           | 176.94    | 0.96        | 3463.53    | -398.16      | 61.37*  | 0.83      |
| Mississippi | 104.08    |           |             |            |              |         | (+3.09)   |
|             |           |           |             |            |              |         | 0.79      |
| Iowa        | 98.48     | 167.42    | 0.88        | 3260.94    | -554.3       | 64.71*  | (+8.94)   |
| Minnesota   | 97.81     | 166.28    | 0.93        | 3049.68    | -544.44      | 109.52* | 0.77      |

|           |       |        |      |         |         |         | (+3.06)     |
|-----------|-------|--------|------|---------|---------|---------|-------------|
| Kansas    | 93.06 | 158.2  | 0.8  | 3797.38 | -16.79  | 60.41*  | 0.88        |
|           |       |        |      |         |         |         | (+60.37)    |
| Missouri  | 92.86 | 157.86 | 0.91 | 3320.52 | -155.16 | 127.91* | 0.82        |
|           |       |        |      |         |         |         | (+4.70)     |
| Alabama   | 91.07 | 154.82 | 0.98 | 3537.19 | 43.86   | 107.95* | 0.86        |
|           |       |        |      |         |         |         | (+8.38)     |
| Washingto | 87.01 | 147.91 | 0.76 | 3190.27 | -470.05 | 100.35* | 0.75 (<br>- |
| n         |       |        |      |         |         |         | 9.06)       |
| Pennsylva | 84.48 | 143.61 | 0.98 | 3125.15 | -242.45 | 179.46  | 0.82        |
| nia       |       |        |      |         |         |         | (+5.69)     |
| South     |       |        |      |         |         |         | 0.86        |
| Carolina  | 76.43 | 129.94 | 0.99 | 3408.92 | -117.11 | 98.10*  | (+4.58)     |
|           |       |        |      |         |         |         | 0.83        |
| Arkansas  | 66.47 | 113    | 0.95 | 3454.82 | -399.12 | 65.31*  | (+3.27)     |
|           |       |        |      |         |         |         | 0.80        |
| Nebraska  | 63.92 | 108.67 | 0.73 | 3573.9  | -497.99 | 37.90*  | (+14.09)    |
| North     |       |        |      |         |         |         | 0.73        |
| Dakota    | 58.34 | 99.17  | 0.88 | 3196.86 | -970.68 | 16.56*  | (+36.89)    |
|           | 58.28 | 99.07  | 0.93 | 3412.27 | 2.81    | 138.42  | 0.84        |
| Tennessee |       |        |      |         |         |         | (+4.45)     |
|           |       | 97.07  | 0.99 | 2973.04 | -371.57 | 255.29  | 0.80        |
| New York  | 57.1  |        |      |         |         |         | (+0.21)     |
|           |       |        |      |         |         |         | 0.82        |
| Kentucky  | 55.72 | 94.72  | 0.95 | 3277.73 | -350.37 | 99.29*  | (+6.06)     |
|           |       |        |      |         |         |         | 0.76 (<br>- |
| Oregon    | 54.19 | 92.12  | 0.77 | 3255.7  | -607.02 | 62.36*  | 8.18)       |
|           |       |        |      |         |         |         | 0.85        |
| Virginia  | 51.82 | 88.09  | 0.95 | 3363.66 | 135.41  | 169.3   | (+8.14)     |
|           |       |        |      |         |         |         | 0.83        |
| Maryland  | 43.88 | 74.6   | 0.94 | 3299.65 | -203.41 | 117.36  | (+4.49)     |
|           |       |        |      |         |         |         | 0.88        |
| Oklahoma  | 42.46 | 72.19  | 0.7  | 3849.78 | 286.92  | 89.52*  |             |
|           |       |        |      |         |         |         | (+47.71)    |
| South     | 40.68 | 69.16  | 0.84 | 3415.54 | -790.88 | 17.54*  | 0.77        |
| Dakota    |       |        |      |         |         |         | (+17.41)    |
| Wyoming   | 32.95 | 56.01  | 0.4  | 3920.32 | -336.02 | 14.50*  | 0.83        |
|           |       |        |      |         |         |         | (+33.58)    |
| Arizona   | 31.18 | 53     | 0.84 | 4478.35 | 1158.88 | 98.4    | 0.97        |
|           |       |        |      |         |         |         | (+47.38)    |
| Delaware  | 28.46 | 48.38  | 0.96 | 3323.57 | -811.14 | 18.48*  | 0.79        |
|           |       |        |      |         |         |         | (+28.60)    |
| New       | 27.39 | 46.57  | 0.92 | 3205.13 | -357.23 | 119.29  | 0.80        |

| Jersey     |         |              |         |                    |          |           | (+2.54)   |
|------------|---------|--------------|---------|--------------------|----------|-----------|-----------|
| New        |         |              |         |                    |          |           | 0.96      |
| Mexico     | 22.62   | 38.46        | 0.78    | 4537.33            | 349.23   | 34.70*    | (+66.08)  |
|            |         |              | 0.33    | 4184.84            | 201.62   | 41.68     | 0.87      |
| Nevada     | 14.5    | 24.64        |         |                    |          |           | (+105.59) |
|            | 6.55    | 11.14        | 0.99    | 2908.94            | -1266.17 | 21.07     | 0.74      |
| Maine      |         |              |         |                    |          |           | (+267.30) |
| West       |         |              |         |                    | -871     | 37.84     | 0.78      |
| Virginia   | 5.53    | 9.4          | 0.98    | 3220.49            |          |           | (+32.69)  |
| Massachus  |         |              | 0.99    | 3103.71            | -612.44  | 91.62     | 0.80      |
| etts       | 3.88    | 6.6          |         |                    |          |           | (+8.20)   |
|            |         |              |         |                    |          |           | 0.74      |
| Vermont    | 2.26    | 3.84         | 1       | 2893.16            | -1252.7  | 10.33     | (+367.32) |
| New        |         |              |         | 2933.13<br>-1178.4 |          |           | 0.67      |
| Hampshire  | 1.48    | 2.52         | 0.83    |                    | 19.7     | (+211.79) |           |
| Connecticu | 0.64    | 1.09<br>0.98 |         |                    |          |           | 0.78      |
| t          |         |              | 3108.38 | -879               | 49.59    | (+25.26)  |           |
| Rhode      | 0.13    | 0.22         | 1       | 3152.27            | -1037.22 | 13.29     | 0.78      |
| Island     |         |              |         |                    |          |           | (+105.58) |
|            |         |              |         |                    |          |           | 0.84      |
| Overall    | 4703.36 | 7995.71      | 0.84    | 3446.23            | 72.21    | 5583.35   | (+38.36)  |

## **Reference**

- 1. Gidden, M. J. *et al.* Aligning climate scenarios to emissions inventories shifts global benchmarks. *Nature* **624**, 102–108 (2023).
- 2. Kruitwagen, L. *et al.* A global inventory of photovoltaic solar energy generating units. *Nature* **598**, 604–610 (2021).
- 3. Lamboll, R. D. *et al.* Assessing the size and uncertainty of remaining carbon budgets. *Nat. Clim. Change* **13**, 1360–1367 (2023).
- 4. Pourasl, H. H., Barenji, R. V. & Khojastehnezhad, V. M. Solar energy status in the world: A comprehensive review. *Energy Rep.* **10**, 3474–3493 (2023).
- 5. International Renewable Energy Agency. *Future of Solar Photovoltaic-Deployment Investment Technology Grid Integration and Socio-Economic Aspects*. (International Renewable Energy Agency IRENA, 2019).
- 6. Wei, S., Chen, Y. & Zeng, Z. An unexpectedly large proportion of photovoltaic facilities installed on cropland. *Innov. Energy* **2**, 100070 (2025).
- 7. Sturchio, M. A., Gallaher, A. & Grodsky, S. M. Ecologically informed solar enables a sustainable energy transition in US croplands. *Proc. Natl. Acad. Sci.* **122**, e2501605122 (2025).
- 8. Wu, W. *et al.* Assessment of the ecological and environmental effects of large-scale photovoltaic development in desert areas. *Sci. Rep.* **14**, 22456 (2024).
- 9. Wang, K. *et al.* Deploying photovoltaic systems in global open-pit mines for a

- clean energy transition. *Nat. Sustain.* https://doi.org/10.1038/s41893-025-01594-w (2025) doi:10.1038/s41893-025-01594-w.
- 10. Zhang, Z. *et al.* Worldwide rooftop photovoltaic electricity generation may mitigate global warming. *Nat. Clim. Change* **15**, 393–402 (2025).
- 11. Chen, S. *et al.* Deploying solar photovoltaic energy first in carbon-intensive regions brings gigatons more carbon mitigations to 2060. *Commun. Earth Environ.* **4**, 369 (2023).
- 12. Maka, A. O. M., Ghalut, T. & Elsaye, E. The pathway towards decarbonisation and net-zero emissions by 2050: The role of solar energy technology. *Green Technol. Sustain.* **2**, 100107 (2024).
- 13. International Renewable Energy Agency. *Global Renewables Outlook-Energy Transformation 2050*. (International Renewable Energy Agency IRENA, 2020).
- 14. Adeh, E. H., Good, S. P., Calaf, M. & Higgins, C. W. Solar PV Power Potential is Greatest Over Croplands. *Sci. Rep.* **9**, 11442 (2019).
- 15. Lei, Y. *et al.* Co-benefits of carbon neutrality in enhancing and stabilizing solar and wind energy. *Nat. Clim. Change* **13**, 693–700 (2023).
- 16. Daskalova, G. N. & Kamp, J. Abandoning land transforms biodiversity. *Science* **380**, 581–583 (2023).
- 17. Crawford, C. L., Yin, H., Radeloff, V. C. & Wilcove, D. S. Rural land abandonment is too ephemeral to provide major benefits for biodiversity and climate. *Sci. Adv.* **8**, eabm8999 (2022).

- 18. Prishchepov, A. V. *et al.* The progress and potential directions in the remote sensing of farmland abandonment. *Remote Sens. Environ.* **331**, 115019 (2025).
- 19. Crawford, C. L., Wiebe, R. A., Yin, H., Radeloff, V. C. & Wilcove, D. S. Biodiversity consequences of cropland abandonment. *Nat. Sustain.* **7**, 1596–1607 (2024).
- 20. Zheng, Q. *et al.* The neglected role of abandoned cropland in supporting both food security and climate change mitigation. *Nat. Commun.* **14**, 6083 (2023).
- 21. Arora, V. K. & Montenegro, A. Small temperature benefits provided by realistic afforestation efforts. *Nat. Geosci.* **4**, 514–518 (2011).
- 22. Stern, R. *et al.* Photovoltaic fields largely outperform afforestation efficiency in global climate change mitigation strategies. *PNAS Nexus* **2**, pgad352 (2023).
- 23. Busch, J. *et al.* Cost-effectiveness of natural forest regeneration and plantations for climate mitigation. *Nat. Clim. Change* **14**, 996–1002 (2024).
- 24. Saxena, A., Brown, C., Arneth, A. & Rounsevell, M. Advanced photovoltaic technology can reduce land requirements and climate impact on energy generation. *Commun. Earth Environ.* **5**, 586 (2024).
- 25. Stid, J. T. *et al.* Impacts of agrisolar co-location on the food–energy–water nexus and economic security. *Nat. Sustain.* **8**, 702–713 (2025).
- 26. Helveston, J. P., He, G. & Davidson, M. R. Quantifying the cost savings of global solar photovoltaic supply chains. *Nature* **612**, 83–87 (2022).
- 27. Yap, W., Wu, A. N., Miller, C. & Biljecki, F. Revealing building operating carbon

- dynamics for multiple cities. *Nat. Sustain.* https://doi.org/10.1038/s41893-025- 01615-8 (2025) doi:10.1038/s41893-025-01615-8.
- 28. Beckage, B., Moore, F. C. & Lacasse, K. Incorporating human behaviour into Earth system modelling. *Nat. Hum. Behav.* **6**, 1493–1502 (2022).
- 29. Zhou, Q., Yue, W., Li, M., Hu, H. & Zhang, L. Spatial assessment of settlement consolidation potential: insights from Zhejiang Province, China. *Humanit. Soc. Sci. Commun.* **12**, 551 (2025).
- 30. Rudin, C. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nat. Mach. Intell.* **1**, 206–215 (2019).
- 31. Ngubane, Z., Bergion, V., Dzwairo, B., Stenström, T. A. & Sokolova, E. Multicriteria decision analysis framework for engaging stakeholders in river pollution risk management. *Sci. Rep.* **14**, 7125 (2024).
- 32. Jones, K. R., Watson, J. E. M., Possingham, H. P. & Klein, C. J. Incorporating climate change into spatial conservation prioritisation: A review. *Biol. Conserv.* **194**, 121–130 (2016).
- 33. Strassburg, B. B. N. *et al.* Global priority areas for ecosystem restoration. *Nature* **586**, 724–729 (2020).
- 34. Fujita, K. S. *et al.* Georectified polygon database of ground-mounted large-scale solar photovoltaic sites in the United States. *Sci. Data* **10**, 760 (2023).
- 35. Lark, T. J., Spawn, S. A., Bougie, M. & Gibbs, H. K. Cropland expansion in the

- United States produces marginal yields at high costs to wildlife. *Nat. Commun.* **11**, 4295 (2020).
- 36. Xie, Y. *et al.* Cropland abandonment between 1986 and 2018 across the United States: spatiotemporal patterns and current land uses. *Environ. Res. Lett.* **19**, 044009 (2024).
- 37. Zhou, J. *et al.* Land suitability evaluation of large-scale photovoltaic plants using structural equation models. *Resour. Conserv. Recycl.* **198**, 107179 (2023).
- 38. Hronis, C. & Beall, D. R. Housing Characteristics Overview from the 2020 Residential Energy Consumption Survey (RECS). (2020).
- 39. Mai, T. *et al.* Electric Technology Adoption and Energy Consumption. https://doi.org/10.7799/1461472 (25 AD).
- 40. Qiu, S. *et al.* A shift from human-directed to undirected wild land disturbances in the USA. *Nat. Geosci.* **18**, 989–996 (2025).
- 41. Dunnett, S., Sorichetta, A., Taylor, G. & Eigenbrod, F. Harmonised global datasets of wind and solar farm locations and power. *Sci. Data* **7**, 130 (2020).
- 42. Center For International Earth Science Information Network (CIESIN) & Columbia University. Gridded Population of the World, Version 4 (GPWv4): Population Density, Revision 11. *NASA Socioeconomic Data and Applications Center (SEDAC)* vol. 4.11 NASA Socioeconomic Data and Applications Center https://doi.org/10.7927/H49C6VHW (2017).
- 43. Kummu, M., Kosonen, M. & Masoumzadeh Sayyar, S. Downscaled gridded

- global dataset for gross domestic product (GDP) per capita PPP over 1990–2022. *Sci. Data* **12**, 178 (2025).
- 44. Liu, Z., Huang, S., Fang, C., Guan, L. & Liu, M. Global urban and rural settlement dataset from 2000 to 2020. *Sci. Data* **11**, 1359 (2024).
- 45. Arderne, C., Zorn, C., Nicolas, C. & Koks, E. E. Predictive mapping of the global power system using open data. *Sci. Data* **7**, 19 (2020).
- 46. Nirandjan, S., Koks, E. E., Ward, P. J. & Aerts, J. C. J. H. A spatially-explicit harmonized global dataset of critical infrastructure. *Sci. Data* **9**, 150 (2022).
- 47. Karger, D. N., Brun, P. & Zilker, F. CHELSA-monthly climate data at high resolution. *EnviDat* https://doi.org/10.16904/envidat.686 (2025).
- 48. Ghosh, S., Ganguly, D., Dey, S. & Chowdhury, S. G. Future photovoltaic potential in India: navigating the interplay between air pollution control and climate change mitigation. *Environ. Res. Lett.* **19**, 124030 (2024).
- 49. IFI TWG. *Harmonized IFI Default Grid Factors 2021 v3.2*. https://unfccc.int/documents/461676 (2021).
- 50. Qiu, T. *et al.* Potential assessment of photovoltaic power generation in China. *Renew. Sustain. Energy Rev.* **154**, 111900 (2022).
- 51. IEA. Power carbon intensity in key regions in the Stated Policies Scenario, 2000– 2040. (2019).
- 52. Bolinger, M. & Bolinger, G. Land Requirements for Utility-Scale PV: An Empirical Update on Power and Energy Density. *IEEE J. Photovolt.* **12**, 589–594

(2022).

- 53. Walker, W. S. *et al.* The global potential for increased storage of carbon on land. *Proc. Natl. Acad. Sci.* **119**, e2111312119 (2022).
- 54. Cook-Patton, S. C. *et al.* Mapping carbon accumulation potential from global natural forest regrowth. *Nature* **585**, 545–550 (2020).
- 55. Ma, H. *et al.* The global distribution and environmental drivers of aboveground versus belowground plant biomass. *Nat. Ecol. Evol.* **5**, 1110–1122 (2021).
- 56. *Global Soil Organic Carbon Sequestration Potential Map – GSOCseq v.1.1*. https://doi.org/10.4060/cb9002en (2022).
- 57. Harris, N. L. *et al.* Global maps of twenty-first century forest carbon fluxes. *Nat. Clim. Change* **11**, 234–240 (2021).
- 58. Fischer, G. *et al. Global Agro-Ecological Zones v4–Model Documentation*. (Food & Agriculture Org., 2021).
- 59. Intergovernmental Panel on Climate Change (IPCC). *2019 Refinement to the 2006 IPCC Guidelines for National Greenhouse Gas Inventories*. https://www.ipcc-nggip.iges.or.jp/public/2019rf/vol4.html (2019).
- 60. Byers, E. *et al.* AR6 scenarios database. (2022).
- 61. Cointe, B. The AR6 Scenario Explorer and the history of IPCC Scenarios Databases: evolutions and challenges for transparency, pluralism and policyrelevance. *Npj Clim. Action* **3**, 3 (2024).
- 62. Cherp, A., Vinichenko, V., Tosun, J., Gordon, J. A. & Jewell, J. National growth

- dynamics of wind and solar power compared to the growth required for global climate targets. *Nat. Energy* **6**, 742–754 (2021).
- 63. FAO. *The Future of Food and Agriculture – Alternative Pathways to 2050*. https://www.fao.org/global-perspectives-studies/food-agriculture-projections-to-2050/en/.
- 64. Strassburg, B. B. N. *et al.* Global priority areas for ecosystem restoration. *Nature* **586**, 724–729 (2020).
- 65. Hoppe, R. A. Structure and finances of US farms: Family farm report, 2014 edition. (2014).
- 66. Paulson, N., Schnitkey, G. & Zulauf, C. Outlook for Farmland Values in 2025. *Farmdoc Dly.* **14**, (2024).