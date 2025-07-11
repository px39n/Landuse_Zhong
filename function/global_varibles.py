


# # Calculate point density for clustering
# xy_coords = np.vstack([df['lon'], df['lat']])
# z = gaussian_kde(xy_coords)(xy_coords)
# # Save the point density data to pickle for later use
# density_data = {
#     'coordinates': xy_coords,
#     'density': z
# }
# # Save the point density data to pickle
# with open('data/point_density.pkl', 'wb') as f:
#     pickle.dump(density_data, f)

abandon_2d_variable = [
    "current_abandonment",
    "recultivation", 
    "abandonment_duration",
    "abandonment_year"
]
fea_3d_variable = [
    'GDPpc',
    'GDPtot',   
    'GURdist',
    'Population',
    'gdmp',
    'rsds',
    'tas',
    'wind'
]
fea_2d_variable = [
    'DEM',
    'Powerdist',
    'PrimaryRoad',
    'SecondaryRoad',
    'Slope',
    'TertiaryRoad'
]
PATHS = {
    'abandonment': r"D:\xarray\merged_chunk_2\*.nc",
    'feature':     "D:/xarray/aligned2/Feature_all/*.nc",
    'csv':         "aligned_for_training0519.csv",
    'prediction':  "",
    'prediction_us': "",
    'test_output': "positive_samples_test_500.csv",
    'output':      "positive_samples_full_with_features.csv",    
    'us_county':     r'data\cb_2018_us_county_5m.shp',
    'CN_sheng': r'data\sheng2022.shp',
    'World_shp': r'data\main_ADM_0.shp', 
    'us_abandon': r'data\us_abandon_clean.csv',
    'us_pv_embedding': r'data\training_embedding.csv',
    'point_density': r'data\point_density.pkl',
    'data_prediction_net_benefit': r'data\4.data_prediction_net_benefit.csv',
    'data_strategies': r'data\4.df_strategies.csv',

    # 以下数据是5.1模块计算成本-效益生成分析结果
    
    'df_pv_npv': r'data/5.1_photovoltaic_npv_time_series_corrected_2020usd.csv',
    'df_pv_rcp_year':r'data/5.1_photovoltaic_summary_corrected_by_rcp_year_2020usd.csv',
    'df_pv_rcp_overall':r'data/5.1_photovoltaic_rcp_averages_corrected_2020usd.csv',
    'df_pv_2050':r'data/5.1_photovoltaic_with_economic_feasibility_corrected.csv',

    'df_agricultural_npv':r'data/5.1_agricultural_npv_time_series_corrected_2020usd.csv',
    'df_agricultural_rcp_year':r'data/5.1_agricultural_summary_corrected_by_rcp_year_2020usd.csv',
    'df_agricultural_rcp_overall':r'data/5.1_agricultural_rcp_averages_corrected_2020usd.csv',
    'df_agricultural_2050':r'data/5.1_agricultural_production_with_economic_feasibility_corrected.csv',

    'df_afforestation_npv': r'data/5.1_afforestation_npv_time_series_corrected_2020usd.csv',
    'df_afforestation_rcp_year': r'data/5.1_afforestation_summary_corrected_by_rcp_year_2020usd.csv',
    'df_afforestation_rcp_overall': r'data/5.1_afforestation_rcp_averages_corrected_2020usd.csv',
    'df_afforestation_2050': r'data/5.1_afforestation_with_economic_feasibility_corrected.csv',

    'df_natural_npv': r'data/5.1_natural_restoration_npv_time_series_2020usd.csv',
    


    

    

    

    


}

ZERO_COLS = [
     'GDPpc', 'GDPtot', 'GURdist', 'Population',
    'PrimaryRoad', 'SecondaryRoad', 'TertiaryRoad', 'gdmp'
]
YEARS = [2018, 2020]

NUMERIC_FEATURES = [
    'lat','lon','GDPpc', 'GDPtot', 'GURdist', 'DEM','Slope',
    'Population','Powerdist','PrimaryRoad','SecondaryRoad','TertiaryRoad',
    'gdmp','rsds','tas','wind'
]
CAT_COLS = ['landcover']

ABANDON_COLS = ['abandonment_year','abandonment_duration', 'current_abandonment']

NONE_ABANDON_COLS = ['recultivation']

time=['2018-01-01','2020-01-01']


