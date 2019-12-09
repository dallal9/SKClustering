   
#     if missing_val != 0:
        
#         imputation_types = ['mean', 'median', 'mode']
#         imputed_data = data.copy()
# ​
#         results = pd.DataFrame()
#         for index, num_imput_type in enumerate(imputation_types):
                       
#             num_cols = list(num_cols) 
#             imputed_data[num_cols] = numeric_impute(data, num_cols, num_imput_type)
#             metafeatures1['num_imput_type'] = num_imput_type
#             metafeatures = all_metafeatures(imputed_data, num_cols, metafeatures1)
#             df = pd.DataFrame([metafeatures])
#             results = pd.concat([results, df], axis=0)
#     else:
#         metafeatures1['num_imput_type'] = None
#         metafeatures = all_metafeatures(data, num_cols, metafeatures1)
#         results = pd.DataFrame([metafeatures])
    
#     dataset_name = file.split('\\')[-1]
#     results['dataset'] = dataset_name