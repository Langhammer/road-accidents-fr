import numpy as np
import pandas as pd

def read_csv_of_year(years=None, data_categories=None):
    '''Imports the 4 csv files for the given time range and returns them as a dictionary
    If no years/categories are specified, all years/categories will be loaded
    '''

    french_categories = {'characteristics': 'caracteristiques',
                         'locations':'lieux',
                         'persons':'usagers',
                         'vehicles':'vehicules'}
    
    # If no years are specified, all years will be loaded
    if years is None:
        years = list(range(2005,2022))

    # If no categories are specified, all categories will be loaded
    if data_categories is None:
        data_categories = french_categories.keys()
    df_dict = {}

    for year in years:
        # In 2009, the characteristics dataset uses tab instead of comma
        if year==2009:
            separators = ['\t'] + [',']*3
        else:
            separators = [',']*4

        # The name separator is used in the filename is changed in 2017
        if year>=2017:
            name_separator = '-'
            if year>=2019:
                separators = [';']*4
        else: 
            name_separator = '_'

        
        # The data will be read and stored in a dictionary of dictionaries, 
        # e.g. {2005: {'characteristics': <pd.DataFrame>}}
        this_df_dict = {}        
        this_year_str = str(year)
        for this_category, this_sep in zip(data_categories, separators):
            # We need the French name of the category for the filename
            this_french_category = french_categories[this_category]
            this_file_path_and_name = '../data/annual_accidents/'+this_year_str+'/' + this_french_category+name_separator+this_year_str+'.csv'
            this_df_dict[this_category] = pd.read_csv(this_file_path_and_name, encoding='utf-8', sep=this_sep, low_memory=False)
        df_dict[year] = this_df_dict

    # The datasets will be merged, resulting in a dict like
    # df_dict = {'characteristics': <pd.DataFrame>}, where the DataFrame contains the information for all years requested.
    dict_of_category_dfs = {}
    for this_category in data_categories:
        dict_of_category_dfs[this_category] = pd.concat([df_dict[year][this_category] for year in years], ignore_index=True)
    return dict_of_category_dfs