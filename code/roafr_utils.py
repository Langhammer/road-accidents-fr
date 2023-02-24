"""
roafr_utils provides functions that are available to all notebooks
"""
import pandas as pd
from pyproj import Transformer


def df_testing_info(df):
    """Returns a DataFrame that describes the given DataFrame"""
    df_dtypes_and_na_counts = pd.DataFrame({'dtypes':df.dtypes, 'n_na': df.isna().sum()})
    return pd.concat([df.describe().T, df_dtypes_and_na_counts])

def df_compare_to_description(df, description_filepath):
    '''Check whether or not the data is up-to-date 
    if DataFrame file can't be tracked on github because of it's file size)
    '''
    pd.testing.assert_frame_equal(left=(pd.read_pickle(description_filepath)), \
                         right=df_testing_info(df),\
                         check_dtype=False, check_exact=False)

def df_to_pickle(df, label, filepath='../data/'):
    """Exports a large DataFrame to pickle and creates a descriptive pickle file
    for tracking in GitHub. The filepath must not contain the filename.
    """
    df_check_info = df_testing_info(df)
    df_check_info.to_pickle(filepath+label+'.check')
    df.to_pickle(filepath+label+'.p')

def df_from_pickle(filepath):
    """Reads a large DataFrame from pickle and checks for consistency with 
    accompanying _check.p file.
    """
    df = pd.read_pickle(filepath)
    description_filepath = filepath.rstrip('.p') + '.check'
    df_compare_to_description(df=df, description_filepath=description_filepath)
    return df

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

def df_geotransform(df, lat_col='latitude', lon_col='longitude'):
    """Transforms WGS84 latitude/longitude to web mercator"""
    geotransformer = Transformer.from_crs("EPSG:4326", "EPSG:3857")
    lat, lon = geotransformer.transform(df[lat_col], df[lon_col])   # pylint: disable=E0633 # Attempting to unpack a non-sequence (unpacking-non-sequence)
    df.loc[:,lat_col] = lat
    df.loc[:,lon_col] = lon
    return df
