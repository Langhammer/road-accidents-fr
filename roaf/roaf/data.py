"""
roafr_utils provides functions that are available to all notebooks
"""
import pandas as pd
from pyproj import Transformer
from bokeh.io import output_file
from bokeh.models import ColumnDataSource, HoverTool, WheelZoomTool
from bokeh.plotting import figure, show, curdoc
from bokeh.embed import file_html
from bokeh.resources import CDN
from datetime import datetime

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

def df_to_pickle(df, label, filepath='../data/processed/'):
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
            this_file_path_and_name = '../data/raw/annual_accidents/'+ \
                                        this_year_str + '/' + \
                                        this_french_category + \
                                        name_separator + \
                                        this_year_str + \
                                        '.csv'
            this_df_dict[this_category] = pd.read_csv(this_file_path_and_name, 
                                                      encoding='utf-8', 
                                                      sep=this_sep, 
                                                      low_memory=False)
        df_dict[year] = this_df_dict

    # The datasets will be merged, resulting in a dict like
    # df_dict = {'characteristics': <pd.DataFrame>}, where the DataFrame contains the information 
    # for all years requested.
    dict_of_category_dfs = {}
    for this_category in data_categories:
        dict_of_category_dfs[this_category] = pd.concat(
                                                [df_dict[year][this_category] for year in years], 
                                                ignore_index=True)
    return dict_of_category_dfs

def df_geotransform(df, lat_col='latitude', lon_col='longitude'):
    """Transforms WGS84 latitude/longitude to web mercator"""
    geotransformer = Transformer.from_crs("EPSG:4326", "EPSG:3857")
    lat, lon = geotransformer.transform(df[lat_col], df[lon_col])   # pylint: disable=E0633
    df.loc[:,lat_col] = lat
    df.loc[:,lon_col] = lon
    return df

def plot_geodata(df, plot_date, output_path, n_plot_max=10_000, figsize=None, return_html=False):
    """Plot gps data on map"""
    output_file(output_path)

    TOOLTIPS = [
        ("index", "@accident_id"),
        ("(lat, lon)", "(@lat, @lon)"),
        ("severity", "@severity_label"),
    ]

    p = figure(x_range=(-750_000, 1_125_000), y_range=(5_755_000, 5_955_000),
            x_axis_type="mercator", y_axis_type="mercator", tooltips=TOOLTIPS)
    
    if figsize is not None:
        if isinstance(figsize, int):
            p.width = figsize
            p.height = figsize
        elif len(figsize)==2: 
            p.width = figsize[0]
            p.height = figsize[1]
        else:
            p.width = figsize=[0]
            p.height = figsize=[0]

    p.add_tile('STAMEN_TONER')

    # Size of sample of data points to plot. 
    # More than 10_000 data points can become very slow
    plot_cols = ['accident_id', 'longitude', 'latitude', 'severity']
    plot_df = df[df['date'].apply(datetime.date)==plot_date][plot_cols]

    if len(plot_df)>n_plot_max:
        plot_df = plot_df.sample(n=n_plot_max)

    colors = plot_df['severity'].replace({1:'blue', 2:'orangered', 3:'red'})
    severity_labels = plot_df['severity'].replace({1:'Unharmed', 2:'Injured', 3:'Killed'})
    markers = plot_df['severity'].replace({1:'circle', 2:'square', 3:'triangle'})

    source = ColumnDataSource(data={'accident_id': plot_df['accident_id'],
                                    'lat':plot_df['latitude'],
                                    'lon':plot_df['longitude'],
                                    'severity': plot_df['severity'],
                                    'color':colors,
                                    'severity_label':severity_labels,
                                    'marker':markers})

    # Add accident locations
    c = p.circle(x='lat',
                y='lon', 
                size=15,
                fill_alpha=0.8,
                fill_color='color',
                line_color='grey',
                line_width=1,
                source=source)

    HoverTool(tooltips=TOOLTIPS, renderers=[c])
    p.toolbar.active_scroll = p.select_one(WheelZoomTool)

    # Change bokeh theme
    curdoc().theme = 'dark_minimal'

    if return_html:
        curdoc().add_root(p)
        return file_html(p,CDN,'map')
    else:
        show(p)