<img src="/images/headers/readme_header.svg"  width="1080"><small><i>
*Photo by [Bennett Tobias](https://unsplash.com/fr/@bwtobias) on Unsplash, modified*
</i></small>

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Langhammer/road-accidents-fr/main)  [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://langhammer-road-accidents-fr-streamlit-app-xnqwbs.streamlit.app/)  
[![Tests](https://github.com/Langhammer/road-accidents-fr/actions/workflows/tests.yml/badge.svg)](https://github.com/Langhammer/road-accidents-fr/actions/workflows/tests.yml) [![Pylint](https://github.com/Langhammer/road-accidents-fr/actions/workflows/pylint.yml/badge.svg)](https://github.com/Langhammer/road-accidents-fr/actions/workflows/pylint.yml)  [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![black](https://github.com/Langhammer/road-accidents-fr/actions/workflows/black.yml/badge.svg)](https://github.com/Langhammer/road-accidents-fr/actions/workflows/black.yml)

# About
The aim of this project is to analyze and evaluate the factors influencing the occurence 
and the severity of traffic accidents. For this purpose, the traffic accident data of the 
French government are visualized and the significance of the individual variables is 
investigated by means of machine learning.  

Prior to starting this project, I had already worked with this dataset in a team project 
as part of my training at a data science bootcamp. 
From that project, I only took over the parts that I conceived and programmed myself.

# Structure
Generally, the Jupyter Notebooks are converted to the py:percent format via 
[Jupytext](https://github.com/mwouts/jupytext). These files do not contain any output by design. 
You can use Jupytext to convert these files to Jupyter Notebooks. 

There will be an output branch which will contain the converted files with output. 
**The .ipynb files of this project will not be up-to-date most of the time.** 
To update the notebook views, you can run make run-all-notebooks.
| Python script | Jupyter Notebook | Title | Description |
| ------------- | ---------------- | ----- | ----------- |
| [nb_1.py](https://github.com/Langhammer/road-accidents-fr/tree/main/notebooks/nb_1.py) | <a href="https://nbviewer.org/github/Langhammer/road-accidents-fr/blob/main/notebooks/VIEW_nb_1.ipynb" target="_blank" rel="noopener noreferrer">![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)</a> | Data Import and Cleaning      |
| [nb_2.py](https://github.com/Langhammer/road-accidents-fr/tree/main/notebooks/nb_2.py) | <a href="https://nbviewer.org/github/Langhammer/road-accidents-fr/blob/main/notebooks/VIEW_nb_2.ipynb" target="_blank" rel="noopener noreferrer">![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)</a> | Visualization                 |
| [nb_3.py](https://github.com/Langhammer/road-accidents-fr/tree/main/notebooks/nb_3.py) | <a href="https://nbviewer.org/github/Langhammer/road-accidents-fr/blob/main/notebooks/VIEW_nb_3.ipynb" target="_blank" rel="noopener noreferrer">![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)</a> | Conventional Machine Learning | XGBoost, Random Forest |
| [nb_4.py](https://github.com/Langhammer/road-accidents-fr/tree/main/notebooks/nb_4.py) | <a href="https://nbviewer.org/github/Langhammer/road-accidents-fr/blob/main/notebooks/VIEW_nb_4.ipynb" target="_blank" rel="noopener noreferrer">![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)</a> | Artificial Neural Networks | Dense Neural Networks with TensorFlow/Keras and Coral_Ordinal |




# References
* [Unprocessed Datasets provided by the French Ministry of the Interior and the Overseas](https://www.data.gouv.fr/en/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2021/)
* [INSEE Population Data](https://www.insee.fr/fr/statistiques/6011070?sommaire=6011075)
* [INSEE Department Codes](https://www.insee.fr/fr/information/5057840)
