![Pylint](https://github.com/Langhammer/road-accidents-fr/actions/workflows/pylint.yml/badge.svg)  [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![black](https://github.com/Langhammer/road-accidents-fr/actions/workflows/black.yml/badge.svg)](https://github.com/Langhammer/road-accidents-fr/actions/workflows/black.yml)
Road Accident Injuries in France  
(road-accidents-fr)
=================
<img src="images/eiffel_slow.gif" alt="Photography of the Eiffel Tower" width="400"/>  

*Photo by [Bennett Tobias](https://unsplash.com/fr/@bwtobias) on Unsplash, modified*

# About
The aim of this project is to analyze and evaluate the factors influencing the occurence and the severity of traffic accidents. For this purpose, the traffic accident data of the French government are visualized and the significance of the individual variables is investigated by means of machine learning.  

Prior to starting this project, I had already worked with this dataset in a team project as part of my training at a data science bootcamp. 
From that project, I only took over the parts that I conceived and programmed myself.

# Structure
Generally, the Jupyter Notebooks are converted to the py:percent format via [Jupytext](https://github.com/mwouts/jupytext). These files do not contain any output by design. You can use Jupytext to convert these files to Jupyter Notebooks. 

There will be an output branch which will contain the converted files with output. **The .ipynb files of this project will not be up-to-date most of the time.**
| Python script | Jupyter Notebook | Title | Description
| ------------- | ---------------- | ----- | ----------- |
| [nb_1.py](https://github.com/Langhammer/road-accidents-fr/tree/main/code/nb_1.py) | [nb_1_view.ipynb](https://github.com/Langhammer/road-accidents-fr/tree/main/code/nb_1_view.ipynb) | Data Import and Cleaning |
| [nb_2.py](https://github.com/Langhammer/road-accidents-fr/tree/main/code/nb_2.py)  | [nb_2_view.ipynb](https://github.com/Langhammer/road-accidents-fr/tree/main/code/nb_2_view.ipynb) | Visualization |
| [nb_3.py](https://github.com/Langhammer/road-accidents-fr/tree/main/code/nb_3.py)  | [nb_3_view.ipynb](https://github.com/Langhammer/road-accidents-fr/tree/main/code/nb_3_view.ipynb) | Conventional Machine Learning | XGBoost, Random Forest
| [roafr_utils.py](https://github.com/Langhammer/road-accidents-fr/tree/main/code/roafr_utils.py) | -- | --- | Useful functions


# References
* [Unprocessed Datasets provided by the French Ministry of the Interior and the Overseas](https://www.data.gouv.fr/en/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2021/)
* [INSEE Population Data](https://www.insee.fr/fr/statistiques/6011070?sommaire=6011075)
* [INSEE Department Codes](https://www.insee.fr/fr/information/5057840)
