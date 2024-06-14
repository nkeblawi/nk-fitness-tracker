# nk-fitness-tracker

# ---------------------------------------------------------- #
#    BASED ON THE QUANTIFIED SELF BOOK                       #
# ---------------------------------------------------------- #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
# ---------------------------------------------------------- #                                                  

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Supervised ML model that uses fitness data to predict multiple classification targets. 
Accelerometer and Gyroscope data based on original report linked below:
<a target="_blank"
    href="https://secure-res.craft.do/v2/DkCrM8qa8MpqYUv1hZTca1NmEQN8BUD3jgq4E4hUHHYsSECHyPEAMTuaPRwgmvY9KMGbjTiSXxGeD7e4SJpRu6vjQCpDRVbKBT3ywX4ZgDEdyoWBQqxvdJYxVxyQcMqvptguFPNpAqP4UWV7Ub9hpX9iyYUdXqXhQy4foenh4nasYefmgkpSP3MFzrPaz2Ma6jwhTCgzJSMEvfdNeAywK2Mz1JNqaAk8jUwyVp8zpBNxcQzDiwmvvnWdapkVkZmwRTkNbF3iKM5qbMWgnpQa2fhcEzXebG7qq3tC6etT9mErJRZSBrhEXkvDCRhLnsMD9vPzLALSyuBuX9DR6vfKUUs7qEPXArtHkU52wtg2oWfJShZeHcigvgQhbfgXY1o8QAV8W35YeqQYeVZ8SHwZt9TsfkhUEHReUVBYH7hKKdYEjtsJnkkCZ4ncoC9PSdQsSr8BTb9MbvyZTQfEgvBP2HqmtcM45ZLkj/Mini%20Master%20Project%20-%20Exploring%20the%20Possibilities%20of%20Context%20Aware%20Applications%20for%20Strength%20Training.pdf">
    Mini Master Project Report
</a>

MetaMotion dataset can be downloaded here (unzipped into ./data/raw/ folder):
<a target="_blank"
    href="https://secure-res.craft.do/v2/VDcx9pyWxusPMveFX3m6KG6HXbjF2gSLkdV3zTrPX8WWrkjoh6aJinsjsSg9tEdgeMZcjDWdtZd28EhN2o2xY1Ui9TfDF5BLtGfUvYhVMqbVgdBdG7UWggpP3rR3DnS5CP9iupmM9rQQPpc9EREkeFXTSsmWXLbb98D3kdakxcembuRAC65ewTeSez8H1yd1GqFYoL76ZhHHGYrL1a4QgNa3G1pHhMLMViLV1PjeuDVxboZBTgp4S8SUsyZZDTixk5jNFwM8BZxff3Mwd8JtxQYkKkGsj8mVm75oGZaFbSGXAkLTsP/MetaMotion.zip">
    MetaMotion.zip
</a>

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for nk_fitness_tracker
│                         and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── nk_fitness_tracker                <- Source code for use in this project.
    │
    ├── __init__.py    <- Makes nk_fitness_tracker a Python module
    │
    ├── data           <- Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── features       <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    │
    ├── models         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```

--------

