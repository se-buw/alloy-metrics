# Alloy Metrics

## Introduction
This repository contains the artifacts for the paper "On Writing Alloy Models: Metrics and a new Dataset" submitted to the 11th International Conference on Rigorous State Based Methods (ABZ). The repository contains the following structure:

```bash
+---analysis    # Python scripts used to analyze the data
|       a4f_model_analysis.py
|       ...
+---data        # Raw data used in the analysis
|   +---code    # Alloy models saved as .als files
|   +---json    # JSON files on the datasets        
+---lib   
+---results     # Results of the analysis
|   |   a4f_chain_longest_status.csv
|   |   alloyEx_spec_analysis.csv
|   |   fmp_chain_longest_status.csv
|   |   ...
|   +---plots   # Plots generated from the analysis
|           a4f_halstead_clustered_std_dev.pdf
|           fmp_halstead_clustered_std_dev.csv
|           ...         
+---src             # Java source code for the metrics                 
|-alloy-metrics.jar # Jar file for the metrics
|-requirements.txt  # Python requirements
```

## Requirements
- Python >= 3.8
- Java >= 17

## Preparing the environment
- **Alloy4Fun Dataset**
    - Download the Alloy4Fun dataset from [https://zenodo.org/records/8123547](https://zenodo.org/records/8123547)
    - All the json files should be placed in the `data/json/a4f` folder
- **FMP Dataset**
    - Download the FMP dataset from [https://zenodo.org/records/14865553](https://zenodo.org/records/14865553)
    - All the json files should be placed in the `data/json/fmp.json` 

## Running the analysis
All the scripts are located in the `analysis` folder. To run the analysis, execute the following command:

```bash
python analysis/<script_name>.py
```

## Results
The preliminary results are stored in the `results` folder. The plots are stored in the `results/plots` folder.
All the data related to the Alloy4Fun dataset is prefixed with `a4f_` and the FMP dataset is prefixed with `fmp_`.


## License
This repository is licensed under the MIT License. Please see the LICENSE file for more details.
