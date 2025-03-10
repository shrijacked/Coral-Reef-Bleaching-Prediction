# Coral Reef Bleaching Prediction: A Machine Learning Approach

This project analyzes and predicts coral reef bleaching events in Indian waters using machine learning algorithms. By combining oceanographic data, climate indices, and coral reef biodiversity information, we've developed models to forecast bleaching events and identify key environmental drivers affecting coral health.

## Project Overview

Coral reef bleaching is a significant ecological concern exacerbated by climate change. This research utilizes multiple environmental parameters including sea surface temperature (SST), ocean chemistry (pH, pCO2), climate oscillations (ENSO, IOD), and coral genera diversity to predict bleaching events across four reef systems in India. The project implements a comprehensive data pipeline that processes diverse data formats and applies machine learning algorithms to identify patterns associated with bleaching occurrences.

## Data Sources and Processing

### Primary Data Sources

The project integrates data from several scientific repositories and research papers:

#### Sea Surface Temperature (SST)
SST data was obtained from NOAA time series datasets. The data was converted to a dataframe format and combined for all four reef systems with an added reef name column. The final dataset contains 58,702 rows of SST observations across the study sites.

#### Ocean Chemistry
Ocean chemistry parameters including salinity, fCO2, pCO2, and pH were extracted from the NCEI OceanSODA-ETHZ .nc dataset using the netCDF library. The data was filtered based on latitude and longitude coordinates for the four reef systems. After removing missing values (4,920 rows), the final dataset contains 21,156 rows.

#### Coral Genera
Taxonomic data on coral genera was extracted from the National Biodiversity Authority's "Coral Reefs in India" paper. This information was manually compiled into a structured format and converted to a dataframe organized by reef location.

#### Bleaching Instances
Bleaching event data was collected from research by Thinesh Thangadurai et al. and BCO DMO time series datasets. The Excel data was converted to a dataframe, filtered for Indian reefs, and classified according to reef location based on coordinates. Bleaching intensity was converted to a binary classification. The final dataset contains 55 rows of bleaching observations.

#### Climate Indices
Indian Ocean Dipole (IOD) and El Niño Southern Oscillation (ENSO) data was sourced from NASA time series JSON files. The decimal date format was converted to day-month-year format for integration with other datasets.

## Data Preprocessing Pipeline

### Initial Processing
Each dataset underwent specific transformations to standardize formats:
- Converted various file formats (txt, nc, xlsx, json) to standardized CSV files
- Applied geographic filtering to focus on Indian reef systems
- Standardized date formats across all datasets

### Data Integration
The datasets were unified through several key steps:
1. Standardized date formats and reef naming conventions across all datasets
2. Combined datasets using date fields (day-month-year) as the primary key
3. For datasets lacking daily resolution (e.g., bleaching instances), monthly or reef-name matching was employed
4. Daily data was aggregated to monthly values to match the temporal resolution of bleaching data
5. Calculated monthly statistics for oceanographic variables:
   - Mean values for salinity, CO2 metrics, pH, IOD, and ENSO
   - Maximum, minimum, and mean values for SST
   - Degree Heating Week (DHW) calculations
6. Months without recorded bleaching instances were assigned a value of 0
7. The final dataset was filtered to include only the period 1995-2020, where data coverage was most complete
8. Performed one-hot encoding on the coral genera column
9. The final integrated dataset contains 1,201 rows

### Feature Engineering and Selection
- Applied normalization to standardize the scale of different variables
- Conducted correlation analysis to identify relationships between variables
- Removed redundant or low-correlation features to improve model performance

## Machine Learning Models

Four machine learning approaches were implemented and compared:

### Random Forest
- **Best For**: General-purpose prediction of bleaching events
- **Key Strengths**: Handles missing data, provides interpretable results, and effectively captures nonlinear relationships
- **Limitations**: Lower computational efficiency compared to XGBoost, slightly reduced accuracy

### XGBoost
- **Best For**: High-accuracy prediction of bleaching events
- **Key Strengths**: Captures complex variable relationships, efficient memory usage, effectively handles imbalanced data
- **Limitations**: Requires careful hyperparameter tuning, less interpretable than Random Forest

### Support Vector Machine (SVM)
- **Best For**: Datasets with complex decision boundaries
- **Key Strengths**: Effective with high-dimensional data, handles nonlinear classification problems
- **Limitations**: Computationally intensive for large datasets, sensitive to hyperparameter selection

### K-Means Clustering
- **Best For**: Identifying natural groupings in oceanographic conditions
- **Key Strengths**: Discovers patterns without requiring labeled data
- **Limitations**: Not directly predictive of bleaching events, requires standardized data, sensitive to initial cluster selection

## Repository Structure

```
├── data/
│   ├── raw/                # Original data files
│   ├── processed/          # Processed datasets
│   └── final/              # Final merged datasets for modeling
├── notebooks/
│   ├── dataAnalysis.ipynb  # Main data analysis notebook
│   └── models/             # Model implementation notebooks
├── results/
│   ├── figures/            # Generated visualizations
│   └── model_outputs/      # Model performance metrics
├── README.md               # Project documentation
└── requirements.txt        # Package dependencies
```

## Installation and Usage

### Prerequisites
- Python 3.8+
- Required packages: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, netCDF4

### Setup
1. Clone the repository:
```bash
git clone https://github.com/shrijacked/mlpr-project.git
cd mlpr-project
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the analysis notebooks:
```bash
jupyter notebook notebooks/dataAnalysis.ipynb
```

## Results and Applications

This project demonstrates the potential for machine learning approaches to predict coral bleaching events based on oceanographic and ecological data. The models developed can serve as early warning systems for reef managers and conservationists, allowing for proactive measures to mitigate bleaching impacts.

## Future Work

Potential extensions to this project include:
- Incorporating remote sensing data for improved spatial coverage
- Developing real-time prediction capabilities
- Extending the analysis to other reef systems globally
- Implementing more sophisticated deep learning approaches

## Resources

- [GitHub Repository](https://github.com/shrijacked/mlpr-project.git)
- [Project Presentation](https://www.canva.com/design/DAGhKdT3EY0/2l8et6y9x9T-3-5SA6E70A/edit)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

