# Coral Reef Bleaching Prediction: A Machine Learning Approach

This project analyzes and predicts coral reef bleaching events in Indian waters using machine learning algorithms. By combining oceanographic data, climate indices, and coral reef biodiversity information, we've developed models to forecast bleaching events and identify key environmental drivers affecting coral health.

## Project Overview

Coral reef bleaching is a significant ecological concern exacerbated by climate change. This research utilizes multiple environmental parameters including sea surface temperature (SST), ocean chemistry (pH, pCO2), climate oscillations (ENSO, IOD), and coral genera diversity to predict bleaching events across four reef systems in India. Currently, our work has focused on data gathering, preprocessing, and analysis. The machine learning models have been proposed and planned for future implementation.

## Data Sources and Processing

### Primary Data Sources

The project integrates data from several scientific repositories and research papers:

- **Sea Surface Temperature (SST)**: Data obtained from NOAA time series datasets was converted to a dataframe format and combined for all four reef systems.
- **Ocean Chemistry**: Parameters including salinity, fCO2, pCO2, and pH were extracted from the NCEI OceanSODA-ETHZ .nc dataset using the netCDF library, with data filtered based on latitude and longitude for the four reef systems.
- **Coral Genera**: Taxonomic data on coral genera was extracted from the National Biodiversity Authority's "Coral Reefs in India" paper and compiled into a structured dataframe.
- **Bleaching Instances**: Bleaching event data was collected from research by Thinesh Thangadurai et al. and BCO DMO time series datasets. The Excel data was converted into a dataframe, filtered for Indian reefs, and classified (bleaching intensity was converted to a binary classification).
- **Climate Indices**: Indian Ocean Dipole (IOD) and El Niño Southern Oscillation (ENSO) data were sourced from NASA time series JSON files, with date formats standardized for integration.

## Data Preprocessing Pipeline

### Initial Processing

Each dataset underwent transformations to standardize formats:
- Converted various file formats (txt, nc, xlsx, json) into standardized CSV files.
- Applied geographic filtering to focus on Indian reef systems.
- Standardized date formats across all datasets.

### Data Integration

The datasets were unified through these key steps:
- Standardized date formats and reef naming conventions.
- Combined datasets using date fields (day-month-year) as the primary key.
- For datasets with coarser resolution (e.g., bleaching instances), monthly or reef-name matching was employed.
- Aggregated daily data into monthly values.
- Calculated monthly statistics for oceanographic variables (e.g., mean salinity, CO2 metrics, pH, IOD, ENSO; maximum, minimum, and mean SST; Degree Heating Week calculations).
- Assigned a value of 0 for months without recorded bleaching.
- Filtered the final dataset to include only the period 1995–2020.
- Performed one-hot encoding on the coral genera data.

### Feature Engineering and Selection

- Applied normalization to standardize variable scales.
- Conducted correlation analysis to identify relationships between variables.
- Removed redundant or low-correlation features to improve model performance.

## Machine Learning Models

We have proposed to use the following machine learning models for predicting bleaching events:
- **Random Forest**: General-purpose prediction with interpretable results.
- **XGBoost**: High-accuracy prediction, effective with complex variable relationships.
- **Support Vector Machine (SVM)**: Suitable for high-dimensional and non-linear data.
- **K-Means Clustering**: Useful for identifying natural groupings in oceanographic conditions.

*Detailed model implementation and evaluation will be added in future updates.*

## Repository Structure

*Directory structure will update soon.*

## Installation and Usage

To clone the repository, run:

```bash
git clone https://github.com/shrijacked/mlpr-project.git
```

## Future Work

Next steps include:
- Implementing and evaluating the proposed machine learning models.
- Expanding data integration with additional sources (e.g., remote sensing data).
- Refining our preprocessing pipeline and further analysis.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

--- 
