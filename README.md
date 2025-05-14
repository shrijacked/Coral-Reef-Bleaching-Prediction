
# ReefCast: Coral Bleaching Prediction in India Using Environmental and Ecological Data
## Abstract
Coral reefs in India are increasingly threatened by climate change, yet most existing coral bleaching assessments remain retrospective, relying predominantly on sea surface temperature (SST) or image-based models. These approaches often neglect region-specific ecological factors and species-level responses, limiting their predictive utility. This study presents a predictive machine learning framework tailored specifically to Indian reef systems, incorporating localized, multi-variable environmental data. While global studies such as McCalla et al. (2023), which reported a 96% accuracy and an R² of 0.25 using Random Forest models, have advanced coral bleaching prediction, they lack regional specificity necessary for applications in India. Existing research on Indian reefs focuses on documenting habitat degradation and bleaching events but fails to employ predictive modeling based on integrated environmental and ecological features. Our model leverages variables including SST, pH, fCO2, salinity, temperature, turbidity, coral species composition, Degree Heating Weeks (DHW), and large-scale climatic drivers such as the Indian Ocean Dipole (IOD) and El Niño–Southern Oscillation (ENSO) indices. These features capture India’s distinct monsoon-driven oceanographic conditions and ecological diversity. Data were curated from reputable sources including NOAA, NASA, NCEI, BCO-DMO, and peer-reviewed literature. After rigorous preprocessing, we employed LSTM neural networks, XGBoost, LightGBM, and Random Forest algorithms. Among these, the LSTM model performed best, achieving a recall of 0.88, an F2 score of 0.78, and a ROC AUC of 0.97. By integrating diverse, region-specific variables, our model enhances the accuracy of coral bleaching forecasts and facilitates proactive reef management, offering a scalable and adaptable framework for ecological forecasting in other underrepresented reef systems globally.

The workflow includes **data preprocessing**, **exploratory data analysis (EDA)**, and **model training/evaluation**.

![Project Architecture](https://ai3011.plaksha.edu.in/Spring%202025/Images/Rishit%20Anand.png)

---

## 📁 Project Structure

```

mlpr-project/
│
├── data\_analysis/
│   ├── eda\_final\_dataset.ipynb
│   └── eda\_before\_australiandatasets.ipynb
│
├── model/
│   ├── lstm.ipynb
│   └── other\_models/
│       ├── randomforest.ipynb
│       ├── xgboost.ipynb
│       └── lightgbm.ipynb
│
├── scripts/
│   ├── preprocessing\_australian.ipynb
│   ├── preprocessing\_latest\_india.ipynb
│   └── preprosseing\_india.ipynb
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md

````

---

## 🧩 Main Components

- **`data_analysis/`**: Notebooks for exploratory data analysis (EDA) on the datasets.
- **`model/lstm.ipynb`**: LSTM (deep learning) model for time-series prediction of bleaching.
- **`model/other_models/`**: Notebooks for Random Forest, XGBoost, and LightGBM models.
- **`scripts/`**: Data preprocessing scripts for different datasets (Australia, India, etc.).
- **`requirements.txt`**: List of required Python packages.

---

## ⚙️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/shrijacked/Reefcast.git
````

### 2. Create and Activate a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

---

## 🚀 Usage

### 🔹 Data Preprocessing

Use the notebooks in the `scripts/` directory to preprocess raw datasets:

* `preprocessing_australian.ipynb`
* `preprocessing_latest_india.ipynb`
* `preprosseing_india.ipynb`

### 🔹 Exploratory Data Analysis

Explore the data using notebooks in `data_analysis/`:

* `eda_final_dataset.ipynb`
* `eda_before_australiandatasets.ipynb`

### 🔹 Model Training & Evaluation

Train and evaluate models using:

* `model/lstm.ipynb` for LSTM (deep learning)
* Notebooks in `model/other_models/` for Random Forest, XGBoost, and LightGBM

---

## 📄 License

This project is licensed under the terms of the [LICENSE](./LICENSE) file.

---
