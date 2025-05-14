
# ReefCast: Coral Bleaching Prediction in India Using Environmental and Ecological Data

This project aims to analyze and predict coral reef bleaching events using various machine learning models, including **LSTM** (deep learning), **Random Forest**, **XGBoost**, and **LightGBM**. The workflow includes **data preprocessing**, **exploratory data analysis (EDA)**, and **model training/evaluation**.

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
git clone https://github.com/your-username/mlpr-project.git
cd mlpr-project
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

## 📦 Requirements

See `requirements.txt` for the full list of dependencies. Main libraries include:

* `pandas`
* `numpy`
* `scikit-learn`
* `xgboost`
* `lightgbm`
* `matplotlib`
* `seaborn`
* `tensorflow`
* `scipy`
* `statsmodels`
* `jupyter`

---

## 📄 License

This project is licensed under the terms of the [LICENSE](./LICENSE) file.

---

```

Let me know if you'd like to add dataset details, results visualizations, or citations for your models or sources!
```
