# Evidently AI Data Monitoring

A Python project for exploring **data drift, data quality, and model performance monitoring** using Evidently AI and scikit-learn.

## Project Overview

This project generates synthetic reference and current datasets, evaluates them using Evidently, and trains a classification model to demonstrate model performance monitoring.

## Features

- **Data Drift Monitoring:** Compare reference and current datasets to identify changes in data distributions.
- **Data Quality Monitoring:** Generate a report summarizing data quality.
- **Model Performance Monitoring:** Train a Logistic Regression classifier and generate a classification performance report.
- **HTML Reports:** Save monitoring reports locally for interactive review.

## Tech Stack

- Python
- Evidently AI
- Pandas
- Scikit-learn

## Project Structure

```text
evidently-data-monitoring/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Generated Reports

Running `python app.py` generates the following HTML reports locally:

- `data_drift_report.html`
- `data_quality_report.html`
- `model_performance_report.html`
- `data_drift_quality_report.html`

These generated reports are not committed to the repository.
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/Gargik283/evidently-data-monitoring.git
cd evidently-data-monitoring
```

Replace `Gargik282` with your GitHub username.

### 2. Create and activate a virtual environment

On Windows:

```bash
py -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the project

```bash
python app.py
```

After execution, check the generated HTML reports in the project directory.

## Generated Reports

| Report | Purpose |
|---|---|
| `data_drift_report.html` | Examines changes between reference and current data |
| `data_quality_report.html` | Summarizes data quality |
| `model_performance_report.html` | Presents classification model performance |
| `data_drift_quality_report.html` | Combined drift and data quality report |

## Note

This project uses synthetic data to demonstrate monitoring concepts. It is a learning and portfolio project, not a production monitoring system.

## Author

**Gargi Kundu**

- GitHub: [Gargik283](https://github.com/Gargik283)
