import pandas as pd 
import numpy as np 

np.random.seed(42) 

# Reference(training-like) -> no drift 
ref = pd.DataFrame({ 
    "age": np.random.normal(30, 5, 2000).astype(int), 
    "income": np.random.normal(50000, 12000, 2000), 
    "city": np.random.choice(["A", "B", "C"], 2000, p=[0.5, 0.3, 0.2]) 
}) 
ref["target"] = (ref["income"] > 52000).astype(int) 

# Current(production-like) -> drift added 
cur = pd.DataFrame({ 
    "age": np.random.normal(35, 6, 2000).astype(int), # age shift 
    "income": np.random.normal(55000, 15000, 2000), # income shift 
    "city": np.random.choice( ["A", "B", "C", "D"], 2000, p=[0.4, 0.3, 0.2, 0.1] ) # new category D added 
}) 
cur["target"] = (cur["income"] > 52000).astype(int) 

from evidently import Report 
from evidently.presets import DataDriftPreset

# Initialize the report layout
report = Report(metrics=[DataDriftPreset()]) 

# FIX: Capture the evaluation output returned by .run()
my_eval = report.run(reference_data=ref, current_data=cur) 

# FIX: Save using the evaluation snapshot object
my_eval.save_html("data_drift_report.html") 

print("Data drift report generated and saved as 'data_drift_report.html'.")


from evidently.presets import DataSummaryPreset

quality_report = Report(metrics=[DataSummaryPreset()])
# FIX: Capture the evaluation output returned by .run()

quality_eval = quality_report.run(reference_data=ref, current_data=cur)

# FIX: Save using the evaluation snapshot object
quality_eval.save_html("data_quality_report.html")
print("Data quality report generated and saved as 'data_quality_report.html'.")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X_ref = ref[["age", "income"]]
y_ref = ref["target"]

model = LogisticRegression()
model.fit(X_ref, y_ref)

ref["predictions"] = model.predict_proba(X_ref)[:, 1]
cur["predictions"] = model.predict_proba(cur[["age", "income"]])[:, 1]

# FIX: Import modern DataDefinition structures instead of legacy ColumnMapping
from evidently import DataDefinition, BinaryClassification, Dataset
from evidently.presets import ClassificationPreset

# FIX: Map column configuration using the latest API guidelines
data_definition = DataDefinition(
    classification=[
        BinaryClassification(
            target="target",
            prediction_probas="predictions",
            pos_label=1
        )
    ]
)

# Convert your dataframes into the required Dataset wrapper objects
ref_dataset = Dataset.from_pandas(ref, data_definition=data_definition)
cur_dataset = Dataset.from_pandas(cur, data_definition=data_definition)

performance_report = Report(metrics=[ClassificationPreset()])

# FIX: Pass the wrapped datasets straight into your .run method
performance_eval = performance_report.run(reference_data=ref_dataset, current_data=cur_dataset)
performance_eval.save_html("model_performance_report.html")
print("Model performance report generated and saved as 'model_performance_report.html'.")


from evidently import Report
from evidently.presets import DataDriftPreset

report = Report(
    metrics=[
        DataDriftPreset()
    ],
    include_tests=True
)

evaluation = report.run(
    reference_data=ref,
    current_data=cur
)

evaluation.save_html("data_drift_quality_report.html")

print("Data drift and quality report generated and saved as 'data_drift_quality_report.html'.")

