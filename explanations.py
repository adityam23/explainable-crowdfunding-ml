#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.inspection import PartialDependenceDisplay
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from features import create_feature_pipeline, process_features
from models import CrowdfundingTrainer

FILENAME = "full_dataset.csv"

# ## Data Exploration
df = pd.read_csv(FILENAME, low_memory=False)

# #### Some columns can be dropped as they are not relevant/similar to others 
df = df.drop(columns=['id','name','location','converted_pledged_amount','usd_pledged'], errors='ignore')

# #### Full dataset is too big to process locally. We will randomly sample 10,000 records
df_sampled = df.sample(n=10000, random_state=42).reset_index(drop=True)

y = df_sampled["state"]
X_raw = df_sampled.drop(columns=["state"])

# ## Feature engineering

# Define columns
categorical = X_raw.select_dtypes(include=['object', 'category']).columns.tolist()
if 'blurb' in categorical: categorical.remove('blurb')

numeric = X_raw.select_dtypes(include=['int64', 'float64', 'bool']).columns.tolist()
other_text_cols = ['blurb_wc', 'dale_chall', 'flesch_kincaid', 'smog', 'gun_fog']

# Create preprocessor
preprocessor, tf_vectorizer = create_feature_pipeline(categorical, numeric)

# Process features
X_df = process_features(df_sampled, preprocessor, tf_vectorizer, is_train=True)

# Separate text-only features for Model 1
text_features = [col for col in X_df.columns if col in tf_vectorizer.get_feature_names_out()] + other_text_cols
X_text_df = X_df[text_features]

# ## Model training

X_train_text, X_test_text, y_train, y_test = train_test_split(X_text_df, y, test_size=0.2, stratify=y, random_state=42)

# ### MODEL 1 - Text only Model
trainer_text = CrowdfundingTrainer(model_type='xgb', params={
    'eval_metric': 'logloss', 
    'scale_pos_weight': 3,
    'max_depth': 16, 
    'learning_rate': 0.1, 
    'reg_alpha': 0.1, 
    'reg_lambda': 0.25
})

trainer_text.train(X_train_text, y_train)

# Evaluate
metrics_text = trainer_text.evaluate(X_test_text, y_test)
print(f"Text-only metrics: {metrics_text}")

# Explain model 1 (text-only)
explainer_text = shap.TreeExplainer(trainer_text.model)
shap_values_text = explainer_text.shap_values(X_test_text)
shap.summary_plot(shap_values_text, X_test_text)

# Partial Dependence
features_pd = ['dale_chall', 'blurb_wc', 'flesch_kincaid', 'gun_fog']  
PartialDependenceDisplay.from_estimator(
    trainer_text.model,
    X_test_text,
    features_pd,
    kind="average", 
    subsample=50,  
    grid_resolution=20,
    random_state=42
)
plt.tight_layout()
plt.show()

# ### MODEL 2 - Text + Features Model

# Remove spotlight (only present in successful)
if 'spotlight' in X_df.columns:
    X_df = X_df.drop(columns='spotlight')

# Remove social proof features
for col in ['pledged', 'backers_count']:
    if col in X_df.columns:
        X_df = X_df.drop(columns=col)

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_df, y, test_size=0.2, stratify=y, random_state=42)

trainer_all = CrowdfundingTrainer(model_type='xgb', params={
    'eval_metric': 'logloss', 
    'scale_pos_weight': 1,
    'max_depth': 8, 
    'learning_rate': 0.1, 
    'reg_alpha': 0.1, 
    'reg_lambda': 0.1
})

trainer_all.train(X_train_all, y_train_all)

# Evaluate
metrics_all = trainer_all.evaluate(X_test_all, y_test_all)
print(f"All features metrics: {metrics_all}")

# Explain model 2 (all features)
explainer_all = shap.TreeExplainer(trainer_all.model)
shap_values_all = explainer_all.shap_values(X_test_all)
shap.summary_plot(shap_values_all, X_test_all)

# ## Local Explanations using LIME

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train_all),
    feature_names=X_train_all.columns.tolist(),
    class_names=['fail', 'success'],
    mode='classification'
)

# Explain one instance
idx = 0
instance = X_test_all.iloc[idx].values
exp = explainer.explain_instance(
    data_row=instance,
    predict_fn=trainer_all.model.predict_proba
)
exp.save_to_file(f'Lime_{idx}.html')
print(f"LIME explanation saved to Lime_{idx}.html")

