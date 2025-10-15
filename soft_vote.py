
import pandas as pd
import numpy as np

# Read the prediction results from different models
try:
    cnn_results = pd.read_csv("cnn_Nclass.out", sep="\t")
except:
    cnn_results = pd.DataFrame()

try:
    nn_results = pd.read_csv("nn_Nclass.out", sep="\t")
except:
    nn_results = pd.DataFrame()

try:
    xgb_results = pd.read_csv("xgb_Nclass.out", sep="\t")
except:
    xgb_results = pd.DataFrame()

# Combine results (soft voting)
if not cnn_results.empty and not nn_results.empty and not xgb_results.empty:
    # Assuming the results have the same structure
    f1 = cnn_results.copy()
    # Simple averaging for soft voting
    if 'prediction' in f1.columns:
        f1['prediction'] = (cnn_results['prediction'] + nn_results['prediction'] + xgb_results['prediction']) / 3
elif not cnn_results.empty:
    f1 = cnn_results.copy()
elif not nn_results.empty:
    f1 = nn_results.copy()
elif not xgb_results.empty:
    f1 = xgb_results.copy()
else:
    # Create empty dataframe if no results
    f1 = pd.DataFrame(columns=['prediction'])

# Replace numeric predictions with class names
if not f1.empty and 'prediction' in f1.columns:
    f1['prediction'] = f1['prediction'].astype(str)
    f1['prediction'].replace({"0":"thioredoxin reductase", "1":"cytochrome c peroxidase", "2":"peroxidase", "3":"glutathione peroxidase", "4":"nickel superoxide dismutase", "5":"alkyl hydroperoxide reductase", "6":"thioredoxin 1", "7":"thioredoxin 2", "8":"glutaredoxin 1", "9":"glutaredoxin 2", "10":"catalase", "11":"catalase-peroxidase", "12":"superoxide dismutase 2", "13":"superoxide dismutase 1", "14":"NADH peroxidase", "15":"superoxide reductase", "16":"Mn-containing catalase", "17":"monothiol glutaredoxin", "18":"thiol peroxidase", "19":"peroxiredoxin 5", "20":"peroxiredoxin 6", "21":"peroxiredoxin 1", "22":"alkyl hydroperoxide reductase 1", "23":"rubrerythrin", "24":"peroxiredoxin 3", "25":"glutaredoxin 3"}, inplace=True)

f1.to_csv("final_Nclass.out",sep="\t",index=False)
