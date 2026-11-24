"""Copy of Safetywashing's analysis.py (github.com/centerforaisafety/safetywashing).
Only two changes: data paths point at data/raw/safetywashing/repo/data/, and the
'PLOTTING DEMOS' section at the end is removed so it runs non-interactively."""
from pathlib import Path
SW_DATA = str(Path(__file__).resolve().parents[2] / "data" / "raw" / "safetywashing" / "repo" / "data") + "/"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Define capability benchmarks
cap_names = ["logiqa", "piqa", "hellaswag", "winogrande", 'superglue_copa', "medqa_4options", "arc_challenge", "mmlu", "minerva_math", "lambada_openai", "gsm8k", "bbh"]

def load_data():
    """
    Load data from CSV files and return as pandas DataFrames.
    """
    base_model_df = pd.read_csv(SW_DATA + 'benchmarks_base_models.csv', index_col='model')
    chat_model_df = pd.read_csv(SW_DATA + 'benchmarks_chat_models.csv', index_col='model')
    evals_df = pd.read_csv(SW_DATA + 'benchmarks_info.csv', index_col='benchmark')
    return base_model_df, chat_model_df, evals_df

def find_nans(model_df):
    """
    Identifies and prints the locations of NaNs in the dataframe.
    """
    df_filtered = model_df.drop(columns=["model_size", "FLOP", "name", "type"])

    missing_fraction = df_filtered.isnull().mean().mean()
    print(f"Fraction missing data: {missing_fraction}")

    missing_data = df_filtered.isnull()
    missing_indices = missing_data.stack()[missing_data.stack()].index

    for row, col in missing_indices:
        print(f"Missing data at row {row}, column {col}")

# Load data and print missing values
base_model_df, chat_model_df, evals_df = load_data()
find_nans(base_model_df)
find_nans(chat_model_df)

##### RUN ANALYSIS ON BASE AND CHAT MODELS #####

def run_analysis(model_df, evals_df, cap_names, label, correlation_type = "spearman"):
    """
    Runs normalization, computes and plots correlation matrix, performs PCA, and prints analysis results. Returns modified evals_df and model_df, and eigenvalues and correlation matrix from PCA.

    We compute the correlation of each safety benchmark individually, dropping the rows that contains NaNs for each safety benchmark (if any).
    """

    # Normalize data
    normalized_df = model_df.drop(columns=["model_size", "FLOP", "name", "type"])
    normalized_df = normalized_df.dropna(axis=1, how='all')
    normalized_df = (normalized_df - normalized_df.mean()) / normalized_df.std(ddof=0)
    normalized_df = normalized_df.dropna(axis=1, how='all') # In case one column has a std dev of zero and became NaN during normalization

    df_cap = normalized_df[cap_names].copy().dropna() # capabilities evals dataframe
    safety_names = [col for col in normalized_df.columns if col not in cap_names]
    df_safety = normalized_df[safety_names].copy() # "safety" evals dataframe

    # Compute capabilities correlation matrix & statistics of correlations between capabilities benchmarks
    cap_matrix = df_cap.corr(method=correlation_type).to_numpy()
    triu_indices = np.triu_indices_from(cap_matrix, k=1)
    upper_tri_values = cap_matrix[triu_indices]
    mean_correlation = np.mean(upper_tri_values)
    std_dev_correlation = np.std(upper_tri_values)

    # Compute capabilities scores
    eigenvals, eigenvecs = np.linalg.eigh(cap_matrix)
    pc = eigenvecs[:, -1]
    pc = pc if np.abs(pc).max() == pc.max() else -pc
    model_cap_score = df_cap.to_numpy() @ pc

    # Update dataframes with capabilities scores
    df_safety["cap_score"] = model_cap_score
    df_cap["cap_score"] = model_cap_score
    evals_df_copy = evals_df.copy()

    # Compute correlations for each evaluation and update dataframes
    for safety_name in safety_names:
        safety_task_df = df_safety[[safety_name]].dropna() # Drop NaNs
        score_df = df_safety[["cap_score"]].loc[safety_task_df.index]
        reduced_df = pd.concat([safety_task_df, score_df], axis=1) # Concatenate with capabilities scores
        corr_value = reduced_df.corr(method=correlation_type).to_numpy()[0, 1] # 2x2 matrix where we take entry (0,1)
        evals_df_copy.loc[safety_name, f"{label}_{correlation_type}_correlations"] = corr_value # populate correlation column

    for cap_name in cap_names:
        cap_task_df = df_cap[[cap_name]].dropna()
        score_df = df_cap[["cap_score"]].loc[cap_task_df.index]
        reduced_df = pd.concat([cap_task_df, score_df], axis=1)
        corr_value = reduced_df.corr(method=correlation_type).to_numpy()[0, 1]
        evals_df_copy.loc[cap_name, f"{label}_{correlation_type}_correlations"] = corr_value
    
    # Calculate total variance explained
    variance_explained_pc1 = eigenvals[-1] / np.sum(eigenvals)

    # Print analysis results
    print(f"\n***** Results for {label} *****")
    print(f"Mean correlation between two capabilities benchmarks: {mean_correlation*100:.1f}")
    print(f"Standard deviation of correlation between two capabilities benchmarks: {std_dev_correlation*100:.1f}")
    print(f"Total capabilities variance explained from PC1: {variance_explained_pc1*100:.1f}")

    print("\nCAPABILITIES CORRELATIONS:")
    for cap_name in cap_names:
        print(f"{cap_name} {100*evals_df_copy.loc[cap_name, f'{label}_{correlation_type}_correlations']:.2f}")

    print("\nSAFETY CORRELATIONS:")
    for safety_name in safety_names:
        print(f"{safety_name} {100*evals_df_copy.loc[safety_name, f'{label}_{correlation_type}_correlations']:.1f}")

    print("\nMODEL SCORES:")
    model_cap_dict = dict(sorted(list(zip(model_df.index, model_cap_score)), key=lambda item: item[1]))
    for k, v in model_cap_dict.items():
        print(f"{k}, {v:.2f}")

    # Update model dataframe with scores
    model_df_copy = model_df.copy()
    model_df_copy["score"] = model_cap_score

    return evals_df_copy, model_df_copy, eigenvals, cap_matrix

evals_df, base_model_df, base_eigenvals, base_cap_matrix = run_analysis(base_model_df, evals_df, cap_names, "Base", "spearman")
evals_df, chat_model_df, chat_eigenvals, chat_cap_matrix = run_analysis(chat_model_df, evals_df, cap_names, "Chat", "spearman")

