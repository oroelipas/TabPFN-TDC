import json
import pandas as pd
import numpy as np
np.random.seed(42)

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
ckpt_path = os.path.join(current_dir, "tabpfn_ckpts")
os.environ["TABPFN_MODEL_CACHE_DIR"] = ckpt_path

from rdkit import Chem
from rdkit import RDLogger
from rdkit.ML.Descriptors import MoleculeDescriptors
from tabpfn import TabPFNRegressor, TabPFNClassifier

# Optionally, if you want to test RDKit + XGBoost
#import xgboost as xgb

MAX_TABPFN_INFERENCE_SIZE = 10000


def clean_fragments(mol):
    """
    Removes disconnected fragments (ions or other unconnected atoms)
    returns the largest fragment (presumably the main molecule).
    """
    # Split the molecule into disconnected fragments
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)

    # Select the largest fragment (by number of atoms)
    return max(fragments, key=lambda m: m.GetNumAtoms())


def get_rdkit_x_y(data):
    """
    Computes RDKit molecular descriptors for a dataset and returns features (X) and labels (Y).

    For each SMILES string in the 'Drug' column of the input DataFrame, this function:
      - Converts the SMILES to an RDKit molecule.
      - Cleans disconnected fragments, keeping only the largest one.
      - Calculates 217 molecular descriptors using RDKit.
      - Handles infinite descriptor values by replacing them with NaN.
      - Returns a DataFrame of descriptors (X) and the corresponding target values (Y).

    Parameters
    ----------
    data : pd.DataFrame
        A pandas DataFrame with at least two columns:
          - 'Drug': SMILES strings.
          - 'Y': Target values (labels or regression targets).

    Returns
    -------
    X : pd.DataFrame
        DataFrame containing RDKit molecular descriptors for each compound.
    Y : pd.Series
        Target values corresponding to each compound.
    """
    # Disable RDKit warnings. Too dirty: It raises a warning every time the SMILES has more than 1 fragment
    RDLogger.DisableLog('rdApp.*')

    desc_names = [name for name, _ in Chem.Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    data_rdkit = []

    for i in range(data.shape[0]):
        smiles = data.iloc[i]["Drug"]
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            print(f"Warning: Invalid SMILES '{smiles}' at index {i}. Filling with NaNs.")
            features = [np.nan] * len(desc_names)
        else:
            mol = clean_fragments(mol)
            features = calculator.CalcDescriptors(mol)

            if np.any(np.isinf(features)):
                # Replace inf or -inf with NaN (It happens in solubility_aqsoldb trainset)
                print(f"Warning: Infinite value in descriptor for SMILES '{smiles}' at index {i}. Replacing with NaN.")
                features = [np.nan if np.isinf(f) else f for f in features]

        data_rdkit.append(features)

    RDLogger.EnableLog('rdApp.*')

    return pd.DataFrame(data_rdkit, columns=desc_names), data["Y"]


def eval_TabPFN(x_train, y_train, x_test, seed=42, n_estimators=8, finetuned=True):

    if x_train.shape[0] > MAX_TABPFN_INFERENCE_SIZE:
        # Randomly sample the training set if it's too large
        print(f"Reducing training set size from {x_train.shape[0]} to {MAX_TABPFN_INFERENCE_SIZE} samples for TabPFN.")
        indices = np.random.choice(x_train.index, size=MAX_TABPFN_INFERENCE_SIZE, replace=False)
        x_train = x_train.loc[indices]
        y_train = y_train.loc[indices]

    if np.unique(y_train).size == 2:  # Binary classification
        if finetuned:
            model = TabPFNClassifier(
                n_estimators=n_estimators,
                random_state=seed,
                model_path=os.path.join(ckpt_path, 'tabpfn-v2-classifier-finetuned-zk73skhh.ckpt')
            )
        else:
            model = TabPFNClassifier(n_estimators=n_estimators, random_state=seed)

        model.fit(x_train, y_train)
        return model.predict_proba(x_test)[:, 1]

    else:  # Regression
        model = TabPFNRegressor(n_estimators=n_estimators, random_state=seed)
        model.fit(x_train, y_train)
        return model.predict(x_test)


def to_float32(x):
    # XGBoost can not handle > float32 values, so we clip them
    x = x.astype(np.float32)
    th = np.finfo(np.float32).max  # Limite del float32
    if np.any(x > th):
        print(f"Warning: Values greater than 3.4e38 found in x. Clipping to 3.4e38.")
        # replace values greater than th with th
        x[x > th] = th
    return x


def eval_XGBoost(x_train, y_train, x_test, seed):

    x_train = to_float32(x_train)
    x_test = to_float32(x_test)

    if np.unique(y_train).size == 2: # classification
        model = xgb.XGBClassifier(eval_metric='logloss', random_state=seed, tree_method='hist')
        model.fit(x_train, y_train)
        return model.predict_proba(x_test)[:, 1]
    
    else:  # regression
        model = xgb.XGBRegressor(eval_metric='rmse', random_state=seed, tree_method='hist')
        model.fit(x_train, y_train)
        return model.predict(x_test)


def save_json(data, filename='results.json'):
    import os
    if not os.path.exists('results'):
        os.makedirs('results')

    with open(os.path.join('results', filename), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    

def load_json(filename):
    with open('results/' +  filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
