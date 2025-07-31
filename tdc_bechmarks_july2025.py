import numpy as np
import pandas as pd

benchmark_results = {
    'caco2_wang': {
        'metric': 'MAE (↓)',
        'task': 'regression',
        'leaderboard':{
            'CaliciBoost': [0.256, 0.006],
            'XG Boost': [0.274, 0.004],
            'MapLight': [0.276, 0.005],
            'BaseBoosting': [0.285, 0.005],
            'MolMapNet-D': [0.287, 0.005],
            'MapLight + GNN': [0.287, 0.005],
            'XGBoost': [0.289, 0.011],
            'DeepMol (AutoML)': [0.297, 0.008],
            'MolE': [0.310, 0.010],
            'Basic ML': [0.321, 0.005],
            'ADMETrix': [0.326, 0.042],
            'Chemprop-RDKit': [0.330, 0.024],
            'CFA': [0.335, 0.033],
            'Euclia ML model': [0.341, 0.004],
            'Chemprop': [0.344, 0.015],
            'MiniMol': [0.350, 0.018],
            'RDKit2D + MLP (DeepPurpose)': [0.393, 0.024],
            'AttentiveFP': [0.401, 0.032],
            'CNN (DeepPurpose)': [0.446, 0.036],
            'ContextPred': [0.502, 0.036],
            'NeuralFP': [0.530, 0.102],
            'AttrMasking': [0.546, 0.052],
            'GCN': [0.599, 0.104],
            'Morgan + MLP (DeepPurpose)': [0.908, 0.060],
            'admet_ai_v2': [6.328, 0.101]
        }
    },
    'bioavailability_ma': {
        'metric': 'AUROC (↑)',
        'task': 'classification',
        'leaderboard':{
            'MiniMol': [0.942, 0.002], # Original paper reports 0.699, I take the value from the Leaderboard
            'MapLight + GNN': [0.938, 0.002],
            'ZairaChem': [0.935, 0.006],
            'MapLight': [0.930, 0.002],
            'SimGCN': [0.929, 0.010],
            'AttrMasking': [0.929, 0.006],
            'CFA': [0.928, 0.008],
            'ADMETrix': [0.926, 0.008],
            'ContextPred': [0.923, 0.005],
            'DeepMol (AutoML)': [0.922, 0.000],
            'RDKit2D + MLP (DeepPurpose)': [0.918, 0.007],
            'CNN (DeepPurpose)': [0.908, 0.012],
            'NeuralFP': [0.902, 0.020],
            'GCN': [0.895, 0.021],
            'AttentiveFP': [0.892, 0.012],
            'Chemprop-RDKit': [0.886, 0.016],
            'Morgan + MLP (DeepPurpose)': [0.880, 0.006],
            'Chemprop': [0.860, 0.036],
            'Euclia ML model': [0.845, 0.003],
            'Basic ML': [0.818, 0.000],
            'MolE': [0.654, 0.028]  # Extracted from the paper. No leaderboard submission
        }
    },
    'lipophilicity_astrazeneca': {
        'metric': 'MAE (↓)',
        'task': 'regression',
        'leaderboard':{
            'MiniMol': [0.456, 0.008],
            'Chemprop-RDKit': [0.467, 0.006],
            'MolE': [0.469, 0.009],
            'Chemprop': [0.470, 0.009],
            'BaseBoosting KyQVZ6b2': [0.479, 0.007],
            'ADMETrix': [0.512, 0.010],
            'CMPNN': [0.515, 0.008],
            'MapLight + GNN': [0.525, 0.003],
            'ContextPred': [0.535, 0.012],
            'StackingRegressor (DeepMol)': [0.538, 0.001],
            'MapLight': [0.539, 0.002],
            'GCN': [0.541, 0.011],
            'AttrMasking': [0.547, 0.024],
            'NeuralFP': [0.563, 0.023],
            'AttentiveFP': [0.572, 0.007],
            'RDKit2D + MLP (DeepPurpose)': [0.574, 0.017],
            'Basic ML': [0.617, 0.003],
            'Euclia ML model': [0.621, 0.005],
            'CFA': [0.626, 0.013],
            'DeepMol (AutoML)': [0.656, 0.012],
            'Morgan + MLP (DeepPurpose)': [0.701, 0.009],
            'CNN (DeepPurpose)': [0.743, 0.020]
        }
    },
    'solubility_aqsoldb':{
        'metric': 'MAE (↓)',
        'task': 'regression',
        'leaderboard':{
            'MiniMol': [0.741, 0.013],
            'Chemprop-RDKit': [0.761, 0.025],
            'DeepMol (AutoML)': [0.775, 0.006],
            'AttentiveFP': [0.776, 0.008],
            'MapLight + GNN': [0.789, 0.003],
            'MapLight': [0.792, 0.002],
            'MolE': [0.792, 0.005],
            'CMPNN': [0.796, 0.038],
            'RDKit2D + MLP (DeepPurpose)': [0.827, 0.047],
            'Basic ML': [0.828, 0.002],
            'Chemprop': [0.829, 0.022],
            'GCN': [0.907, 0.020],
            'CFA': [0.939, 0.030],
            'NeuralFP': [0.947, 0.016],
            'CNN (DeepPurpose)': [1.023, 0.023],
            'AttrMasking': [1.026, 0.020],
            'ContextPred': [1.040, 0.045],
            'Euclia ML model': [1.076, 0.016],
            'Morgan + MLP (DeepPurpose)': [1.203, 0.019]
        }
    },
    'hia_hou':{
        'metric': 'AUROC (↑)',
        'task': 'classification',
        'leaderboard':{
            'MiniMol': [0.993, 0.005],
            'DeepMol (AutoML)': [0.990, 0.002],
            'MapLight + GNN': [0.989, 0.001],
            'RFStacker': [0.988, 0.002],
            'MapLight': [0.986, 0.000],
            'ADMETrix': [0.985, 0.003],
            'Chemprop-RDKit': [0.981, 0.002],
            'CFA': [0.981, 0.009],
            'AttrMasking': [0.978, 0.006],
            'ContextPred': [0.975, 0.004],
            'AttentiveFP': [0.974, 0.007],
            'RDKit2D + MLP (DeepPurpose)': [0.972, 0.008],
            'Chemprop': [0.965, 0.005],
            'MolE': [0.963, 0.019],
            'ZairaChem': [0.948, 0.018],
            'NeuralFP': [0.943, 0.014],
            'GCN': [0.936, 0.024],
            'Euclia ML model': [0.926, 0.017],
            'CNN (DeepPurpose)': [0.869, 0.026],
            'Basic ML': [0.818, 0.010],
            'Morgan + MLP (DeepPurpose)': [0.807, 0.072]
        }
    },
    'pgp_broccatelli': {
        'metric': 'AUROC (↑)',
        'task': 'classification',
        'leaderboard': {
            'MiniMol': [0.942, 0.002], # This value was not in the Leaderboard. Extracted from the GitHub. Paper reports 0.99 but I think it's a typo.
            'MapLight + GNN': [0.938, 0.002],
            'ZairaChem': [0.935, 0.006],
            'MapLight': [0.930, 0.002],
            'SimGCN': [0.929, 0.010],
            'AttrMasking': [0.929, 0.006],
            'ContextPred': [0.923, 0.005],
            'RDKit2D + MLP (DeepPurpose)': [0.918, 0.007],
            'MolE': [0.915, 0.005],
            'CNN (DeepPurpose)': [0.908, 0.012],
            'NeuralFP': [0.902, 0.020],
            'GCN': [0.895, 0.021],
            'AttentiveFP': [0.892, 0.012],
            'Chemprop-RDKit': [0.886, 0.016],
            'Morgan + MLP (DeepPurpose)': [0.880, 0.006],
            'Chemprop': [0.860, 0.036],
            'Euclia ML model': [0.845, 0.003],
            'Basic ML': [0.818, 0.000]
        }
    },
    'bbb_martins': {
        'metric': 'AUROC (↑)',
        'task': 'classification',
        'leaderboard': {
            'MiniMol': [0.924, 0.003],
            'CFA': [0.920, 0.006],
            'MapLight': [0.916, 0.001],
            'Lantern RADR Ensemble': [0.915, 0.002],
            'MapLight + GNN': [0.913, 0.001],
            'Lantern RADR Deep Neural Network': [0.912, 0.003],
            'ZairaChem': [0.910, 0.024],
            'Lantern RADR Random Forest': [0.908, 0.002],
            'ADMETrix': [0.906, 0.035],
            'Lantern RADR SVM': [0.905, 0.007],
            'Lantern RADR Logistic Regression': [0.903, 0.002],
            'MolE': [0.903, 0.005],
            'SimGCN': [0.901, 0.007],
            'ContextPred': [0.897, 0.004],
            'AttrMasking': [0.892, 0.012],
            'CMPNN': [0.890, 0.016],
            'RDKit2D + MLP (DeepPurpose)': [0.889, 0.016],
            'Chemprop-RDKit': [0.869, 0.027],
            'DeepMol (AutoML)': [0.868, 0.007],
            'AttentiveFP': [0.855, 0.011],
            'GCN': [0.842, 0.016],
            'NeuralFP': [0.836, 0.009],
            'Morgan + MLP (DeepPurpose)': [0.823, 0.015],
            'Chemprop': [0.821, 0.112],
            'Basic ML': [0.811, 0.013],
            'CNN (DeepPurpose)': [0.781, 0.030],
            'Euclia ML model': [0.725, 0.019]
        }
    },
    'ppbr_az': {
        'metric': 'MAE (↓)',
        'task': 'regression',
        'leaderboard': {
            'Gradient Boost': [7.440, 0.024],
            'MapLight + GNN': [7.526, 0.106],
            'MapLight': [7.660, 0.058],
            'MiniMol': [7.696, 0.125],
            'Chemprop': [7.788, 0.210],
            'BaseBoosting KyQVZ6b2': [7.914, 0.096],
            'DeepMol (AutoML)': [7.990, 0.104],
            'MolE': [8.073, 0.335],
            'ADMETrix': [8.200, 0.114],
            'Chemprop-RDKit': [8.288, 0.173],
            'CFA': [8.680, 0.262],
            'Basic ML': [9.185, 0.000],
            'NeuralFP': [9.292, 0.384],
            'AttentiveFP': [9.373, 0.335],
            'ContextPred': [9.445, 0.224],
            'Euclia ML model': [9.942, 0.121],
            'RDKit2D + MLP (DeepPurpose)': [9.994, 0.319],
            'AttrMasking': [10.075, 0.202],
            'GCN': [10.194, 0.373],
            'CNN (DeepPurpose)': [11.106, 0.358],
            'Morgan + MLP (DeepPurpose)': [12.848, 0.362]
        }
    },
    'vdss_lombardo': {
        'metric': 'Spearman (↑)',
        'task': 'regression',
        'leaderboard': {
            'MapLight + GNN': [0.713, 0.007],
            'MapLight': [0.707, 0.009],
            'MolE': [0.654, 0.031],
            'CFA': [0.628, 0.023],
            'Basic ML': [0.627, 0.010],
            'Euclia ML model': [0.609, 0.014],
            'SimGCN': [0.582, 0.031],
            'RDKit2D + MLP (DeepPurpose)': [0.561, 0.025],
            'AttrMasking': [0.559, 0.019],
            'MiniMol': [0.535, 0.027],
            'DeepMol (AutoML)': [0.497, 0.011],
            'Morgan + MLP (DeepPurpose)': [0.493, 0.011],
            'Chemprop': [0.491, 0.046],
            'ContextPred': [0.485, 0.092],
            'ADMETrix': [0.475, 0.022],
            'GCN': [0.457, 0.050],
            'Chemprop-RDKit': [0.389, 0.075],
            'NeuralFP': [0.258, 0.162],
            'AttentiveFP': [0.241, 0.145],
            'CNN (DeepPurpose)': [0.226, 0.114]
        }
    },
    'cyp2c9_veith': {
        'metric': 'AUPRC (↑)',
        'task': 'classification',
        'leaderboard': {
            'MapLight + GNN': [0.859, 0.001],
            'ContextPred': [0.839, 0.003],
            'AttrMasking': [0.829, 0.003],
            'MiniMol': [0.823, 0.006],
            'MolE': [0.801, 0.003],
            'ADMETrix': [0.789, 0.004],
            'ZairaChem': [0.786, 0.004],
            'MapLight': [0.783, 0.002],
            'Chemprop-RDKit': [0.777, 0.003],
            'ColorRefinement + Weighted Ensemble LGBM': [0.767, 0.003],
            'DeepMol (AutoML)': [0.758, 0.002],
            'Chemprop': [0.754, 0.002],
            'CFA': [0.751, 0.006],
            'AttentiveFP': [0.749, 0.004],
            'RDKit2D + MLP (DeepPurpose)': [0.742, 0.006],
            'NeuralFP': [0.739, 0.010],
            'GCN': [0.735, 0.004],
            'Morgan + MLP (DeepPurpose)': [0.715, 0.004],
            'CNN (DeepPurpose)': [0.713, 0.006],
            'Basic ML': [0.556, 0.000],
            'Euclia ML model': [0.536, 0.003]
        }
    },
    'cyp2d6_veith': {
        'metric': 'AUPRC (↑)',
        'task': 'classification',
        'leaderboard': {
            'MapLight + GNN': [0.790, 0.001],
            'ContextPred': [0.739, 0.005],
            'MapLight': [0.723, 0.003],
            'AttrMasking': [0.721, 0.009],
            'MiniMol': [0.719, 0.004],
            'ADMETrix': [0.718, 0.006],
            'DeepMol (AutoML)': [0.685, 0.000],
            'MolE': [0.682, 0.008],
            'Chemprop-RDKit': [0.673, 0.007],
            'CFA': [0.664, 0.012],
            'Chemprop': [0.649, 0.016],
            'AttentiveFP': [0.646, 0.014],
            'ZairaChem': [0.644, 0.085],
            'NeuralFP': [0.627, 0.009],
            'GCN': [0.616, 0.020],
            'RDKit2D + MLP (DeepPurpose)': [0.616, 0.007],
            'Morgan + MLP (DeepPurpose)': [0.587, 0.011],
            'CNN (DeepPurpose)': [0.544, 0.053],
            'Basic ML': [0.358, 0.000],
            'Euclia ML model': [0.348, 0.004]
        }
    },
    'cyp3a4_veith': {
        'metric': 'AUPRC (↑)',
        'task': 'classification',
        'leaderboard': {
            'MapLight + GNN': [0.916, 0.000],
            'ContextPred': [0.904, 0.002],
            'AttrMasking': [0.902, 0.002],
            'ADMETrix': [0.884, 0.001],
            'MapLight': [0.881, 0.001],
            'MiniMol': [0.877, 0.001],
            'Chemprop-RDKit': [0.876, 0.003],
            'ZairaChem': [0.875, 0.002],
            'DeepMol (AutoML)': [0.867, 0.002],
            'MolE': [0.867, 0.003],
            'Chemprop': [0.862, 0.003],
            'CFA': [0.855, 0.004],
            'AttentiveFP': [0.851, 0.006],
            'NeuralFP': [0.849, 0.004],
            'GCN': [0.840, 0.010],
            'RDKit2D + MLP (DeepPurpose)': [0.829, 0.007],
            'Morgan + MLP (DeepPurpose)': [0.827, 0.009],
            'CNN (DeepPurpose)': [0.821, 0.003],
            'Euclia ML model': [0.696, 0.004],
            'Basic ML': [0.654, 0.000]
        }
    },
    'cyp2d6_substrate_carbonmangels': {
        'metric': 'AUPRC (↑)',
        'task': 'classification',
        'leaderboard': {
            'ContextPred': [0.736, 0.024],
            'DeepMol (AutoML)': [0.731, 0.037],
            'MapLight + GNN': [0.720, 0.002],
            'MapLight': [0.713, 0.009],
            'CFA': [0.704, 0.015],
            'AttrMasking': [0.704, 0.028],
            'MolE': [0.699, 0.018],
            'MiniMol': [0.695, 0.032],
            'Chemprop-RDKit': [0.686, 0.031],
            'ZairaChem': [0.685, 0.029],
            'RDKit2D + MLP (DeepPurpose)': [0.677, 0.047],
            'Morgan + MLP (DeepPurpose)': [0.671, 0.066],
            'Chemprop': [0.632, 0.037],
            'GCN': [0.617, 0.039],
            'AttentiveFP': [0.574, 0.030],
            'NeuralFP': [0.572, 0.062],
            'Euclia ML model': [0.498, 0.015],
            'CNN (DeepPurpose)': [0.485, 0.037],
            'Basic ML': [0.478, 0.018]
        }
    },
    'cyp3a4_substrate_carbonmangels':{
        'metric': 'AUROC (↑)',
        'task': 'classification',
        'leaderboard': {
            'MolE': [0.670, 0.018],
            'CFA': [0.667, 0.019],
            'MiniMol': [0.663, 0.008],
            'CNN (DeepPurpose)': [0.662, 0.031],
            'DeepMol (AutoML)': [0.655, 0.003],
            'MapLight': [0.650, 0.006],
            'MapLight + GNN': [0.647, 0.008],
            'SimGCN': [0.640, 0.016],
            'RDKit2D + MLP (DeepPurpose)': [0.639, 0.012],
            'Morgan + MLP (DeepPurpose)': [0.633, 0.013],
            'ZairaChem': [0.630, 0.008],
            'Euclia ML model': [0.629, 0.027],
            'Chemprop-RDKit': [0.619, 0.030],
            'ContextPred': [0.609, 0.025],
            'Basic ML': [0.605, 0.000],
            'Chemprop': [0.596, 0.018],
            'GCN': [0.590, 0.023],
            'AttrMasking': [0.582, 0.021],
            'NeuralFP': [0.578, 0.020],
            'AttentiveFP': [0.576, 0.025]
        }
    },
    'cyp2c9_substrate_carbonmangels':{
        'metric': 'AUPRC (↑)',
        'task': 'classification',
        'leaderboard': {
            'MiniMol': [0.474, 0.025],
            'MolE': [0.446, 0.062],
            'ZairaChem': [0.441, 0.033],
            'MapLight + GNN': [0.437, 0.008],
            'Random Forest': [0.437, 0.022],
            'SimGCN': [0.433, 0.017],
            'CFA': [0.417, 0.010],
            'DeepMol (AutoML)': [0.417, 0.005],
            'MapLight': [0.415, 0.008],
            'Chemprop-RDKit': [0.400, 0.008],
            'ContextPred': [0.392, 0.026],
            'Chemprop': [0.382, 0.019],
            'AttrMasking': [0.381, 0.045],
            'Morgan + MLP (DeepPurpose)': [0.380, 0.015],
            'AttentiveFP': [0.375, 0.032],
            'CNN (DeepPurpose)': [0.367, 0.059],
            'RDKit2D + MLP (DeepPurpose)': [0.360, 0.040],
            'NeuralFP': [0.359, 0.059],
            'Euclia ML model': [0.347, 0.018],
            'GCN': [0.344, 0.051],
            'Basic ML': [0.281, 0.000]
        }
    },
    'half_life_obach': {
        'metric': 'Spearman (↑)',
        'task': 'regression',
        'leaderboard': {
            'CFA': [0.576, 0.025],
            'MapLight': [0.562, 0.008],
            'MapLight + GNN': [0.557, 0.034],
            'MolE': [0.549, 0.024],
            'Euclia ML model': [0.547, 0.032],
            'Voting Regressor (KNN, SVM)': [0.544, 0.034],
            'MiniMol': [0.495, 0.042],
            'DeepMol (AutoML)': [0.485, 0.039],
            'Basic ML': [0.438, 0.011],
            'SimGCN': [0.392, 0.065],
            'ADMETrix': [0.372, 0.026],
            'Morgan + MLP (DeepPurpose)': [0.329, 0.083],
            'Chemprop': [0.265, 0.032],
            'Chemprop-RDKit': [0.239, 0.019],
            'GCN': [0.239, 0.100],
            'RDKit2D + MLP (DeepPurpose)': [0.184, 0.111],
            'NeuralFP': [0.177, 0.165],
            'AttrMasking': [0.151, 0.068],
            'ContextPred': [0.129, 0.114],
            'AttentiveFP': [0.085, 0.068],
            'CNN (DeepPurpose)': [0.038, 0.138]
        }
    },
    'clearance_microsome_az': {
        'metric': 'Spearman (↑)',
        'task': 'regression',
        'leaderboard': {
            'MapLight + GNN': [0.630, 0.010],
            'MiniMol': [0.628, 0.005],
            'MapLight': [0.626, 0.008],
            'CFA': [0.625, 0.012],
            'RFStacker': [0.625, 0.002],
            'MolE': [0.607, 0.027],
            'Chemprop-RDKit': [0.599, 0.025],
            'SimGCN': [0.597, 0.025],
            'RDKit2D + MLP (DeepPurpose)': [0.586, 0.014],
            'AttrMasking': [0.585, 0.034],
            'ContextPred': [0.578, 0.007],
            'Euclia ML model': [0.572, 0.010],
            'ADMETrix': [0.556, 0.015],
            'Chemprop': [0.555, 0.022],
            'DeepMol (AutoML)': [0.553, 0.013],
            'GCN': [0.532, 0.033],
            'NeuralFP': [0.529, 0.015],
            'Basic ML': [0.518, 0.005],
            'Morgan + MLP (DeepPurpose)': [0.492, 0.020],
            'AttentiveFP': [0.365, 0.055],
            'CNN (DeepPurpose)': [0.252, 0.116]
        }    
    },
    'clearance_hepatocyte_az': {
        'metric': 'Spearman (↑)',
        'task': 'regression',
        'leaderboard': {
            'CFA': [0.536, 0.020],
            'MapLight + GNN': [0.498, 0.009],
            'MapLight': [0.466, 0.012],
            'ADMETrix': [0.447, 0.028],
            'MiniMol': [0.446, 0.029],
            'Basic ML': [0.440, 0.003],
            'DeepMol (AutoML)': [0.440, 0.011],
            'ContextPred': [0.439, 0.026],
            'Chemprop': [0.431, 0.006],
            'Chemprop-RDKit': [0.430, 0.021],
            'Euclia ML model': [0.424, 0.008],
            'AttrMasking': [0.413, 0.028],
            'NeuralFP': [0.401, 0.037],
            'RDKit2D + MLP (DeepPurpose)': [0.382, 0.007],
            'MolE': [0.381, 0.038],
            'GCN': [0.366, 0.063],
            'AttentiveFP': [0.289, 0.022],
            'Morgan + MLP (DeepPurpose)': [0.272, 0.068],
            'CNN (DeepPurpose)': [0.235, 0.021]
        }
    },
    'herg': {
        'metric': 'AUROC (↑)',
        'task': 'classification',
        'leaderboard': {
            'MapLight + GNN': [0.880, 0.002],
            'CFA': [0.875, 0.014],
            'SimGCN': [0.874, 0.014],
            'MapLight': [0.871, 0.004],
            'ZairaChem': [0.856, 0.009],
            'MiniMol': [0.846, 0.016],
            'RDKit2D + MLP (DeepPurpose)': [0.841, 0.020],
            'Chemprop-RDKit': [0.840, 0.007],
            'ADMETrix': [0.836, 0.025],
            'AttentiveFP': [0.825, 0.007],
            'MolE': [0.813, 0.009],
            'AttrMasking': [0.778, 0.046],
            'DeepMol (AutoML)': [0.763, 0.015],
            'ContextPred': [0.756, 0.023],
            'CNN (DeepPurpose)': [0.754, 0.037],
            'Euclia ML model': [0.749, 0.032],
            'GCN': [0.738, 0.038],
            'Morgan + MLP (DeepPurpose)': [0.736, 0.023],
            'NeuralFP': [0.722, 0.034],
            'Chemprop': [0.721, 0.045],
            'Basic ML': [0.715, 0.011]
        }
    },
    'ames': {
        'metric': 'AUROC (↑)',
        'task': 'classification',
        'leaderboard': {
            'MolE': [0.883, 0.005],
            'ZairaChem': [0.871, 0.002],
            'ADMETrix': [0.870, 0.006],
            'MapLight + GNN': [0.869, 0.002],
            'MapLight': [0.868, 0.002],
            'CFA': [0.852, 0.005],
            'Chemprop-RDKit': [0.850, 0.004],
            'MiniMol': [0.849, 0.004],
            'DeepMol (AutoML)': [0.847, 0.007],
            'CMPNN': [0.843, 0.009],
            'Chemprop': [0.842, 0.014],
            'AttrMasking': [0.842, 0.008],
            'ContextPred': [0.837, 0.009],
            'RDKit2D + MLP (DeepPurpose)': [0.823, 0.011],
            'NeuralFP': [0.823, 0.006],
            'GCN': [0.818, 0.010],
            'AttentiveFP': [0.814, 0.008],
            'Morgan + MLP (DeepPurpose)': [0.794, 0.008],
            'CNN (DeepPurpose)': [0.776, 0.015],
            'Euclia ML model': [0.755, 0.003],
            'Basic ML': [0.716, 0.000]
        }
    },
    'dili': { 
        'metric': 'AUROC (↑)',
        'task': 'classification',
        'leaderboard': {
            'MiniMol': [0.956, 0.006],
            'ZairaChem': [0.925, 0.005],
            'AttrMasking': [0.919, 0.008],
            'CFA': [0.919, 0.014],
            'MapLight + GNN': [0.917, 0.005],
            'SimGCN': [0.909, 0.011],
            'ADMETrix': [0.906, 0.016],
            'Chemprop': [0.899, 0.008],
            'Chemprop-RDKit': [0.887, 0.011],
            'MapLight': [0.887, 0.006],
            'AttentiveFP': [0.886, 0.015],
            'DeepMol (AutoML)': [0.885, 0.014],
            'RDKit2D + MLP (DeepPurpose)': [0.875, 0.019],
            'Euclia ML model': [0.873, 0.024],
            'ContextPred': [0.861, 0.018],
            'GCN': [0.859, 0.033],
            'NeuralFP': [0.851, 0.026],
            'Morgan + MLP (DeepPurpose)': [0.832, 0.021],
            'CNN (DeepPurpose)': [0.792, 0.016],
            'Basic ML': [0.700, 0.000],
            'MolE': [0.577, 0.021]
        }
    },
    'ld50_zhu': {
        'metric': 'MAE (↓)',
        'task': 'regression',
        'leaderboard': {
            'BaseBoosting KyQVZ6b2': [0.552, 0.009],
            'ADMETrix': [0.573, 0.010],
            'MiniMol': [0.585, 0.008],
            'MACCS keys + autoML': [0.588, 0.005],
            'Chemprop': [0.606, 0.024],
            'DeepMol (AutoML)': [0.614, 0.004],
            'MapLight': [0.621, 0.003],
            'QuGIN': [0.622, 0.015],
            'Chemprop-RDKit': [0.625, 0.022],
            'CFA': [0.630, 0.012],
            'CMPNN': [0.631, 0.021],
            'MapLight + GNN': [0.633, 0.003],
            'Basic ML': [0.636, 0.001],
            'Euclia ML model': [0.646, 0.011],
            'GCN': [0.649, 0.026],
            'Morgan + MLP (DeepPurpose)': [0.649, 0.019],
            'NeuralFP': [0.667, 0.020],
            'ContextPred': [0.669, 0.030],
            'CNN (DeepPurpose)': [0.675, 0.011],
            'AttentiveFP': [0.678, 0.012],
            'RDKit2D + MLP (DeepPurpose)': [0.678, 0.003],
            'AttrMasking': [0.685, 0.025],
            'MolE': [0.823, 0.019]
        }
    }
}


def format_result(result):
    # param result: list of two values [metric, std]
    return f"{result[0]:.3f} ± {result[1]:.3f}"


def format_results(results_dic):
    # param results_dic: dictionary with {dataset_name: [metric, std]}
    return {k: [format_result(v)] for k, v in results_dic.items()}


def get_leaderboard_position(results):
    """
    Get the leaderboard position of the results for each task.
    The position is determined by comparing the result with the leaderboard values.
    param results: dictionary with {dataset_name: [metric, std]}
    return a dictionary with: {dataset_name: position}.
    """
    positions = {}
    for task, data in benchmark_results.items():
        leaderboard = [v[0] for v in data['leaderboard'].values()]
        result_value = results[task][0]
        if data['metric'].endswith('(↑)'):
            positions[task] = np.sum(np.array(leaderboard) > result_value) + 1
        elif data['metric'].endswith('(↓)'):
            positions[task] = np.sum(np.array(leaderboard) < result_value) + 1
    return positions


def merge_dfs(dfs: list) -> pd.DataFrame:
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(df, left_index=True, right_index=True, how='inner')
    return merged_df


def create_results_df(results_dict: dict, results_name: str, format=True, include_leaderboard_position=True) -> pd.DataFrame:
    """
    Creates a DataFrame from a dictionary of results.
    Formats the result as "metric ± std" and optionally adds the leaderboard position.

    Parameters
    ----------
    results_dict : dict
        Dictionary of results in the form {dataset_name: [metric, std]}.
    results_name : str
        Name of the method for which the results are being created.
    format : bool, optional
        If True, formats results as a string. If False, keeps metric and std as separate columns.
    include_leaderboard_position : bool, optional
        If True, adds the leaderboard position for each dataset.

    Returns
    -------
    pd.DataFrame
        A formatted DataFrame with results and (optionally) leaderboard positions.
    """
    if format:
        results_df = pd.DataFrame.from_dict(format_results(results_dict), orient='index', columns=[results_name + 'result'])
    else:
        results_df = pd.DataFrame.from_dict(results_dict, orient='index', columns=[results_name + 'metric', results_name + 'std'])
    
    results_df.index.name = 'dataset'

    if include_leaderboard_position:
        # Add leaderboard position for each result
        positions_dict = get_leaderboard_position(results_dict)
        results_df[results_name + 'position'] = results_df.index.map(positions_dict)
    
    return results_df


def get_sota():
    """
    Return the state-of-the-art (SOTA) results for each dataset in the benchmark.
    It creates a dictionary in the same format as the results of a method: {dataset_name: [metric, std]}
    """
    sota_dict = {}
    for dataset, data in benchmark_results.items():
        if '↓' in data['metric']:
            sota = min(data['leaderboard'].values(), key=lambda x: x[0])
        else:
            sota = max(data['leaderboard'].values(), key=lambda x: x[0])
        sota_dict[dataset] = sota
    return sota_dict


def get_sota_df(format=True):
    """
    Creates a DataFrame with SOTA results and description of each dataset: 
    It adds this columns:
        - metric used (AUC, MAE..)
        - task (classification or regression).
    """
    tasks_dict = {k: [v['metric'], v['task']] for k, v in benchmark_results.items()}  # {dataset_name: [metric_name, task]}
    tasks_df = pd.DataFrame.from_dict(tasks_dict, orient='index', columns=['metric', 'task'])
    tasks_df.index.name = 'dataset'

    sota_dict = get_sota()
    sota_df = create_results_df(sota_dict, "SOTA ", format=format, include_leaderboard_position=False)

    return merge_dfs([tasks_df, sota_df])

