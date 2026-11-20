import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Création du dossier 'figure' s'il n'existe pas
os.makedirs('figure', exist_ok=True)
sns.set_theme(style='whitegrid')

# ==========================================
# 1. CHARGEMENT ET ANALYSE EXPLORATOIRE (EDA)
# ==========================================
df = pd.read_csv('CASP.csv')

print('--- 1. INFORMATIONS GÉNÉRALES ---')
print(f"Nombre d'observations initiales : {df.shape[0]}")
print(f'Nombre de variables : {df.shape[1]}')
print('\nNature des variables :')
print(df.dtypes)

print('\nValeurs manquantes par variable :')
print(df.isnull().sum())

# Gestion des doublons
duplicates_count = df.duplicated().sum()
print(f"\nNombre d'observations dupliquées : {duplicates_count}")
if duplicates_count > 0:
  df = df.drop_duplicates().reset_index(drop=True)
  print(
      'Observations dupliquées supprimées. Nouveau total :'
      f' {df.shape[0]}'
  )

print('\nStatistiques descriptives :')
print(df.describe())

print('\nMatrice de corrélation initiale :')
corr_matrix = df.corr()
print(corr_matrix)

# Sauvegarde de la matrice de corrélation
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matrice de corrélation des variables')
plt.tight_layout()
plt.savefig('figure/correlation_matrix.png')
plt.close()

# Graphiques univariés regroupés
n_vars = len(df.columns)
fig, axes = plt.subplots(n_vars, 2, figsize=(12, 4 * n_vars))
for i, col in enumerate(df.columns):
  sns.histplot(df[col], kde=True, ax=axes[i, 0], color='skyblue')
  axes[i, 0].set_title(f'Distribution de {col}')
  sns.boxplot(x=df[col], ax=axes[i, 1], color='lightgreen')
  axes[i, 1].set_title(f'Boxplot de {col}')
plt.tight_layout()
plt.savefig('figure/univariate_all_variables.png')
plt.close()

# Graphiques 2D regroupés
features = [col for col in df.columns if col != 'RMSD']
n_feat = len(features)
n_rows_2d = int(np.ceil(n_feat / 3))
fig, axes = plt.subplots(n_rows_2d, 3, figsize=(15, 4 * n_rows_2d))
axes = axes.flatten()

for i, col in enumerate(features):
  sns.scatterplot(x=df[col], y=df['RMSD'], ax=axes[i], alpha=0.3, color='purple')
  axes[i].set_title(f'{col} vs RMSD')
  axes[i].set_xlabel(col)
  axes[i].set_ylabel('RMSD')

for j in range(i + 1, len(axes)):
  fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('figure/bivariate_scatter_plots.png')
plt.close()


# ==========================================
# 2. SUPPRESSION DES VARIABLES FORTEMENT CORRÉLÉES
# ==========================================
# Calcul de la matrice de corrélation absolue entre prédicteurs
corr_features = df.drop(columns=['RMSD']).corr().abs()
upper_tri = corr_features.where(
    np.triu(np.ones(corr_features.shape), k=1).astype(bool)
)

# Identification des variables ayant une corrélation > 0.85 avec une autre
threshold = 0.90
to_drop = [
    column for column in upper_tri.columns if any(upper_tri[column] > threshold)
]

print(f'\nVariables fortement corrélées (> {threshold}) identifiées :')
print(to_drop)

# Suppression des variables redondantes
df_filtered = df.drop(columns=to_drop)
print(f'Nombre de variables après nettoyage : {df_filtered.shape[1] - 1}')


# ==========================================
# 3. PRÉPARATION DES DONNÉES ET DIVISION
# ==========================================
X = df_filtered.drop(columns=['RMSD'])
y = df_filtered['RMSD']

X_train_df, X_test_df, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = X_train_df.values
X_test = X_test_df.values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 4. IMPLÉMENTATION DES MODÈLES ML
# ==========================================
ols = LinearRegression()
ols.fit(X_train_scaled, y_train)

rf_params = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20],
    'min_samples_split': [5, 10],
}
rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    rf_params,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
print(f'\nMeilleurs paramètres Random Forest : {rf_grid.best_params_}')

mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
mlp.fit(X_train_scaled, y_train)

models = {
    'OLS': (ols, X_train_scaled, X_test_scaled),
    'Random Forest': (best_rf, X_train, X_test),
    'MLP': (mlp, X_train_scaled, X_test_scaled),
}


# ==========================================
# 5. ÉVALUATION CLASSIQUE (TRAIN & TEST)
# ==========================================
train_metrics = []
test_metrics = []

for name, (model, X_tr, X_te) in models.items():
  y_pred_train = model.predict(X_tr)
  train_metrics.append({
      'Modèle': name,
      'RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
      'MAE': mean_absolute_error(y_train, y_pred_train),
      'R2': r2_score(y_train, y_pred_train),
  })

  y_pred_test = model.predict(X_te)
  test_metrics.append({
      'Modèle': name,
      'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
      'MAE': mean_absolute_error(y_test, y_pred_test),
      'R2': r2_score(y_test, y_pred_test),
  })

print('\n--- Performances sur l\'ensemble d\'ENTRAÎNEMENT (Filtré) ---')
print(pd.DataFrame(train_metrics).to_string(index=False))

print('\n--- Performances sur l\'ensemble de TEST (Filtré) ---')
print(pd.DataFrame(test_metrics).to_string(index=False))


# ==========================================
# 6. PRÉDICTION CONFORME (Jackknife, Jackknife+, CV)
# ==========================================
alpha = 0.1
n_train = len(y_train)
cp_results = []

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, (model, X_tr, X_te) in models.items():
  fold_models = []
  fold_residuals = []

  for train_idx, val_idx in kf.split(X_tr):
    X_f_tr, X_f_val = X_tr[train_idx], X_tr[val_idx]
    y_f_tr, y_f_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    f_model = clone(model)
    f_model.fit(X_f_tr, y_f_tr)

    val_preds = f_model.predict(X_f_val)
    fold_res = np.abs(y_f_val - val_preds)

    fold_models.append(f_model)
    fold_residuals.extend(fold_res)

  # 1. Jackknife Standard
  global_preds = model.predict(X_tr)
  global_res = np.abs(y_train - global_preds)
  q_jk = np.quantile(global_res, np.ceil((n_train + 1) * (1 - alpha)) / n_train)

  test_preds = model.predict(X_te)
  cp_results.append({
      'Modèle': name,
      'Méthode CP': 'Jackknife (Standard)',
      'Couv. Empirique': f'{np.mean((y_test >= test_preds - q_jk) & (y_test <= test_preds + q_jk)):.2%}',
      'Largeur Moyenne': np.mean(2 * q_jk),
  })

  # 2. Jackknife+
  test_preds_folds = np.array([fm.predict(X_te) for fm in fold_models])
  fold_res_arr = np.array(fold_residuals)
  q_jk_plus = np.quantile(
      fold_res_arr, np.ceil((n_train + 1) * (1 - alpha)) / n_train
  )

  lower_jk = np.min(test_preds_folds, axis=0) - q_jk_plus
  upper_jk = np.max(test_preds_folds, axis=0) + q_jk_plus
  cp_results.append({
      'Modèle': name,
      'Méthode CP': 'Jackknife+',
      'Couv. Empirique': f'{np.mean((y_test >= lower_jk) & (y_test <= upper_jk)):.2%}',
      'Largeur Moyenne': np.mean(upper_jk - lower_jk),
  })

  # 3. CV Conformal
  q_cv = np.quantile(
      fold_res_arr, np.ceil((n_train + 1) * (1 - alpha)) / n_train
  )
  lower_cv_int = test_preds - q_cv
  upper_cv_int = test_preds + q_cv
  cp_results.append({
      'Modèle': name,
      'Méthode CP': 'CV Conformal',
      'Couv. Empirique': f'{np.mean((y_test >= lower_cv_int) & (y_test <= upper_cv_int)):.2%}',
      'Largeur Moyenne': np.mean(upper_cv_int - lower_cv_int),
  })

df_cp = pd.DataFrame(cp_results)
print(
    '\n--- Résultats de la Prédiction Conforme (Après Suppression des'
    ' Colinéarités) ---'
)
print(df_cp.to_string(index=False))