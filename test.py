# =========================================================
# 1. IMPORTS
# =========================================================
import numpy as np
import pandas as pd
import warnings

# Sklearn pour la validation et les métriques
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, accuracy_score  # À adapter selon la métrique

# Modèles classiques
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# =========================================================
# 2. CONFIGURATION GLOBALE
# =========================================================
warnings.filterwarnings("ignore")
SEED = 42
N_FOLDS = 5
TARGET = "Heart Disease"

# =========================================================
# 3. CHARGEMENT DES DONNÉES
# =========================================================
# TODO: Charger les CSV
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train[TARGET] = train[TARGET].map({"Absence": 0, "Presence": 1})
# Séparation Features (X) / Target (y)
X = train.drop(columns=[TARGET, "id"])  # On enlève l'ID et la cible
y = train[TARGET]
X_test = test.drop(columns=["id"])
test_ids = test["id"]


# =========================================================
# 4. FEATURE ENGINEERING (Ingénierie des caractéristiques)
# =========================================================
def create_features(df):
    df_out = df.copy()
    # Défi 1 : La différence avec le max théorique (220 - Age)
    df_out["HR_diff"] = (220 - df_out["Age"]) - df_out["Max HR"]

    # Défi 2 : Le ratio Tension / Âge
    df_out["BP_Age_ratio"] = df_out["BP"] / df_out["Age"]

    df_out["High_Risk_Blood"] = df_out["Max HR"] / df_out["Age"]

    df_out["High_Risk_Blood"] = ((df_out["Cholesterol"] > 240) & (df_out["BP"] > 130)).astype(int)

    return df_out


X = create_features(X)
X_test = create_features(X_test)

# =========================================================
# 5. STRATÉGIE DE VALIDATION (Cross-Validation)
# =========================================================
# StratifiedKFold préserve la proportion des classes dans chaque fold
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# Tableaux pour stocker les prédictions
oof_preds = np.zeros(len(X))  # Prédictions sur le train (Out-Of-Fold)
test_preds = np.zeros(len(X_test))  # Prédictions sur le test

# =========================================================
# 6. BOUCLE D'ENTRAÎNEMENT ET ÉVALUATION
# =========================================================
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    print(f"--- Fold {fold + 1}/{N_FOLDS} ---")

    # Séparation Train / Validation pour ce fold
    X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

    # TODO: C'est ICI qu'il faut faire du Target Encoding si tu en as besoin,
    # en apprenant sur X_train_fold et en appliquant sur X_val_fold et X_test.

    # Initialisation du modèle (Exemple avec LightGBM)
    model = LGBMClassifier(random_state=SEED, n_estimators=500)

    # Entraînement
    model.fit(X_train_fold, y_train_fold)

    # Prédiction sur le fold de validation
    # On utilise predict_proba()[:, 1] pour avoir la probabilité de la classe 1
    val_preds = model.predict_proba(X_val_fold)[:, 1]
    oof_preds[val_idx] = val_preds

    # Évaluation du fold
    fold_score = roc_auc_score(y_val_fold, val_preds)
    print(f"Score AUC du fold: {fold_score:.4f}\n")

    # TODO: Prédiction sur le jeu de test X_test
    # Comment vas-tu accumuler les prédictions du test à chaque fold ?
    test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS

# Score global sur tout le jeu d'entraînement
print(f"=== SCORE CV FINAL (OOF) : {roc_auc_score(y, oof_preds):.4f} ===")

# =========================================================
# 7. SÉRIALISATION / SOUMISSION
# =========================================================
submission = pd.DataFrame({
    "id": test_ids,

    TARGET: test_preds
})

submission.to_csv("submission.csv", index=False)
print("Fichier de soumission créé avec succès !")