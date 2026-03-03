# =========================================================
# 1. IMPORTS
# =========================================================
import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# =========================================================
# 2. CONFIGURATION GLOBALE
# =========================================================
warnings.filterwarnings("ignore")
SEED = 42
N_FOLDS = 5
N_TRIALS = 50
N_WEIGHT_TRIALS = 100
TARGET = "Heart Disease"

# =========================================================
# 3. CHARGEMENT DES DONNÉES
# =========================================================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train[TARGET] = train[TARGET].map({"Absence": 0, "Presence": 1})
X = train.drop(columns=[TARGET, "id"])
y = train[TARGET]
X_test = test.drop(columns=["id"])
test_ids = test["id"]


# =========================================================
# 4. FEATURE ENGINEERING
# =========================================================
def create_features(df):
    df_out = df.copy()
    df_out["HR_diff"] = (220 - df_out["Age"]) - df_out["Max HR"]
    df_out["BP_Age_ratio"] = df_out["BP"] / df_out["Age"]
    df_out["High_Risk_Blood"] = df_out["Max HR"] / df_out["Age"]
    df_out["High_Risk_Blood"] = (
        (df_out["Cholesterol"] > 240) & (df_out["BP"] > 130)
    ).astype(int)
    return df_out


X = create_features(X)
X_test = create_features(X_test)

# =========================================================
# 5. FONCTIONS OBJECTIVE POUR OPTUNA
# =========================================================


def objective_lgbm(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        "random_state": SEED,
        "verbosity": -1,
    }

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
        X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

        model = LGBMClassifier(**params)
        model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
        )

        val_preds = model.predict_proba(X_val_fold)[:, 1]
        scores.append(roc_auc_score(y_val_fold, val_preds))

    return float(np.mean(scores))


def objective_xgb(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "random_state": SEED,
        "verbosity": 0,
        "use_label_encoder": False,
        "eval_metric": "auc",
    }

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
        X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

        model = XGBClassifier(**params)
        model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=False,
        )

        val_preds = model.predict_proba(X_val_fold)[:, 1]
        scores.append(roc_auc_score(y_val_fold, val_preds))

    return float(np.mean(scores))


def objective_catboost(trial):
    params = {
        "depth": trial.suggest_int("depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "iterations": trial.suggest_int("iterations", 100, 2000),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0, 10),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 10),
        "random_state": SEED,
        "verbose": 0,
        "eval_metric": "AUC",
    }

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
        X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

        model = CatBoostClassifier(**params)
        model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            early_stopping_rounds=50,
            verbose=False,
        )

        val_preds = model.predict_proba(X_val_fold)[:, 1]
        scores.append(roc_auc_score(y_val_fold, val_preds))

    return float(np.mean(scores))


# =========================================================
# 6. OPTIMISATION DES 3 MODÈLES
# =========================================================
print("=" * 60)
print("OPTIMISATION LIGHTGBM")
print("=" * 60)

study_lgbm = optuna.create_study(
    direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)
)
study_lgbm.optimize(objective_lgbm, n_trials=N_TRIALS, show_progress_bar=True)

best_params_lgbm = study_lgbm.best_params
best_score_lgbm = study_lgbm.best_value
print(f"\nMeilleur AUC LightGBM: {best_score_lgbm:.4f}")
print(f"Meilleurs params: {best_params_lgbm}")

print("\n" + "=" * 60)
print("OPTIMISATION XGBOOST")
print("=" * 60)

study_xgb = optuna.create_study(
    direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)
)
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS, show_progress_bar=True)

best_params_xgb = study_xgb.best_params
best_score_xgb = study_xgb.best_value
print(f"\nMeilleur AUC XGBoost: {best_score_xgb:.4f}")
print(f"Meilleurs params: {best_params_xgb}")

print("\n" + "=" * 60)
print("OPTIMISATION CATBOOST")
print("=" * 60)

study_cat = optuna.create_study(
    direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)
)
study_cat.optimize(objective_catboost, n_trials=N_TRIALS, show_progress_bar=True)

best_params_cat = study_cat.best_params
best_score_cat = study_cat.best_value
print(f"\nMeilleur AUC CatBoost: {best_score_cat:.4f}")
print(f"Meilleurs params: {best_params_cat}")

# =========================================================
# 7. GÉNÉRATION DES PRÉDICTIONS OOF
# =========================================================
print("\n" + "=" * 60)
print("GÉNÉRATION DES PRÉDICTIONS OOF")
print("=" * 60)

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_preds_lgbm = np.zeros(len(X))
oof_preds_xgb = np.zeros(len(X))
oof_preds_cat = np.zeros(len(X))

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

    model_lgbm = LGBMClassifier(**best_params_lgbm, random_state=SEED, verbosity=-1)
    model_lgbm.fit(X_train_fold, y_train_fold)
    oof_preds_lgbm[val_idx] = model_lgbm.predict_proba(X_val_fold)[:, 1]

    model_xgb = XGBClassifier(
        **best_params_xgb,
        random_state=SEED,
        verbosity=0,
        use_label_encoder=False,
        eval_metric="auc",
    )
    model_xgb.fit(X_train_fold, y_train_fold, verbose=False)
    oof_preds_xgb[val_idx] = model_xgb.predict_proba(X_val_fold)[:, 1]

    model_cat = CatBoostClassifier(
        **best_params_cat, random_state=SEED, verbose=0, eval_metric="AUC"
    )
    model_cat.fit(X_train_fold, y_train_fold, verbose=False)
    oof_preds_cat[val_idx] = model_cat.predict_proba(X_val_fold)[:, 1]

print(f"OOF AUC LightGBM: {roc_auc_score(y, oof_preds_lgbm):.4f}")
print(f"OOF AUC XGBoost: {roc_auc_score(y, oof_preds_xgb):.4f}")
print(f"OOF AUC CatBoost: {roc_auc_score(y, oof_preds_cat):.4f}")

# =========================================================
# 8. OPTIMISATION DES POIDS DE BLENDING
# =========================================================
print("\n" + "=" * 60)
print("OPTIMISATION DES POIDS DE BLENDING")
print("=" * 60)


def objective_weights(trial):
    w_lgbm = trial.suggest_float("w_lgbm", 0.0, 1.0)
    w_xgb = trial.suggest_float("w_xgb", 0.0, 1.0)
    w_cat = trial.suggest_float("w_cat", 0.0, 1.0)

    total = w_lgbm + w_xgb + w_cat
    if total == 0:
        return 0.0

    w_lgbm_norm = w_lgbm / total
    w_xgb_norm = w_xgb / total
    w_cat_norm = w_cat / total

    blended_preds = (
        w_lgbm_norm * oof_preds_lgbm
        + w_xgb_norm * oof_preds_xgb
        + w_cat_norm * oof_preds_cat
    )

    return float(roc_auc_score(y, blended_preds))


study_weights = optuna.create_study(
    direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)
)
study_weights.optimize(
    objective_weights, n_trials=N_WEIGHT_TRIALS, show_progress_bar=True
)

best_weights = study_weights.best_params
best_blended_score = study_weights.best_value

total = best_weights["w_lgbm"] + best_weights["w_xgb"] + best_weights["w_cat"]
w_lgbm_final = best_weights["w_lgbm"] / total
w_xgb_final = best_weights["w_xgb"] / total
w_cat_final = best_weights["w_cat"] / total

print(f"\nMeilleur AUC Blended: {best_blended_score:.4f}")
print(
    f"Poids optimaux: LGBM={w_lgbm_final:.3f}, XGB={w_xgb_final:.3f}, CAT={w_cat_final:.3f}"
)

# =========================================================
# 9. ENTRAÎNEMENT FINAL ET PRÉDICTIONS TEST
# =========================================================
print("\n" + "=" * 60)
print("ENTRAÎNEMENT FINAL SUR TOUT LE DATASET")
print("=" * 60)

test_preds_lgbm = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))
test_preds_cat = np.zeros(len(X_test))

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    print(f"Fold {fold + 1}/{N_FOLDS}...", end=" ")

    X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]

    model_lgbm = LGBMClassifier(**best_params_lgbm, random_state=SEED, verbosity=-1)
    model_lgbm.fit(X_train_fold, y_train_fold)
    test_preds_lgbm += model_lgbm.predict_proba(X_test)[:, 1] / N_FOLDS

    model_xgb = XGBClassifier(
        **best_params_xgb,
        random_state=SEED,
        verbosity=0,
        use_label_encoder=False,
        eval_metric="auc",
    )
    model_xgb.fit(X_train_fold, y_train_fold, verbose=False)
    test_preds_xgb += model_xgb.predict_proba(X_test)[:, 1] / N_FOLDS

    model_cat = CatBoostClassifier(
        **best_params_cat, random_state=SEED, verbose=0, eval_metric="AUC"
    )
    model_cat.fit(X_train_fold, y_train_fold, verbose=False)
    test_preds_cat += model_cat.predict_proba(X_test)[:, 1] / N_FOLDS

    print("OK")

test_preds = (
    w_lgbm_final * test_preds_lgbm
    + w_xgb_final * test_preds_xgb
    + w_cat_final * test_preds_cat
)

# =========================================================
# 10. RÉSUMÉ FINAL
# =========================================================
print("\n" + "=" * 60)
print("RÉSUMÉ FINAL")
print("=" * 60)
print(f"AUC LightGBM (CV):  {best_score_lgbm:.4f}")
print(f"AUC XGBoost (CV):   {best_score_xgb:.4f}")
print(f"AUC CatBoost (CV):  {best_score_cat:.4f}")
print(f"AUC Blended (OOF):  {best_blended_score:.4f}")
print(f"\nPoids finaux:")
print(f"  LightGBM: {w_lgbm_final:.3f} ({w_lgbm_final * 100:.1f}%)")
print(f"  XGBoost:  {w_xgb_final:.3f} ({w_xgb_final * 100:.1f}%)")
print(f"  CatBoost: {w_cat_final:.3f} ({w_cat_final * 100:.1f}%)")

# =========================================================
# 11. SOUMISSION
# =========================================================
submission = pd.DataFrame({"id": test_ids, TARGET: test_preds})

submission.to_csv("submission_optuna.csv", index=False)
print(f"\nFichier de soumission créé: submission_optuna.csv")
print("=" * 60)
