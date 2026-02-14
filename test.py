import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# --- CONFIGURATION DES CHEMINS ---
DATA_DIR = Path(__file__).resolve().parent
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"


def build_preprocessor(num_cols, bin_cols, cat_cols):
    """Pipeline de transformation des données."""
    return ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), num_cols + bin_cols),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )


def add_cluster_features(X_scaled, kmeans_model, cluster_encoder=None):
    """Ajoute les clusters comme nouvelles variables explicatives."""
    cluster_labels = kmeans_model.predict(X_scaled).reshape(-1, 1)

    if cluster_encoder is None:
        cluster_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        cluster_oh = cluster_encoder.fit_transform(cluster_labels)
        return np.hstack([X_scaled, cluster_oh]), cluster_encoder
    else:
        cluster_oh = cluster_encoder.transform(cluster_labels)
        return np.hstack([X_scaled, cluster_oh]), cluster_encoder


def main():
    # 1. Chargement
    if not all(p.exists() for p in [TRAIN_PATH, TEST_PATH, SAMPLE_SUB_PATH]):
        print("Erreur : Fichiers CSV manquants.")
        return

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

    # 2. Préparation des labels (Conversion Absence/Presence -> 0/1)
    # On le fait avant le split pour que y soit numérique
    target_col = "Heart Disease"
    label_mapping = {"Absence": 0, "Presence": 1}

    X = train_df.drop(columns=["id", target_col])
    # On map la cible : indispensable pour de nombreux algorithmes
    y = train_df[target_col].map(label_mapping)

    # Définition des colonnes
    num_cols = ["Age", "BP", "Cholesterol", "Max HR", "ST depression", "Number of vessels fluro"]
    bin_cols = ["Sex", "FBS over 120", "Exercise angina"]
    cat_cols = ["Chest pain type", "EKG results", "Slope of ST", "Thallium"]

    # 3. Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Preprocessing & Clustering
    preprocessor = build_preprocessor(num_cols, bin_cols, cat_cols)
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    X_test_scaled = preprocessor.transform(test_df.drop(columns=["id"]))

    kmeans = KMeans(n_clusters=7, n_init=10, random_state=42)
    kmeans.fit(X_train_scaled)

    # 5. Augmentation des features
    X_train_aug, cluster_enc = add_cluster_features(X_train_scaled, kmeans)
    X_val_aug, _ = add_cluster_features(X_val_scaled, kmeans, cluster_enc)
    X_test_aug, _ = add_cluster_features(X_test_scaled, kmeans, cluster_enc)

    # 6. Entraînement
    print(f"Entraînement sur {X_train_aug.shape[1]} features (numérique)...")
    model = RandomForestClassifier(n_estimators=250, max_depth=12, random_state=42)
    model.fit(X_train_aug, y_train)

    # 7. Évaluation
    print(f"Score de validation : {model.score(X_val_aug, y_val):.4f}")

    # 8. Inférence et Soumission
    test_preds = model.predict(X_test_aug)  # Ce sont déjà des 0 et 1 ici

    results_df = pd.DataFrame({
        "id": test_df["id"],
        "preds": test_preds
    })

    # Fusion avec le format imposé par sample_submission
    final_submission = sample_sub.drop(columns=[target_col]).merge(results_df, on="id")
    final_submission.rename(columns={"preds": target_col}, inplace=True)

    # Sauvegarde
    final_submission.to_csv(DATA_DIR / "submission_complete.csv", index=False)
    print("Fichier 'submission_complete.csv' généré avec succès (format 0/1).")

    # 9. Export des modèles
    joblib.dump(preprocessor, DATA_DIR / "preprocessor.pkl")
    joblib.dump(kmeans, DATA_DIR / "kmeans.pkl")
    joblib.dump(cluster_enc, DATA_DIR / "cluster_encoder.pkl")
    joblib.dump(model, DATA_DIR / "final_model.pkl")


if __name__ == "__main__":
    main()