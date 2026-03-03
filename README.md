🫀 Kaggle Playground Series - S6E2: Heart Disease Prediction

🏆 Top 1000 Finish (Rank: 839) 

Ce dépôt contient ma solution pour la compétition Kaggle S6E2. L'objectif était de prédire la présence de maladies cardiaques en maximisant l'AUC-ROC. Ma progression vers le Top 1000 a été portée par une approche rigoureuse d'optimisation des hyperparamètres et une stratégie d'ensemble.

🚀 Points Clés de l'Approche
1. Feature Engineering Physiologique

Au-delà des données brutes, j'ai créé des indicateurs métiers pour aider les modèles à capter des signaux non-linéaires :

    HR_diff : Écart entre la fréquence cardiaque maximale théorique (220-âge) et celle observée.

    BP_Age_ratio : Ratio tension artérielle / âge.

    High_Risk_Blood : Feature binaire combinant cholestérol élevé (>240) et hypertension (>130).

2. Le "Trio de Fer" du Boosting

Utilisation des trois algorithmes leaders pour les données tabulaires, chacun apportant une complémentarité sur la gestion des variables :

    LightGBM (Rapidité et Leaf-wise growth)

    XGBoost (Robustesse via régularisation gamma)

    CatBoost (Gestion native optimisée des catégories)

3. Double Optimisation avec Optuna (Bayesian Search)

L'élément différenciateur de ce projet est l'utilisation d'Optuna à deux niveaux :

    Hyperparameter Tuning : Recherche fine de l'espace des paramètres pour chaque modèle (50 trials par modèle avec Stratified 5-Fold CV).

    Weight Blending Optimization : Recherche des poids optimaux pour combiner les probabilités des trois modèles afin de minimiser l'erreur de généralisation sur les prédictions OOF (Out-of-Fold).


📈 Perspectives d'amélioration

    Implémenter un Stacking Classifier (Meta-learner).

    Approfondir l'analyse des valeurs aberrantes (Outliers) sur le cholestérol.

    Tester des techniques d'encodage avancées (Target Encoding) pour les variables catégorielles.

    <img width="2008" height="115" alt="image" src="https://github.com/user-attachments/assets/e72498ed-f26e-4dd3-9039-b3d69fb3e1a9" />
