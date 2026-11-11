"""Modèles, évaluation et choix du seuil de dépistage."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict, cross_validate, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 42


def construire_modeles() -> dict:
    """Trois familles de modèles. L'imputation (médiane) est DANS le pipeline :
    elle est apprise uniquement sur les plis d'entraînement, sans fuite de données."""
    return {
        "Régression logistique": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        ),
        "Forêt aléatoire": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                   class_weight="balanced", random_state=SEED, n_jobs=-1),
        ),
        # XGBoost gère nativement les valeurs manquantes : pas d'imputation.
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=3, scale_pos_weight=1.87,
            eval_metric="logloss", random_state=SEED, n_jobs=-1,
        ),
    }


def comparer_modeles(modeles: dict, X, y, n_repeats: int = 5) -> pd.DataFrame:
    """Validation croisée 5 plis répétée : sur 600 patientes, un seul découpage
    serait trop instable pour départager les modèles."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    lignes = []
    for nom, m in modeles.items():
        s = cross_validate(m, X, y, cv=cv, scoring=["roc_auc", "average_precision"], n_jobs=-1)
        lignes.append({
            "modèle": nom,
            "AUC": s["test_roc_auc"].mean(), "AUC écart-type": s["test_roc_auc"].std(),
            "PR-AUC": s["test_average_precision"].mean(),
        })
    return pd.DataFrame(lignes).sort_values("AUC", ascending=False).reset_index(drop=True)


def probas_hors_plis(modele, X, y) -> np.ndarray:
    """Probabilités prédites pour chaque patiente par un modèle qui ne l'a pas vue."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    return cross_val_predict(modele, X, y, cv=cv, method="predict_proba")[:, 1]


def seuil_pour_sensibilite(y, proba, sensibilite_cible: float = 0.85) -> float:
    """Plus grand seuil qui atteint la sensibilité visée.
    En dépistage, rater une malade (faux négatif) est plus grave qu'une fausse
    alerte, qui sera simplement levée par un test de confirmation (HbA1c)."""
    y = np.asarray(y)
    for s in np.sort(np.unique(proba))[::-1]:
        if (proba[y == 1] >= s).mean() >= sensibilite_cible:
            return float(s)
    return 0.0


def metriques(y, proba, seuil: float) -> dict:
    pred = (proba >= seuil).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "AUC": roc_auc_score(y, proba),
        "PR-AUC": average_precision_score(y, proba),
        "Brier": brier_score_loss(y, proba),
        "sensibilité": tp / (tp + fn),
        "spécificité": tn / (tn + fp),
        "VPP": tp / (tp + fp) if tp + fp else 0.0,
        "VPN": tn / (tn + fn) if tn + fn else 0.0,
        "patientes orientées": (tp + fp) / len(pred),
        "faux négatifs": int(fn),
    }
