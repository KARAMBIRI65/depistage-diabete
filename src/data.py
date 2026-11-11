"""Chargement et nettoyage du jeu de données Pima Indians Diabetes.

Source : Smith, J. W. et al. (1988). Using the ADAP learning algorithm to forecast
the onset of diabetes mellitus. Données du National Institute of Diabetes and
Digestive and Kidney Diseases (NIDDK), diffusées par l'UCI et Kaggle (CC0).
768 patientes d'origine pima (Arizona), âgées de 21 ans et plus.
Cible : diabète de type 2 diagnostiqué dans les 5 ans (1) ou non (0).
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
LOCAL_FILE = DATA_DIR / "diabetes.csv"
URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

# Noms d'origine -> noms lisibles en français.
COLONNES = {
    "Pregnancies": "nb_grossesses",
    "Glucose": "glucose",                       # glycémie 2 h après ingestion de glucose (mg/dL)
    "BloodPressure": "pression_diastolique",    # mm Hg
    "SkinThickness": "pli_cutane",              # pli cutané du triceps (mm)
    "Insulin": "insuline",                      # insuline sérique à 2 h (µU/mL)
    "BMI": "imc",                               # indice de masse corporelle (kg/m²)
    "DiabetesPedigreeFunction": "antecedents_familiaux",  # score d'hérédité
    "Age": "age",
    "Outcome": "diabete",
}

# Dans le fichier brut, une valeur manquante est codée 0 : une glycémie ou un IMC
# nul est physiologiquement impossible. On les remplace par NaN.
ZEROS_IMPOSSIBLES = ["glucose", "pression_diastolique", "pli_cutane", "insuline", "imc"]

CIBLE = "diabete"


def charger_brut() -> pd.DataFrame:
    """Lit le fichier local, sinon le télécharge (aucun compte nécessaire)."""
    if LOCAL_FILE.exists():
        return pd.read_csv(LOCAL_FILE)
    df = pd.read_csv(URL)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOCAL_FILE, index=False)
    return df


def charger_donnees() -> pd.DataFrame:
    """Données renommées en français, zéros impossibles remplacés par NaN."""
    df = charger_brut().rename(columns=COLONNES)
    df[ZEROS_IMPOSSIBLES] = df[ZEROS_IMPOSSIBLES].replace(0, np.nan)
    return df


def separer_X_y(df: pd.DataFrame):
    return df.drop(columns=CIBLE), df[CIBLE]
