import numpy as np
import pytest

from src import data as D
from src import model as M


@pytest.fixture(scope="module")
def donnees():
    return D.separer_X_y(D.charger_donnees())


def test_zeros_impossibles_remplaces(donnees):
    X, y = donnees
    assert len(X) == 768 and set(y.unique()) == {0, 1}
    assert (X[D.ZEROS_IMPOSSIBLES] == 0).sum().sum() == 0
    assert X["insuline"].isna().sum() == 374


def test_seuil_atteint_la_sensibilite():
    y = np.array([0, 0, 0, 1, 1, 1, 1])
    proba = np.array([0.1, 0.4, 0.6, 0.3, 0.5, 0.7, 0.9])
    s = M.seuil_pour_sensibilite(y, proba, 0.75)
    assert s == 0.5
    assert (proba[y == 1] >= s).mean() >= 0.75


def test_modeles_battent_le_hasard(donnees):
    X, y = donnees
    res = M.comparer_modeles(M.construire_modeles(), X, y, n_repeats=1)
    assert (res["AUC"] > 0.75).all()
