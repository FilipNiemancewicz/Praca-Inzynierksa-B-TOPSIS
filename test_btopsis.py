"""Testy jednostkowe modułu btopsis.
p
Uruchomienie: pytest
Dane wzorcowe pochodzą z przykładu Zürich w artykule Roszkowska, Wachowicz,
Michalska (2025), Sustainability 17(22), 10106: Table 3, wzory (5), (20), (21).
"""

import numpy as np
import pandas as pd
import pytest

from btopsis import (
    build_beliefs,
    closeness,
    default_utilities,
    evaluate,
    normalize_weights,
    similarity_matrix,
)

STATES = ["Bardzo_dobra", "Raczej_dobra", "Raczej_zla", "Bardzo_zla"]
DK = "Nie_wiem"

# Zürich, kryteria C1-C5 (Table 3; dla C5 skala już odwrócona), wartości w procentach
ZURICH = [
    [27.74, 42.79, 11.75, 2.83, 14.89],
    [26.77, 43.81, 19.71, 3.65, 6.06],
    [25.92, 51.44, 17.06, 2.36, 3.22],
    [45.12, 36.02, 8.46, 0.73, 9.67],
    [31.20, 36.31, 13.46, 3.56, 15.47],
]


def make_df(objects):
    """objects: słownik {obiekt: lista wierszy [4 oceny + nie wiem]}."""
    rows = []
    for obj, matrix in objects.items():
        for j, vals in enumerate(matrix):
            rows.append({"Obiekt": obj, "Kryterium": f"C{j + 1}",
                         **dict(zip(STATES + [DK], vals))})
    return pd.DataFrame(rows)


def test_similarity_matrix_matches_article():
    """Macierz ze wzoru (21) artykułu dla użyteczności 1; 0,7; 0,4; 0."""
    expected = np.array([
        [1.0, 0.7, 0.4, 0.0],
        [0.7, 1.0, 0.7, 0.3],
        [0.4, 0.7, 1.0, 0.6],
        [0.0, 0.3, 0.6, 1.0],
    ])
    D = similarity_matrix(default_utilities(4))
    assert np.allclose(D, expected)


def test_default_utilities():
    assert default_utilities(4) == [1.0, 0.7, 0.4, 0.0]
    assert default_utilities(2) == [1.0, 0.0]
    assert default_utilities(3) == [1.0, 0.5, 0.0]


def test_pop_vector_zurich_c1():
    """Wektor POP dla Zürichu i C1: [0,3259; 0,5027; 0,1380; 0,0334] (artykuł)."""
    df = make_df({"Zurych": ZURICH})
    _, _, B = build_beliefs(df, "Obiekt", "Kryterium", STATES, DK)
    expected = np.array([0.3259, 0.5027, 0.1380, 0.0334])
    assert np.allclose(B[0, 0], expected, atol=2e-4)


def test_pop_vectors_sum_to_one():
    df = make_df({"Zurych": ZURICH})
    _, _, B = build_beliefs(df, "Obiekt", "Kryterium", STATES, DK)
    assert np.allclose(B.sum(axis=2), 1.0)


def test_percent_and_share_inputs_give_same_result():
    df_pct = make_df({"Zurych": ZURICH})
    df_share = make_df({"Zurych": [[v / 100 for v in row] for row in ZURICH]})
    r1, _ = evaluate(df_pct, "Obiekt", "Kryterium", STATES, DK)
    r2, _ = evaluate(df_share, "Obiekt", "Kryterium", STATES, DK)
    assert np.allclose(r1["R_i"], r2["R_i"])


def test_zurich_closeness_close_to_article():
    """Artykuł podaje R = 0,678 jako średnią z symulacji; dla jednej funkcji
    użyteczności (wzór 20) oczekujemy zgodności do ok. 0,002."""
    df = make_df({"Zurych": ZURICH})
    results, _ = evaluate(df, "Obiekt", "Kryterium", STATES, DK)
    assert abs(results.loc[0, "R_i"] - 0.678) < 0.002


def test_ideal_and_worst_objects():
    best = [[100, 0, 0, 0, 0]] * 3
    worst = [[0, 0, 0, 100, 0]] * 3
    df = make_df({"Najlepszy": best, "Najgorszy": worst})
    results = evaluate(df, "Obiekt", "Kryterium", STATES, DK)[0].set_index("Obiekt")
    assert abs(results.loc["Najlepszy", "R_i"] - 1.0) < 1e-9
    assert abs(results.loc["Najgorszy", "R_i"] - 0.0) < 1e-9


def test_ranking_order():
    df = make_df({
        "Dobry": [[40, 40, 10, 5, 5]] * 2,
        "Sredni": [[15, 35, 30, 10, 10]] * 2,
        "Slaby": [[5, 15, 35, 35, 10]] * 2,
    })
    results, _ = evaluate(df, "Obiekt", "Kryterium", STATES, DK)
    assert list(results["Obiekt"]) == ["Dobry", "Sredni", "Slaby"]
    assert list(results["Rank"]) == [1, 2, 3]


def test_weights_are_normalized():
    df = make_df({"A": ZURICH, "B": [[10, 20, 30, 30, 10]] * 5})
    crits = [f"C{j + 1}" for j in range(5)]
    r1, _ = evaluate(df, "Obiekt", "Kryterium", STATES, DK,
                     weights={c: 1.0 for c in crits})
    r2, _ = evaluate(df, "Obiekt", "Kryterium", STATES, DK,
                     weights={c: 7.0 for c in crits})
    assert np.allclose(r1["R_i"], r2["R_i"])


def test_weights_change_result():
    df = make_df({"A": [[50, 30, 10, 5, 5], [5, 10, 30, 50, 5]],
                  "B": [[5, 10, 30, 50, 5], [50, 30, 10, 5, 5]]})
    r_eq = evaluate(df, "Obiekt", "Kryterium", STATES, DK)[0].set_index("Obiekt")
    r_w = evaluate(df, "Obiekt", "Kryterium", STATES, DK,
                   weights={"C1": 5.0, "C2": 1.0})[0].set_index("Obiekt")
    assert abs(r_eq.loc["A", "R_i"] - r_eq.loc["B", "R_i"]) < 1e-9  # symetria
    assert r_w.loc["A", "R_i"] > r_w.loc["B", "R_i"]


def test_missing_criterion_renormalizes_weights():
    """Obiekt bez jednego kryterium nie jest karany: jego wynik jest liczony
    z pozostałych kryteriów, a wagi są renormalizowane."""
    full = [[20, 40, 20, 10, 10]] * 3
    df_full = make_df({"A": full, "B": [[10, 20, 30, 30, 10]] * 3})
    df_miss = df_full[~((df_full["Obiekt"] == "A") & (df_full["Kryterium"] == "C3"))]
    r_full = evaluate(df_full, "Obiekt", "Kryterium", STATES, DK)[0].set_index("Obiekt")
    r_miss = evaluate(df_miss, "Obiekt", "Kryterium", STATES, DK)[0].set_index("Obiekt")
    # identyczne wiersze dla każdego kryterium -> wynik A nie może się zmienić
    assert abs(r_full.loc["A", "R_i"] - r_miss.loc["A", "R_i"]) < 1e-9


def test_row_without_answers_is_skipped():
    df = make_df({"A": [[20, 40, 20, 10, 10], [0, 0, 0, 0, 100]]})
    _, _, B = build_beliefs(df, "Obiekt", "Kryterium", STATES, DK)
    assert np.isnan(B[0, 1]).all() and not np.isnan(B[0, 0]).any()


def test_duplicate_pairs_raise_error():
    df = make_df({"A": [[20, 40, 20, 10, 10]]})
    df = pd.concat([df, df], ignore_index=True)
    with pytest.raises(ValueError):
        build_beliefs(df, "Obiekt", "Kryterium", STATES, DK)


def test_missing_columns_raise_error():
    df = make_df({"A": [[20, 40, 20, 10, 10]]})
    with pytest.raises(ValueError):
        build_beliefs(df.drop(columns=[DK]), "Obiekt", "Kryterium", STATES, DK)


def test_wrong_number_of_utilities_raises_error():
    df = make_df({"A": [[20, 40, 20, 10, 10]]})
    with pytest.raises(ValueError):
        evaluate(df, "Obiekt", "Kryterium", STATES, DK, utilities=[1.0, 0.5, 0.0])


def test_negative_weights_raise_error():
    with pytest.raises(ValueError):
        normalize_weights({"C1": -1.0, "C2": 2.0}, ["C1", "C2"])


def test_closeness_is_between_zero_and_one():
    df = make_df({"A": ZURICH, "B": [[10, 20, 30, 30, 10]] * 5})
    results, _ = evaluate(df, "Obiekt", "Kryterium", STATES, DK)
    assert ((results["R_i"] >= 0) & (results["R_i"] <= 1)).all()
