"""Warstwa obliczeniowa metody Belief-TOPSIS (B-TOPSIS).

Moduł nie zależy od Streamlit ani Matplotlib, dzięki czemu można go testować
i używać niezależnie od interfejsu. Numery wzorów odnoszą się do rozdziału 1 pracy.
"""

import numpy as np
import pandas as pd


def default_utilities(n):
    """Domyślne użyteczności ocen: 1; 0,7; 0,4; 0 dla skali czterostopniowej.

    Dla innej liczby ocen użyteczności są rozłożone liniowo od 1 do 0.
    """
    if n == 4:
        return [1.0, 0.7, 0.4, 0.0]
    return [float(x) for x in np.linspace(1.0, 0.0, n)]


def similarity_matrix(utilities):
    """Macierz podobieństwa ocen D (N x N): s_ij = 1 - |U(H_i) - U(H_j)|  (wzór 8)."""
    u = np.asarray(utilities, dtype=float)
    return 1.0 - np.abs(u[:, None] - u[None, :])


def build_beliefs(df, obj_col, crit_col, state_cols, dk_col):
    """Buduje znormalizowaną macierz decyzyjną przekonań.

    Zwraca listę obiektów, listę kryteriów oraz tablicę B o wymiarach
    (obiekty x kryteria x N). Niewiedza ("nie wiem") jest rozdzielana
    proporcjonalnie do udziałów ocen (metoda POP, wzór 5). Pary obiekt-kryterium
    bez danych mają w tablicy wartość NaN.
    """
    state_cols = list(state_cols)
    needed = [obj_col, crit_col] + state_cols + [dk_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Brak kolumn w danych: {missing}")
    if df.duplicated(subset=[obj_col, crit_col]).any():
        raise ValueError(
            "Dane zawierają powtórzone pary obiekt–kryterium. "
            "Każda para powinna występować dokładnie raz."
        )

    values = df[state_cols + [dk_col]].apply(pd.to_numeric, errors="coerce")
    objects = list(df[obj_col].unique())
    criteria = list(df[crit_col].unique())
    n = len(state_cols)

    B = np.full((len(objects), len(criteria), n), np.nan)
    obj_idx = {o: i for i, o in enumerate(objects)}
    crit_idx = {c: j for j, c in enumerate(criteria)}

    for pos in range(len(df)):
        vals = values.iloc[pos].to_numpy(dtype=float)
        # dane w procentach (np. 0-100) -> udziały (0-1)
        if np.nansum(vals) > 1.5:
            vals = vals / 100.0
        m = vals[:n]
        total = m.sum()
        if not np.isfinite(total) or total <= 0:
            continue  # brak ocen -> para obiekt-kryterium jest pomijana
        # POP: beta_k + beta_k * beta_H / sum(beta) = beta_k / sum(beta)
        B[obj_idx[df[obj_col].iloc[pos]], crit_idx[df[crit_col].iloc[pos]], :] = m / total

    return objects, criteria, B


def normalize_weights(weights, criteria):
    """Normalizuje wagi kryteriów tak, aby sumowały się do 1."""
    w = np.array([weights[c] for c in criteria], dtype=float)
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Wagi muszą być nieujemne i mieć dodatnią sumę.")
    return w / w.sum()


def closeness(B, weights, D):
    """Odległości D+, D- oraz względna bliskość R_i dla wszystkich obiektów.

    Wzorce mają stałą postać: idealny [1, 0, ..., 0] (wzór 11) i antyidealny
    [0, ..., 0, 1] (wzór 12). Odległość między strukturami przekonań to metryka
    Jousselme'a (wzory 15-16), a agregacja po kryteriach - wzory (13)-(14).
    Gdy obiekt nie ma danych dla któregoś kryterium, wagi pozostałych
    kryteriów są renormalizowane.
    """
    dim = B.shape[2]
    s_plus = np.zeros(dim)
    s_plus[0] = 1.0
    s_minus = np.zeros(dim)
    s_minus[dim - 1] = 1.0

    diff_p = B - s_plus
    diff_m = B - s_minus
    d_p2 = np.clip(0.5 * np.einsum("ijk,kl,ijl->ij", diff_p, D, diff_p), 0, None)
    d_m2 = np.clip(0.5 * np.einsum("ijk,kl,ijl->ij", diff_m, D, diff_m), 0, None)

    valid = ~np.isnan(d_p2)
    W = np.where(valid, np.asarray(weights, dtype=float)[None, :], 0.0)
    row_sum = W.sum(axis=1, keepdims=True)
    W = np.divide(W, row_sum, out=np.zeros_like(W), where=row_sum > 0)

    D_plus = np.sqrt(np.sum(W * np.where(valid, d_p2, 0.0), axis=1))
    D_minus = np.sqrt(np.sum(W * np.where(valid, d_m2, 0.0), axis=1))
    denom = D_plus + D_minus
    R = np.divide(D_minus, denom, out=np.full_like(denom, np.nan), where=denom > 0)
    return D_plus, D_minus, R


def evaluate(df, obj_col, crit_col, state_cols, dk_col, weights=None, utilities=None):
    """Pełna procedura B-TOPSIS. Zwraca (ranking jako DataFrame, macierz D).

    weights   - słownik {kryterium: waga}; domyślnie wagi równe
    utilities - lista użyteczności ocen od najlepszej do najgorszej;
                domyślnie 1; 0,7; 0,4; 0 (dla skali czterostopniowej)
    """
    state_cols = list(state_cols)
    n = len(state_cols)
    if n < 2:
        raise ValueError("Potrzeba co najmniej dwóch kolumn ocen.")
    if utilities is None:
        utilities = default_utilities(n)
    if len(utilities) != n:
        raise ValueError("Liczba użyteczności musi być równa liczbie kolumn ocen.")

    objects, criteria, B = build_beliefs(df, obj_col, crit_col, state_cols, dk_col)
    if weights is None:
        weights = {c: 1.0 for c in criteria}
    w = normalize_weights(weights, criteria)
    D = similarity_matrix(utilities)
    d_plus, d_minus, R = closeness(B, w, D)

    results = pd.DataFrame(
        {
            obj_col: objects,
            "D_plus": np.round(d_plus, 4),
            "D_minus": np.round(d_minus, 4),
            "R_i": np.round(R, 4),
        }
    )
    results["Rank"] = results["R_i"].rank(ascending=False, method="min").astype("Int64")
    results = results.sort_values(by="Rank").reset_index(drop=True)
    return results, D
