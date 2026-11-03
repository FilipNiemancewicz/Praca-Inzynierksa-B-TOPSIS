import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="B-TOPSIS Analyzer", page_icon="📊", layout="wide")

# --- ALGORYTM B-TOPSIS ---
def default_utilities(n):
    """Domyślne użyteczności ocen: 1; 0,7; 0,4; 0 dla skali 4-stopniowej."""
    if n == 4:
        return [1.0, 0.7, 0.4, 0.0]
    return [float(x) for x in np.linspace(1.0, 0.0, n)]


def similarity_matrix(utilities):
    """Macierz podobieństwa ocen D (N x N): s_ij = 1 - |U(H_i) - U(H_j)|  (wzór 8)."""
    u = np.asarray(utilities, dtype=float)
    return 1.0 - np.abs(u[:, None] - u[None, :])


def build_beliefs(df, obj_col, crit_col, state_cols, dk_col):
    """Buduje znormalizowaną macierz decyzyjną przekonań (obiekty x kryteria x N).

    Niewiedza ("nie wiem") jest rozdzielana proporcjonalnie do udziałów ocen (POP, wzór 5).
    """
    df = df.drop_duplicates(subset=[obj_col, crit_col])
    objects = list(df[obj_col].unique())
    criteria = list(df[crit_col].unique())
    n = len(state_cols)

    B = np.full((len(objects), len(criteria), n), np.nan)
    obj_idx = {o: i for i, o in enumerate(objects)}
    crit_idx = {c: j for j, c in enumerate(criteria)}

    for _, row in df.iterrows():
        vals = row[state_cols + [dk_col]].astype(float).values
        # dane w procentach (np. 0-100) -> udziały (0-1)
        if np.nansum(vals) > 1.5:
            vals = vals / 100.0
        m = vals[:n]
        total = m.sum()
        if total <= 0:
            continue  # brak jakichkolwiek ocen -> kryterium pomijane
        # POP: beta_k + beta_k * beta_H / sum(beta) = beta_k / sum(beta)
        B[obj_idx[row[obj_col]], crit_idx[row[crit_col]], :] = m / total

    return objects, criteria, B


def closeness(B, weights, D):
    """Odległości D+, D- oraz względna bliskość R_i dla wszystkich obiektów."""
    dim = B.shape[2]
    s_plus = np.zeros(dim)
    s_plus[0] = 1.0  # wzorzec idealny (wzór 11)
    s_minus = np.zeros(dim)
    s_minus[dim - 1] = 1.0  # wzorzec anty-idealny (wzór 12)

    diff_p = B - s_plus
    diff_m = B - s_minus
    d_p2 = np.clip(0.5 * np.einsum("ijk,kl,ijl->ij", diff_p, D, diff_p), 0, None)
    d_m2 = np.clip(0.5 * np.einsum("ijk,kl,ijl->ij", diff_m, D, diff_m), 0, None)

    valid = ~np.isnan(d_p2)
    W = np.where(valid, weights[None, :], 0.0)
    W = W / W.sum(axis=1, keepdims=True)  # wagi renormalizowane przy brakach

    D_plus = np.sqrt(np.sum(W * np.where(valid, d_p2, 0.0), axis=1))  # wzór (13)
    D_minus = np.sqrt(np.sum(W * np.where(valid, d_m2, 0.0), axis=1))  # wzór (14)
    denom = D_plus + D_minus
    R = np.divide(D_minus, denom, out=np.full_like(denom, np.nan), where=denom > 0)
    return D_plus, D_minus, R


def normalize_weights(weights, criteria):
    w = np.array([weights[c] for c in criteria], dtype=float)
    return w / w.sum()


# --- INTERFEJS STREAMLIT ---
st.title("📊 Wielokryterialna Analiza Decyzyjna B-TOPSIS")
st.markdown(
    "Moduł analityczny metody **Belief-TOPSIS** oparty na teorii struktur"
    " przekonań Dempstera-Shafera i **metryce Jousselme'a**."
)

with st.sidebar:
    st.header("⚙️ 1. Wczytanie danych")
    uploaded_file = st.file_uploader(
        "Wgraj plik (CSV lub Excel)", type=["csv", "xlsx", "xls"]
    )

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("📋 Podgląd wczytanych danych")
        st.dataframe(df, use_container_width=True, height=350)

        col1, col2 = st.columns(2)
        cols = list(df.columns)

        with col1:
            st.markdown("### Mapowanie kolumn strukturalnych")
            obj_col = st.selectbox("Kolumna obiektów (np. Kraj, Miasto):", cols, index=0)
            crit_col = st.selectbox(
                "Kolumna kryteriów (np. Pytanie):",
                cols,
                index=1 if len(cols) > 1 else 0,
            )

        with col2:
            st.markdown("### Mapowanie ocen ankietowych")
            remaining_cols = [c for c in cols if c not in [obj_col, crit_col]]
            dont_know_col = st.selectbox(
                "Kolumna niepewności (np. 'Nie wiem'):",
                remaining_cols,
                index=len(remaining_cols) - 1 if remaining_cols else 0,
            )
            state_cols_candidates = [c for c in remaining_cols if c != dont_know_col]
            state_cols = st.multiselect(
                "Kolumny ocen od NAJLEPSZEJ do NAJGORSZEJ (uporządkowane):",
                state_cols_candidates,
                default=state_cols_candidates,
            )

        if len(state_cols) < 2:
            st.warning("Wybierz co najmniej 2 kolumny ocen (np. dobra i zła).")
        else:
            n_states = len(state_cols)

            with st.sidebar:
                st.markdown("---")
                st.header("⚖️ 2. Parametry metody")

                st.caption(
                    "Użyteczności ocen (od najlepszej do najgorszej). Z nich"
                    " powstaje macierz podobieństwa: s_ij = 1 − |U_i − U_j|."
                )
                defaults = default_utilities(n_states)
                utilities = []
                for idx, state in enumerate(state_cols):
                    if idx == 0 or idx == n_states - 1:
                        val = defaults[idx]
                        st.number_input(
                            f"U({state})",
                            value=float(val),
                            disabled=True,
                            key=f"u_fixed_{idx}_{n_states}",
                        )
                        utilities.append(float(val))
                    else:
                        utilities.append(
                            st.number_input(
                                f"U({state})",
                                min_value=0.0,
                                max_value=1.0,
                                value=float(defaults[idx]),
                                step=0.05,
                                key=f"u_{idx}_{n_states}",
                            )
                        )
                if any(utilities[i] <= utilities[i + 1] for i in range(n_states - 1)):
                    st.warning(
                        "Użyteczności powinny maleć od najlepszej do najgorszej oceny."
                    )

            st.markdown("---")
            st.subheader("🎯 Konfiguracja wag kryteriów (pytań)")
            criteria_list = df[crit_col].unique()

            crit_cols_ui = st.columns(min(len(criteria_list), 4))
            weights = {}
            for idx, crit in enumerate(criteria_list):
                with crit_cols_ui[idx % len(crit_cols_ui)]:
                    weights[crit] = st.number_input(
                        f"Waga: {crit}",
                        min_value=0.01,
                        max_value=10.0,
                        value=1.0,
                        step=0.1,
                    )

            if st.button("🚀 Uruchom B-TOPSIS", type="primary"):
                objects, criteria, B = build_beliefs(
                    df, obj_col, crit_col, state_cols, dont_know_col
                )
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
                results["Rank"] = results["R_i"].rank(ascending=False, method="min").astype(int)
                results = results.sort_values(by="Rank").reset_index(drop=True)

                st.success("✅ Obliczenia zakończone sukcesem!")

                res_col, chart_col = st.columns([1, 1.2])

                with res_col:
                    st.subheader("🏆 Ranking końcowy")
                    st.dataframe(results, use_container_width=True, height=500)
                    csv = results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Pobierz wyniki jako CSV",
                        data=csv,
                        file_name="wyniki_btopsis.csv",
                        mime="text/csv",
                    )

                with chart_col:
                    st.subheader("📈 Wykres względnej bliskości ($R_i$)")
                    fig, ax = plt.subplots(figsize=(8, max(5, len(results) * 0.28)))
                    y_pos = np.arange(len(results))
                    ax.barh(
                        y_pos,
                        results["R_i"],
                        color="#2b5c8f",
                        edgecolor="black",
                        alpha=0.85,
                    )
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(results[obj_col])
                    ax.invert_yaxis()
                    ax.set_xlabel("Wskaźnik bliskości R_i")
                    ax.axvline(
                        0.5,
                        color="red",
                        linestyle="--",
                        alpha=0.6,
                        label="Próg neutralny (0.5)",
                    )
                    ax.grid(axis="x", linestyle=":", alpha=0.6)
                    ax.legend(loc="lower right")
                    st.pyplot(fig)

                with st.expander("🔍 Zobacz macierz podobieństwa (D)"):
                    st.write(f"Wymiar macierzy: {D.shape[0]}x{D.shape[1]}")
                    df_D = pd.DataFrame(D, index=state_cols, columns=state_cols)
                    st.dataframe(df_D.style.format("{:.3f}"))

    except Exception as e:
        st.error(f"Wystąpił błąd podczas przetwarzania: {str(e)}")
else:
    st.info("👈 Wgraj plik z danymi w panelu bocznym, aby rozpocząć analizę.")