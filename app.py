"""Interfejs Streamlit aplikacji B-TOPSIS.

Uruchomienie:  streamlit run app.py
Obliczenia wykonuje moduł btopsis.py; ten plik odpowiada tylko za interfejs.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from btopsis import default_utilities, evaluate

st.set_page_config(page_title="B-TOPSIS Analyzer", page_icon="📊", layout="wide")

st.title("📊 Wielokryterialna Analiza Decyzyjna B-TOPSIS")
st.markdown(
    "Moduł analityczny metody **Belief-TOPSIS** oparty na teorii struktur"
    " przekonań Dempstera-Shafera i **metryce Jousselme'a**."
)


def show_run(run):
    """Wyświetla wyniki jednego uruchomienia (ranking, wykres, macierz D)."""
    results, D, obj_col = run["results"], run["D"], run["obj_col"]

    res_col, chart_col = st.columns([1, 1.2])

    with res_col:
        st.subheader("🏆 Ranking końcowy")
        st.dataframe(results, use_container_width=True, height=500)
        st.download_button(
            label="📥 Pobierz wyniki jako CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="wyniki_btopsis.csv",
            mime="text/csv",
            key=f"download_{run['name']}",
        )

    with chart_col:
        st.subheader("📈 Wykres względnej bliskości ($R_i$)")
        fig, ax = plt.subplots(figsize=(8, max(5, len(results) * 0.28)))
        y_pos = np.arange(len(results))
        ax.barh(y_pos, results["R_i"], color="#2b5c8f", edgecolor="black", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(results[obj_col])
        ax.invert_yaxis()
        ax.set_xlabel("Wskaźnik bliskości R_i")
        ax.axvline(0.5, color="red", linestyle="--", alpha=0.6, label="Próg neutralny (0.5)")
        ax.grid(axis="x", linestyle=":", alpha=0.6)
        ax.legend(loc="lower right")
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("🔍 Zobacz macierz podobieństwa (D)"):
        st.write(f"Wymiar macierzy: {D.shape[0]}x{D.shape[1]}")
        labels = run["state_cols"]
        st.dataframe(pd.DataFrame(D, index=labels, columns=labels).style.format("{:.3f}"))


def show_comparison(runs):
    """Porównuje rankingi z kolejnych uruchomień (analiza wrażliwości)."""
    st.subheader("🔁 Porównanie scenariuszy")

    params = pd.DataFrame(
        {
            "Scenariusz": [r["name"] for r in runs],
            "Użyteczności": [", ".join(f"{u:.2f}" for u in r["utilities"]) for r in runs],
            "Wagi": [
                "równe" if len(set(r["weights"].values())) == 1 else "zmienione"
                for r in runs
            ],
        }
    )
    st.dataframe(params, use_container_width=True, hide_index=True)

    obj_col = runs[0]["obj_col"]
    ranks = pd.DataFrame(
        {r["name"]: r["results"].set_index(obj_col)["Rank"].astype(float) for r in runs}
    ).sort_values(by=runs[0]["name"])
    st.caption("Pozycje w rankingu dla kolejnych scenariuszy:")
    st.dataframe(ranks, use_container_width=True)

    st.caption("Współczynnik korelacji rang Spearmana między scenariuszami:")
    st.dataframe(ranks.corr(method="spearman").style.format("{:.3f}"))


with st.sidebar:
    st.header("⚙️ 1. Wczytanie danych")
    uploaded_file = st.file_uploader(
        "Wgraj plik (CSV lub Excel)", type=["csv", "xlsx", "xls"]
    )

if uploaded_file is not None:
    try:
        # nowy plik -> nowa seria scenariuszy
        if st.session_state.get("file_name") != uploaded_file.name:
            st.session_state["file_name"] = uploaded_file.name
            st.session_state["runs"] = []

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
                    st.warning("Użyteczności powinny maleć od najlepszej do najgorszej oceny.")

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

            runs = st.session_state["runs"]

            if st.button("🚀 Uruchom B-TOPSIS", type="primary"):
                results, D = evaluate(
                    df, obj_col, crit_col, state_cols, dont_know_col,
                    weights=weights, utilities=utilities,
                )
                runs.append(
                    {
                        "name": f"Scenariusz {len(runs) + 1}",
                        "results": results,
                        "D": D,
                        "obj_col": obj_col,
                        "state_cols": list(state_cols),
                        "utilities": list(utilities),
                        "weights": dict(weights),
                    }
                )
                st.success("✅ Obliczenia zakończone sukcesem!")

            if runs:
                st.markdown("---")
                head_col, clear_col = st.columns([4, 1])
                with head_col:
                    chosen = st.selectbox(
                        "Pokaż wyniki scenariusza:",
                        options=list(range(len(runs))),
                        index=len(runs) - 1,
                        format_func=lambda i: runs[i]["name"],
                        key=f"view_run_{len(runs)}",
                    )
                with clear_col:
                    if st.button("🗑️ Wyczyść scenariusze"):
                        st.session_state["runs"] = []
                        st.rerun()

                show_run(runs[chosen])

                if len(runs) >= 2:
                    show_comparison(runs)

    except Exception as e:
        st.error(f"Wystąpił błąd podczas przetwarzania: {str(e)}")
else:
    st.info("👈 Wgraj plik z danymi w panelu bocznym, aby rozpocząć analizę.")
