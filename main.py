import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Universal B-TOPSIS Analyzer", page_icon="📊", layout="wide"
)


# --- KLASA ALGORYTMU B-TOPSIS ---
class UniversalBTOPSIS:

  def __init__(
      self,
      criteria_weights=None,
      state_utilities=None,
      use_classic_jaccard=False,
  ):
    self.criteria_weights = criteria_weights
    self.state_utilities = state_utilities
    self.use_classic_jaccard = use_classic_jaccard

  def _build_similarity_matrix(self, num_states):
    total_dim = num_states + 1
    D = np.eye(total_dim)
    uncertainty_weight = 1.0 / num_states

    # Przypisanie dla stanu niepewności (Don't know / Theta)
    for i in range(num_states):
      D[i, num_states] = uncertainty_weight
      D[num_states, i] = uncertainty_weight

    # Jeśli NIE wybrano klasycznego Jaccarda, budujemy macierz na podstawie wartości ocen: D_ij = 1 - |u_i - u_j|
    if (
        not self.use_classic_jaccard
        and self.state_utilities is not None
        and len(self.state_utilities) == num_states
    ):
      for i in range(num_states):
        for j in range(num_states):
          diff = abs(self.state_utilities[i] - self.state_utilities[j])
          D[i, j] = max(0.0, 1.0 - diff)

    return D

  def _jousselme_distance(self, m1, m2, D):
    diff = m1 - m2
    dist_sq = 0.5 * np.dot(np.dot(diff, D), diff)
    return np.sqrt(max(0.0, dist_sq))

  def evaluate(self, df, obj_col, crit_col, state_cols, dont_know_col):
    objects = df[obj_col].unique()
    criteria = df[crit_col].unique()
    num_states = len(state_cols)
    D = self._build_similarity_matrix(num_states)

    # Normalizacja wag kryteriów
    if self.criteria_weights is None:
      w = {c: 1.0 / len(criteria) for c in criteria}
    else:
      total_w = sum(self.criteria_weights.values())
      w = {c: self.criteria_weights[c] / total_w for c in criteria}

    # Wzorzec idealny (PIS / s+) - stan o najwyższej użyteczności
    s_plus = np.zeros(num_states + 1)
    s_plus[0] = 1.0

    # Wzorzec anty-idealny (NIS / s-) - stan o najniższej użyteczności
    s_minus = np.zeros(num_states + 1)
    s_minus[num_states - 1] = 1.0

    d_plus = []
    d_minus = []

    for obj in objects:
      df_obj = df[df[obj_col] == obj]
      sum_pos = 0.0
      sum_neg = 0.0

      for crit in criteria:
        row = df_obj[df_obj[crit_col] == crit]
        if row.empty:
          continue
        m_vec = row[state_cols + [dont_know_col]].values.flatten().astype(float)

        # Normalizacja wektora jeśli wartości w pliku są w procentach (np. 0-100)
        if np.sum(m_vec) > 1.5:
          m_vec = m_vec / 100.0

        d_p = self._jousselme_distance(m_vec, s_plus, D)
        d_m = self._jousselme_distance(m_vec, s_minus, D)

        sum_pos += w[crit] * (d_p**2)
        sum_neg += w[crit] * (d_m**2)

      d_plus.append(np.sqrt(sum_pos))
      d_minus.append(np.sqrt(sum_neg))

    d_plus = np.array(d_plus)
    d_minus = np.array(d_minus)
    closeness = d_minus / (d_plus + d_minus)

    results = pd.DataFrame({
        obj_col: objects,
        'd_plus': np.round(d_plus, 4),
        'd_minus': np.round(d_minus, 4),
        'R_i': np.round(closeness, 4),
    })
    results['Rank'] = (
        results['R_i'].rank(ascending=False, method='min').astype(int)
    )
    return results.sort_values(by='Rank').reset_index(drop=True), D


# --- INTERFEJS STREAMLIT ---
st.title('📊 Wielokryterialna Analiza Decyzyjna B-TOPSIS')
st.markdown(
    'Uniwersalny moduł analityczny oparty na **Teorii Dempstera-Shafera** i'
    " **metryce Jousselme'a**."
)

with st.sidebar:
  st.header('⚙️ 1. Wczytanie Danych')
  uploaded_file = st.file_uploader(
      'Wgraj plik (CSV lub Excel)', type=['csv', 'xlsx', 'xls']
  )

if uploaded_file is not None:
  try:
    if uploaded_file.name.endswith('.csv'):
      df = pd.read_csv(uploaded_file)
    else:
      df = pd.read_excel(uploaded_file)

    st.subheader('📋 Podgląd wczytanych danych')
    st.dataframe(df, use_container_width=True, height=350)

    col1, col2 = st.columns(2)
    cols = list(df.columns)

    with col1:
      st.markdown('### Mapowanie kolumn strukturalnych')
      obj_col = st.selectbox(
          'Kolumna obiektów (np. Kraj, Miasto):', cols, index=0
      )
      crit_col = st.selectbox(
          'Kolumna kryteriów (np. Pytanie):', cols, index=1 if len(cols) > 1 else 0
      )

    with col2:
      st.markdown('### Mapowanie ocen ankietowych')
      remaining_cols = [c for c in cols if c not in [obj_col, crit_col]]
      dont_know_col = st.selectbox(
          "Kolumna niepewności (np. 'Nie wiem' / 'Don\\'t know'):",
          remaining_cols,
          index=len(remaining_cols) - 1 if remaining_cols else 0,
      )
      state_cols_candidates = [c for c in remaining_cols if c != dont_know_col]
      state_cols = st.multiselect(
          'Kolumny ocen od NAJLEPSZEJ do NAJGORSZEJ (uporządkowane):',
          state_cols_candidates,
          default=state_cols_candidates,
      )

    if len(state_cols) < 2:
      st.warning('Wybierz co najmniej 2 kolumny ocen (np. dobra i zła).')
    else:
      with st.sidebar:
        st.markdown('---')
        st.header('⚖️ 2. Macierz Podobieństwa i Oceny')

        # Domyślnie zaznaczony tryb klasyczny Jaccarda (1 na przekątnej, 0 poza, 0.25 niepewność)
        classic_mode = st.checkbox(
            'Użyj klasycznej macierzy Jaccarda (domyślna)',
            value=True,
            help=(
                'Włącza bazową macierz ze skrajną rozłącznością ocen (1 na'
                ' przekątnej, 0 poza nią i 0.25 dla niepewności).'
            ),
        )

        state_scores = []
        if not classic_mode:
          st.caption(
              'Przypisz użyteczność/punkty poszczególnym ocenom (od 1.0 do 0.0):'
          )
          default_scores = [1.0, 0.7, 0.3, 0.0]
          for idx, state in enumerate(state_cols):
            fallback_val = (
                default_scores[idx]
                if idx < len(default_scores)
                else max(0.0, 1.0 - idx * (1.0 / (len(state_cols) - 1)))
            )
            score = st.number_input(
                f'Wartość: {state}',
                min_value=0.0,
                max_value=1.0,
                value=float(fallback_val),
                step=0.05,
            )
            state_scores.append(score)

      st.markdown('---')
      st.subheader('🎯 Konfiguracja Wag Kryteriów (Pytań)')
      criteria_list = df[crit_col].unique()

      crit_cols_ui = st.columns(min(len(criteria_list), 4))
      weights = {}
      for idx, crit in enumerate(criteria_list):
        with crit_cols_ui[idx % len(crit_cols_ui)]:
          weights[crit] = st.number_input(
              f'Waga: {crit}',
              min_value=0.01,
              max_value=10.0,
              value=1.0,
              step=0.1,
          )

      if st.button('🚀 Uruchom B-TOPSIS', type='primary'):
        engine = UniversalBTOPSIS(
            criteria_weights=weights,
            state_utilities=state_scores,
            use_classic_jaccard=classic_mode,
        )
        results, D = engine.evaluate(
            df, obj_col, crit_col, state_cols, dont_know_col
        )

        st.success('✅ Obliczenia zakończone sukcesem!')

        res_col, chart_col = st.columns([1, 1.2])

        with res_col:
          st.subheader('🏆 Ranking Końcowy')
          st.dataframe(results, use_container_width=True, height=500)

          csv = results.to_csv(index=False).encode('utf-8')
          st.download_button(
              label='📥 Pobierz wyniki jako CSV',
              data=csv,
              file_name='wyniki_btopsis.csv',
              mime='text/csv',
          )

        with chart_col:
          st.subheader('📈 Wykres Względnej Bliskości ($R_i$)')
          fig, ax = plt.subplots(figsize=(8, max(5, len(results) * 0.28)))

          y_pos = np.arange(len(results))
          ax.barh(
              y_pos,
              results['R_i'],
              color='#2b5c8f',
              edgecolor='black',
              alpha=0.85,
          )

          ax.set_yticks(y_pos)
          ax.set_yticklabels(results[obj_col])
          ax.invert_yaxis()
          ax.set_xlabel('Wskaźnik bliskości R_i')
          ax.axvline(
              0.5,
              color='red',
              linestyle='--',
              alpha=0.6,
              label='Próg neutralny (0.5)',
          )
          ax.grid(axis='x', linestyle=':', alpha=0.6)
          ax.legend(loc='lower right')

          st.pyplot(fig)

        with st.expander('🔍 Zobacz wyznaczoną macierz podobieństwa (D)'):
          st.write(f'Wymiar macierzy: {D.shape[0]}x{D.shape[1]}')
          matrix_labels = state_cols + [dont_know_col]
          df_D = pd.DataFrame(D, index=matrix_labels, columns=matrix_labels)
          st.dataframe(df_D.style.format('{:.3f}'))

  except Exception as e:
    st.error(f'Wystąpił błąd podczas przetwarzania: {str(e)}')
else:
  st.info('👈 Wgraj plik z danymi w panelu bocznym, aby rozpocząć analizę.')