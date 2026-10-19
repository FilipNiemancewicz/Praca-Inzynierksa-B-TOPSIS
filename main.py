import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# 1. KLASA MODELU B-TOPSIS
# -----------------------------------------------------------------------------
class BTOPSIS:
    """
    Implementacja metody Belief-TOPSIS opartej na teorii dowodów Dempstera-Shafera
    oraz odległości Jousselme'a.
    """

    def __init__(self, n_crit=6, weights=None):
        if weights is None:
            self.w = np.ones(n_crit) / n_crit
        else:
            self.w = np.array(weights, dtype=float) / np.sum(weights)

    def _jousselme_distance(self, m1, m2):
        """
        Odległość Jousselme'a w 5-elementowej ramie:
        [H1 (b.dobra), H2 (raczej dobra), H3 (raczej zła), H4 (b.zła), Theta (Nie wiem)]
        """
        diff = m1 - m2
        # Macierz podobieństwa D (indeks Jaccarda: |A ∩ B| / |A ∪ B|)
        D = np.eye(5)
        for i in range(4):
            D[i, 4] = 0.25
            D[4, i] = 0.25

        dist_sq = 0.5 * np.dot(np.dot(diff, D), diff)
        return np.sqrt(max(0.0, dist_sq))

    def evaluate(self, decision_data):
        countries = list(decision_data.keys())
        criteria = list(decision_data[countries[0]].keys())
        m = len(countries)
        n = len(criteria)

        # Punkty odniesienia dla stymulant:
        # S+ (wzorzec): 100% przekonania w ocenę H1 (Very good)
        # S- (antywzorzec): 100% przekonania w ocenę H4 (Very bad)
        s_plus = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        s_minus = np.array([0.0, 0.0, 0.0, 1.0, 0.0])

        d_plus = np.zeros(m)
        d_minus = np.zeros(m)

        for i, country in enumerate(countries):
            sum_pos = 0.0
            sum_neg = 0.0
            for j, crit in enumerate(criteria):
                m_ij = decision_data[country][crit]
                sum_pos += self.w[j] * (self._jousselme_distance(m_ij, s_plus) ** 2)
                sum_neg += self.w[j] * (self._jousselme_distance(m_ij, s_minus) ** 2)
            d_plus[i] = np.sqrt(sum_pos)
            d_minus[i] = np.sqrt(sum_neg)

        # Współczynnik względnej bliskości R_i = d- / (d+ + d-)
        closeness = d_minus / (d_plus + d_minus)

        res = pd.DataFrame({
            'Country': countries,
            'd_plus': np.round(d_plus, 4),
            'd_minus': np.round(d_minus, 4),
            'R_i': np.round(closeness, 4)
        })
        res['Rank'] = res['R_i'].rank(ascending=False, method='min').astype(int)
        return res.sort_values(by='Rank').reset_index(drop=True)


# -----------------------------------------------------------------------------
# 2. FUNKCJA WCZYTYWANIA I WYBORU PLIKU
# -----------------------------------------------------------------------------
def get_dataset_path():
    """Pozwala wybrać plik przez okno dialogowe lub konsolę."""
    print("=" * 65)
    print("  MODUŁ ANALITYCZNY B-TOPSIS - WYBÓR PLIKU DANYCH")
    print("=" * 65)

    filepath = None
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        print("Wybierz plik Excela w otwartym oknie dialogowym...")
        filepath = filedialog.askopenfilename(
            title="Wybierz plik z danymi Eurobarometru",
            filetypes=[("Pliki Excel", "*.xlsx *.xls")]
        )
    except Exception:
        pass

    if not filepath:
        default_file = 'DANE MOZLIWE.xlsx'
        prompt = f"Podaj ścieżkę do pliku Excela (Enter = domyślnie '{default_file}'): "
        user_input = input(prompt).strip()
        filepath = user_input if user_input else default_file

    if not os.path.exists(filepath):
        print(f"\n[BŁĄD] Plik '{filepath}' nie został znaleziony!")
        sys.exit(1)

    print(f"\n[OK] Załadowano plik: {os.path.abspath(filepath)}")
    return filepath


# -----------------------------------------------------------------------------
# 3. PARSER DANYCH Z BADANIA EUROBAROMETR
# -----------------------------------------------------------------------------
def load_and_parse_data(filepath):
    xls = pd.ExcelFile(filepath)

    # 6 kluczowych kryteriów diagnostycznych
    criteria_sheets = {
        'C1_Situation_General': 'QA1_1',
        'C2_Economy': 'QA1_2',
        'C3_Job_Situation': 'QA1_4',
        'C4_Household_Finances': 'QA1_5',
        'C5_Employment': 'QA1_6',
        'C6_Public_Services': 'QA1_7'
    }

    # Sprawdzenie obecności arkuszy
    available_sheets = xls.sheet_names
    for name, sheet in criteria_sheets.items():
        if sheet not in available_sheets:
            print(f"[BŁĄD] W pliku brakuje arkusza: '{sheet}'!")
            sys.exit(1)

    # Lista 27 państw członkowskich UE
    eu_27 = [
        'BE', 'BG', 'CZ', 'DK', 'DE', 'EE', 'IE', 'EL', 'ES', 'FR',
        'HR', 'IT', 'CY', 'LV', 'LT', 'LU', 'HU', 'MT', 'NL', 'AT',
        'PL', 'PT', 'RO', 'SI', 'SK', 'FI', 'SE'
    ]

    # Automatyczne wykrycie wiersza z kodami państw w pierwszym arkuszu
    df_sample = pd.read_excel(filepath, sheet_name=criteria_sheets['C1_Situation_General'], header=None)
    header_row_idx = None
    for r in range(15):
        row_vals = [str(x).strip() for x in df_sample.iloc[r].values]
        if 'BE' in row_vals and 'PL' in row_vals:
            header_row_idx = r
            break

    if header_row_idx is None:
        print("[BŁĄD] Nie odnaleziono wiersza z kodami krajów UE w arkuszu!")
        sys.exit(1)

    header = [str(x).replace('\n', ' ').strip() for x in df_sample.iloc[header_row_idx].values]

    # Wykrycie wierszy z kategoriami odpowiedzi w kolumnie 1
    categories = ['Very good', 'Rather good', 'Rather bad', 'Very bad', "Don't know"]
    cat_rows = {}
    for r in range(header_row_idx, min(header_row_idx + 20, len(df_sample))):
        val = str(df_sample.iloc[r, 1]).strip()
        if val in categories:
            cat_rows[val] = r

    if len(cat_rows) < 5:
        print("[BŁĄD] Nie odnaleziono wszystkich 5 kategorii odpowiedzi w tabeli!")
        sys.exit(1)

    print("Parsowanie arkuszy kryteriów diagnostycznych...")
    decision_data = {c: {} for c in eu_27}

    for crit_name, sheet in criteria_sheets.items():
        df_sheet = pd.read_excel(filepath, sheet_name=sheet, header=None)
        for country in eu_27:
            c_idx = header.index(country)
            vals = []
            for cat in categories:
                r_idx = cat_rows[cat]
                v = df_sheet.iloc[r_idx, c_idx]
                if str(v).strip() in ['-', 'nan', '']:
                    vals.append(0.0)
                else:
                    vals.append(float(v))

            vec = np.array(vals)
            # Normalizacja do jedności (zabezpieczenie przed błędami zaokrągleń)
            if vec.sum() > 0:
                vec = vec / vec.sum()
            decision_data[country][crit_name] = vec

    return decision_data, criteria_sheets


# -----------------------------------------------------------------------------
# 4. GŁÓWNY PROGRAM I WIZUALIZACJA
# -----------------------------------------------------------------------------
def main():
    filepath = get_dataset_path()
    decision_data, criteria = load_and_parse_data(filepath)

    print("\nUruchamianie algorytmu B-TOPSIS...")
    model = BTOPSIS(n_crit=len(criteria))
    ranking_df = model.evaluate(decision_data)

    print("\n" + "=" * 65)
    print("  OSTATECZNY RANKING PAŃSTW UE (B-TOPSIS)")
    print("=" * 65)
    print(ranking_df.to_string(index=False))

    # Zapis tabeli wynikowej do CSV / Excel
    out_csv = "ranking_btopsis_ue27.csv"
    ranking_df.to_csv(out_csv, index=False, sep=';', decimal=',')
    print(f"\n[INFO] Wyniki zapisano do pliku: {out_csv}")

    # Wykres słupkowy na potrzeby prezentacji i pracy
    plt.figure(figsize=(12, 6))
    colors = ['#1f77b4' if c != 'PL' else '#d62728' for c in ranking_df['Country']]
    bars = plt.bar(ranking_df['Country'], ranking_df['R_i'], color=colors, edgecolor='black', width=0.7)

    plt.axhline(ranking_df['R_i'].mean(), color='gray', linestyle='--',
                label=f"Średnia UE ({ranking_df['R_i'].mean():.4f})")
    plt.title('Ranking państw Unii Europejskiej wg współczynnika bliskości B-TOPSIS ($R_i$)', fontsize=13, pad=12)
    plt.xlabel('Państwo członkowskie UE', fontsize=11)
    plt.ylabel('Współczynnik bliskości $R_i$', fontsize=11)
    plt.ylim(0.40, 0.60)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    out_img = "wykres_ranking_btopsis.png"
    plt.savefig(out_img, dpi=300)
    print(f"[INFO] Wykres zapisano do pliku: {out_img}")
    plt.show()


if __name__ == '__main__':
    main()