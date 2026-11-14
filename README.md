# Aplikacja B-TOPSIS

Aplikacja webowa (Streamlit) implementująca metodę Belief-TOPSIS (B-TOPSIS)
do wielokryterialnej analizy danych ankietowych.

## Uruchomienie

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

W panelu bocznym wgraj plik, np. `dane_przykladowe.csv`.

## Struktura projektu

| Plik | Zawartość |
|---|---|
| `btopsis.py` | Obliczenia metody (bez zależności od interfejsu) |
| `app.py` | Interfejs Streamlit |
| `tests/test_btopsis.py` | Testy jednostkowe |
| `dane_przykladowe.csv` | Przykładowe dane (Zürich z artykułu + dwa obiekty ilustracyjne) |
| `requirements.txt`, `requirements-dev.txt` | Zależności (dev: dodatkowo pytest) |

## Format danych

Format długi: jeden wiersz = jedna para obiekt–kryterium.
Kolumny: obiekt, kryterium, kolejne oceny (od najlepszej do najgorszej) i kolumna „nie wiem".
Wartości mogą być udziałami (0–1) lub procentami (0–100). Wszystkie kryteria muszą mieć
tę samą orientację skali (pierwsza kolumna = ocena najlepsza); kryteria o odwróconej
skali trzeba przekształcić przed wczytaniem.

## Testy

```bash
pip install -r requirements-dev.txt
pytest
```

## Metoda

Niewiedza ("nie wiem") jest rozdzielana proporcjonalnie do udziałów ocen (POP),
macierz podobieństwa powstaje z użyteczności ocen (s_ij = 1 − |U_i − U_j|),
odległości liczone są metryką Jousselme'a od stałych wzorców: idealnego i antyidealnego.
Domyślne użyteczności dla skali czterostopniowej: 1; 0,7; 0,4; 0.
