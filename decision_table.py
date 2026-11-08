"""Таблица решений эксперта и её проверка на полноту и противоречивость."""
from itertools import product
import pandas as pd
from domain import SYMPTOMS, CAUSES

ANY = "-"                              # «значение не важно»
ATTRS = list(SYMPTOMS)                 # порядок столбцов условий

# Каждая строка: одно правило эксперта: условия по признакам и заключение.
ROWS = [
    # credit_history  employment  collateral  dependents  purpose        -> cause
    ("R1", "bad",      ANY,        ANY,        ANY,        ANY,           "bad_history"),
    ("R2", "good",     "stable",   ANY,        "no",       "consumer",    "low_risk"),
    ("R3", "none",     "stable",   "yes",      ANY,        ANY,           "medium_risk"),
    ("R4", "none",     "unstable", "no",       ANY,        ANY,           "high_risk"),
    ("R5", "good",     "unstable", ANY,        "yes",      ANY,           "manual"),
    ("R6", "good",     "stable",   "no",       "yes",      "consumer",    "medium_risk"),
    ("R7", ANY,        ANY,        ANY,        ANY,        "business",    "manual"),
    ("R8", "good",     "stable",   "yes",      "no",       "real_estate", "low_risk"),
]

def load_table() -> pd.DataFrame:
    df = pd.DataFrame(ROWS, columns=['rule', *ATTRS, "cause"]).set_index("rule")
    # Синтаксический контроль: все значения должны быть из словаря понятий
    for a in ATTRS:
        bad = set(df[a]) - set(SYMPTOMS[a]["values"]) - {ANY}
        assert not bad, f"Неизвестные значения признака {a}: {bad}"
    assert set(df["cause"]) <= set(CAUSES), "Неизвестное заключение"
    return df

def matches(row: pd.Series, case: dict) -> bool:
    return all(row[a] in (ANY, case[a]) for a in ATTRS)

def verify(df: pd.DataFrame):
    """Перебирает все сочетания значений признаков (полный перебор
    допустим: пространство ситуаций мало) и классифицирует каждое
    сочетание."""
    uncovered, conflicts = [], []
    all_cases = [dict(zip(ATTRS, v))
                 for v in product(*(SYMPTOMS[a]["values"] for a in ATTRS))]
    for case in all_cases:
        fired = [r for r, row in df.iterrows() if matches(row, case)]
        causes = {df.loc[r, "cause"] for r in fired}
        if not fired:
            uncovered.append(case)
        elif len(causes) > 1:
            conflicts.append((case, fired))
    return all_cases, uncovered, conflicts

if __name__ == "__main__":
    df = load_table()
    df.to_csv("kb/decision_table.csv", encoding="utf-8")
    print(df.to_string(), "\n")
    all_cases, uncovered, conflicts = verify(df)
    n = len(all_cases)
    print(f"Всего ситуаций: {n}")
    print(f"Покрыто правилами: {n - len(uncovered)} ({(n - len(uncovered)) / n:.0%})")
    print(f"Не покрыто: {len(uncovered)}; "
          f"ситуаций с конфликтом заключений: {len(conflicts)}")
    for case, fired in conflicts[:3]:
        print("  конфликт", fired, "при", case)