"""Генерация синтетического журнала кредитных заявок.

Реальный журнал скоринга заменён моделью: для каждого класса риска
задано распределение числовых параметров и вероятности проявления
признаков. Шум делает данные реалистичными: признаки проявляются
не всегда, параметры варьируются. Фиксированный seed обеспечивает
воспроизводимость.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from domain import SYMPTOMS, PROCESS

RNG = np.random.default_rng(42)
N = 600

# Числовые параметры: (среднее, СКО) «среднего заёмщика».
DEFAULT = {
    "income":    (100, 40),
    "debt_load": (35, 15),
    "age":       (38, 12),
}

# Сдвиг числовых параметров в зависимости от класса риска.
# Внимание: bad_history и manual НЕ сдвигают параметры — они определяются
# категориальными признаками (кредитной историей и целью кредита),
# поэтому по доходу/ПДН/возрасту неотличимы от среднего заёмщика.
SHIFT = {
    "low_risk":    {"income": (130, 35), "debt_load": (18, 8),  "age": (42, 9)},
    "medium_risk": {"income": (75,  25), "debt_load": (42, 10), "age": (35, 10)},
    "high_risk":   {"income": (40,  15), "debt_load": (65, 10), "age": (28, 7)},
    "bad_history": {},
    "manual":      {},
}

# Вероятности «дефектных» (ненормальных) значений признаков при данном классе.
# Нормальные значения: good / stable / yes / no / consumer.
SYMPT = {
    "low_risk": {},
    "medium_risk": {
        "employment":     {"unstable": 0.35},
        "collateral":     {"no": 0.45},
        "purpose":        {"real_estate": 0.55},
    },
    "high_risk": {
        "employment":     {"unstable": 0.80},
        "collateral":     {"no": 0.75},
        "dependents":     {"yes": 0.55},
        "credit_history": {"none": 0.35},
    },
    "bad_history": {
        "credit_history": {"bad": 0.92},
    },
    "manual": {
        "purpose":        {"business": 0.85},
    },
}

NORMAL = {
    "credit_history": "good",
    "employment":     "stable",
    "collateral":     "yes",
    "dependents":     "no",
    "purpose":        "consumer",
}

# Приоры классов (реалистичная структура потока заявок банка).
PRIOR = {
    "low_risk":    0.40,
    "medium_risk": 0.20,
    "high_risk":   0.15,
    "bad_history": 0.10,
    "manual":      0.15,
}


def sample_symptom(risk_class, attr):
    spec = SYMPT[risk_class].get(attr)
    if spec:                                # явно заданное проявление признака
        value, p = next(iter(spec.items()))
        if RNG.random() < p:
            return value
        return NORMAL[attr]
    # фоновый шум: с вероятностью 0.1 — случайное ненормальное значение
    others = [v for v in SYMPTOMS[attr]["values"] if v != NORMAL[attr]]
    return RNG.choice(others) if RNG.random() < 0.1 else NORMAL[attr]


def clip(value, lo, hi):
    return max(lo, min(hi, value))


def generate(n=N):
    risk_classes = RNG.choice(list(PRIOR), size=n, p=list(PRIOR.values()))
    rows = []
    for c in risk_classes:
        row = {a: sample_symptom(c, a) for a in SYMPTOMS}
        for p, (mu, sd) in {**DEFAULT, **SHIFT[c]}.items():
            lo, hi = PROCESS[p]["range"]
            row[p] = round(clip(float(RNG.normal(mu, sd)), lo, hi), 1)
        row["risk_class"] = c
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Генерация журнала кредитных заявок.")
    ap.add_argument("--n", type=int, default=N,
                    help="Число записей (по умолчанию 600, минимум 500)")
    ap.add_argument("--out", default="data/credit_log.csv")
    args = ap.parse_args()

    if args.n < 500:
        raise SystemExit("По заданию требуется не менее 500 записей.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df = generate(args.n)
    df.to_csv(args.out, index=False, encoding="utf-8")

    print(df.head().to_string(), "\n")
    print(df["risk_class"].value_counts().to_string())
    print(f"\nВсего записей: {len(df)}, сохранено в {args.out}")