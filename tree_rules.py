"""Извлечение знаний из данных: дерево решений -> продукционные правила."""
import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import cross_val_score
from domain import SYMPTOMS, PROCESS, CAUSES

df = pd.read_csv("data/credit_log.csv")
y = df["risk_class"]
X_proc = df[list(PROCESS)]                                    # числовые параметры
X_symp = pd.get_dummies(df[list(SYMPTOMS)], prefix_sep="=")   # one-hot признаков


def make_tree():
    # Ограничения глубины и размера листа: компромисс между точностью
    # и интерпретируемостью: каждое правило должно быть понятно эксперту.
    return DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                  random_state=0)


def extract_rules(model, feature_names, min_conf=0.6):
    """Обходит дерево в глубину; путь от корня до листа = одно правило.
    Условия по одному признаку объединяются в интервал."""
    t = model.tree_
    rules = []
    def walk(node, bounds):
        if t.feature[node] == _tree.TREE_UNDEFINED:                 # лист
            counts = t.value[node][0]
            cls = model.classes_[int(np.argmax(counts))]
            conf = counts.max() / counts.sum()
            support = int(t.n_node_samples[node])
            if conf >= min_conf:
                rules.append({"if": {f: [round(lo, 1) if lo is not None
                                         else None,
                                         round(hi, 1) if hi is not None
                                         else None]
                                     for f, (lo, hi) in bounds.items()},
                              "then": cls, "confidence": round(float(conf), 2),
                              "support": support})
            return
        f, thr = feature_names[t.feature[node]], float(t.threshold[node])
        lo, hi = bounds.get(f, (None, None))
        walk(t.children_left[node],  {**bounds, f: (lo, thr)})      # f <= thr
        walk(t.children_right[node], {**bounds, f: (thr, hi)})      # f > thr
    walk(0, {})
    return sorted(rules, key=lambda r: -r["support"])


def fmt(rule):
    parts = []
    for f, (lo, hi) in rule["if"].items():
        unit = PROCESS[f]["unit"]
        if lo is not None and hi is not None: parts.append(f"{lo} < {f} <= {hi} {unit}")
        elif lo is not None:                  parts.append(f"{f} > {lo} {unit}")
        else:                                 parts.append(f"{f} <= {hi} {unit}")
    return (f"ЕСЛИ {' И '.join(parts)} ТО {CAUSES[rule['then']]['name']} "
            f" (дост. {rule['confidence']}, поддержка {rule['support']})")


if __name__ == "__main__":
    for name, X in [("признаки заёмщика",  X_symp),
                    ("числовые параметры", X_proc)]:
        acc = cross_val_score(make_tree(), X, y, cv=5)
        print(f"Точность (5-fold CV), {name}: {acc.mean():.2f} ± {acc.std():.2f}")

    model = make_tree().fit(X_proc, y)
    rules = extract_rules(model, list(X_proc.columns))
    print(f"\nИзвлечено правил: {len(rules)}")
    for r in rules:
        print(" ", fmt(r))
    with open("kb/rules_from_tree.json", "w", encoding="utf-8") as fh:
        json.dump(rules, fh, ensure_ascii=False, indent=2)