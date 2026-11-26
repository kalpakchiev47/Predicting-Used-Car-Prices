"""
main.py
-------
Прогнозиране на цени на употребявани автомобили по таблични признаци
с Decision Tree и Random Forest.

Проект по „Програмиране на изкуствен интелект“
Ученик: Георги Младенов Калпакчиев, Вариант А

Употреба:
    python main.py                 # обучение, оценка, графики и запис на модела
    python main.py --quick         # без GridSearch (по-бързо)
    python main.py --predict       # интерактивна прогноза за нов автомобил
    python main.py --sample        # прогноза за няколко примерни автомобила
    python main.py --no-plots      # без графики
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor, plot_tree

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR / "data.csv"
DEFAULT_MODEL = BASE_DIR / "model" / "used_car_price_model.joblib"
DEFAULT_PLOTS = BASE_DIR / "plots"
DEFAULT_METRICS = BASE_DIR / "metrics.json"

REFERENCE_YEAR = 2026
RANDOM_STATE = 42
TARGET = "price_eur"

NUMERIC_FEATURES = ["brand_segment", "year", "mileage_km", "engine_cc", "power_hp", "owners"]
CATEGORICAL_FEATURES = [
    "make",
    "model",
    "fuel_type",
    "transmission",
    "body_type",
    "condition",
    "service_history",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CATEGORY_VALUES = {
    "make": ["VW", "BMW", "Audi", "Mercedes", "Toyota", "Renault", "Peugeot", "Ford",
             "Opel", "Skoda", "Hyundai", "Kia", "Dacia", "Nissan", "Mazda", "Honda",
             "Seat", "Fiat", "Volvo", "Tesla"],
    "fuel_type": ["petrol", "diesel", "hybrid", "electric"],
    "transmission": ["manual", "automatic"],
    "body_type": ["hatchback", "sedan", "estate", "SUV", "coupe"],
    "condition": ["excellent", "good", "fair"],
    "service_history": ["yes", "no"],
}

SAMPLE_CARS = [
    {
        "make": "VW", "model": "Golf", "brand_segment": 3, "year": 2018, "mileage_km": 120000,
        "engine_cc": 1600, "fuel_type": "diesel", "transmission": "manual",
        "power_hp": 115, "body_type": "hatchback", "owners": 2,
        "condition": "good", "service_history": "yes",
    },
    {
        "make": "BMW", "model": "3 Series", "brand_segment": 4, "year": 2020, "mileage_km": 80000,
        "engine_cc": 2000, "fuel_type": "diesel", "transmission": "automatic",
        "power_hp": 190, "body_type": "sedan", "owners": 1,
        "condition": "excellent", "service_history": "yes",
    },
    {
        "make": "Dacia", "model": "Sandero", "brand_segment": 1, "year": 2012, "mileage_km": 210000,
        "engine_cc": 1000, "fuel_type": "petrol", "transmission": "manual",
        "power_hp": 75, "body_type": "hatchback", "owners": 3,
        "condition": "fair", "service_history": "no",
    },
    {
        "make": "Tesla", "model": "Model 3", "brand_segment": 4, "year": 2021, "mileage_km": 60000,
        "engine_cc": 0, "fuel_type": "electric", "transmission": "automatic",
        "power_hp": 283, "body_type": "sedan", "owners": 1,
        "condition": "excellent", "service_history": "yes",
    },
]


# ---------------------------------------------------------------------------
# Помощни функции
# ---------------------------------------------------------------------------

def configure_console() -> None:
    """Настройва UTF-8 изход, за да не гърми на Windows с кирилица."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def eur(value: float) -> str:
    """Форматира число като цена в евро с интервал за хилядите."""
    return f"{value:,.0f} €".replace(",", " ")


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Не е намерен файлът с данни: {path}\n"
            "Стартирайте първо: python generate_data.py"
        )
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"В data.csv липсват колони: {missing}")
    return df


def build_preprocessor() -> ColumnTransformer:
    # Дърветата не се нуждаят от мащабиране; така праговете в дървото
    # остават в оригинални единици (години, километри, куб. см) и са по-разбираеми.
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def make_pipeline(model: Any) -> TransformedTargetRegressor:
    # Цената се формира мултипликативно (възраст × пробег × гориво × …), затова
    # моделът се учи върху log(цена) и прогнозите се връщат обратно в евро.
    # Така относителните грешки са по-равномерни за евтини и скъпи автомобили.
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )
    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)


def fitted_pipeline(estimator: TransformedTargetRegressor) -> Pipeline:
    """Връща обучения вътрешен Pipeline (предобработка + модел)."""
    return estimator.regressor_


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_pct": float(mean_absolute_percentage_error(y_true, y_pred) * 100.0),
    }


def original_feature(encoded_name: str) -> str:
    """Превръща 'cat__fuel_type_diesel' обратно в 'fuel_type'."""
    base = encoded_name.split("__", 1)[-1]
    for feature in sorted(FEATURES, key=len, reverse=True):
        if base == feature or base.startswith(feature + "_"):
            return feature
    return base


def feature_importance(estimator: TransformedTargetRegressor) -> dict[str, list[dict[str, Any]]]:
    pipeline = fitted_pipeline(estimator)
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    names = list(preprocessor.get_feature_names_out())
    values = np.asarray(model.feature_importances_, dtype=float)

    by_feature: dict[str, float] = {}
    for name, value in zip(names, values):
        key = original_feature(name)
        by_feature[key] = by_feature.get(key, 0.0) + float(value)

    total = sum(by_feature.values()) or 1.0
    aggregated = [
        {"feature": k, "importance": v / total}
        for k, v in sorted(by_feature.items(), key=lambda item: item[1], reverse=True)
    ]
    individual = [
        {"feature": n, "importance": float(v)}
        for n, v in sorted(zip(names, values), key=lambda item: item[1], reverse=True)[:15]
    ]
    return {"aggregated": aggregated, "individual": individual}


def train_and_evaluate(
    data_path: Path = DEFAULT_DATA,
    model_path: Path = DEFAULT_MODEL,
    plots_dir: Path = DEFAULT_PLOTS,
    metrics_path: Path = DEFAULT_METRICS,
    quick: bool = False,
    make_plots: bool = True,
) -> dict[str, Any]:
    configure_console()
    df = load_data(data_path)
    X = df[FEATURES].copy()
    y = df[TARGET].astype(float).to_numpy()

    print("=" * 70)
    print("ПРОГНОЗИРАНЕ НА ЦЕНИ НА УПОТРЕБЯВАНИ АВТОМОБИЛИ")
    print("Георги Младенов Калпакчиев • Вариант А • Decision Tree / Random Forest")
    print("=" * 70)
    print(f"Данни: {len(df)} реда, {len(FEATURES)} признака")
    print(f"Цена: min {eur(y.min())}, медиана {eur(np.median(y))}, max {eur(y.max())}")
    print(f"Липсващи стойности: {int(df[FEATURES].isna().sum().sum())} клетки")
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    print(f"Обучение: {len(X_train)} реда | Тест: {len(X_test)} реда")
    print()

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results: dict[str, Any] = {
        "student": "Георги Младенов Калпакчиев",
        "variant": "А",
        "dataset": {
            "rows": int(len(df)),
            "features": len(FEATURES),
            "missing_cells": int(df[FEATURES].isna().sum().sum()),
            "price_min": float(y.min()),
            "price_median": float(np.median(y)),
            "price_max": float(y.max()),
        },
        "split": {"train": int(len(X_train)), "test": int(len(X_test))},
    }

    # --- Decision Tree -----------------------------------------------------
    print("-" * 70)
    print("1) Decision Tree Regressor")
    print("-" * 70)
    if quick:
        dt = DecisionTreeRegressor(max_depth=8, min_samples_leaf=4, random_state=RANDOM_STATE)
        dt_pipeline = make_pipeline(dt).fit(X_train, y_train)
        dt_best_params = {"max_depth": 8, "min_samples_leaf": 4}
        dt_cv_mae = None
    else:
        dt_grid = {
            "regressor__model__max_depth": [4, 6, 8, 10, 12, None],
            "regressor__model__min_samples_leaf": [1, 2, 4, 8],
            "regressor__model__min_samples_split": [2, 5, 10],
        }
        dt_search = GridSearchCV(
            make_pipeline(DecisionTreeRegressor(random_state=RANDOM_STATE)),
            param_grid=dt_grid,
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=-1,
        )
        dt_search.fit(X_train, y_train)
        dt_pipeline = dt_search.best_estimator_
        dt_best_params = {k.replace("regressor__model__", ""): v for k, v in dt_search.best_params_.items()}
        dt_cv_mae = float(-dt_search.best_score_)
        print(f"Най-добри параметри: {dt_best_params}")
        print(f"CV MAE (обучение): {eur(dt_cv_mae)}")
    dt_pred = dt_pipeline.predict(X_test)
    dt_metrics = metrics(y_test, dt_pred)
    print(f"Тест: MAE={eur(dt_metrics['MAE'])}, RMSE={eur(dt_metrics['RMSE'])}, "
          f"R²={dt_metrics['R2']:.3f}, MAPE={dt_metrics['MAPE_pct']:.1f}%")
    print()

    # --- Random Forest -----------------------------------------------------
    print("-" * 70)
    print("2) Random Forest Regressor")
    print("-" * 70)
    if quick:
        rf = RandomForestRegressor(
            n_estimators=400, max_depth=None, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        rf_pipeline = make_pipeline(rf).fit(X_train, y_train)
        rf_best_params = {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 2}
        rf_cv_mae = None
    else:
        rf_grid = {
            "regressor__model__max_depth": [None, 12],
            "regressor__model__min_samples_leaf": [1, 2, 4],
            "regressor__model__max_features": [1.0, 0.7, 0.5],
        }
        rf_search = GridSearchCV(
            make_pipeline(
                RandomForestRegressor(
                    n_estimators=300, random_state=RANDOM_STATE, n_jobs=1
                )
            ),
            param_grid=rf_grid,
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=-1,
        )
        rf_search.fit(X_train, y_train)
        rf_pipeline = rf_search.best_estimator_
        rf_best_params = {k.replace("regressor__model__", ""): v for k, v in rf_search.best_params_.items()}
        rf_cv_mae = float(-rf_search.best_score_)
        print(f"Най-добри параметри: {rf_best_params}")
        print(f"CV MAE (обучение): {eur(rf_cv_mae)}")

    rf_pred = rf_pipeline.predict(X_test)
    rf_metrics = metrics(y_test, rf_pred)
    print(f"Тест: MAE={eur(rf_metrics['MAE'])}, RMSE={eur(rf_metrics['RMSE'])}, "
          f"R²={rf_metrics['R2']:.3f}, MAPE={rf_metrics['MAPE_pct']:.1f}%")
    print()

    # --- Baseline ----------------------------------------------------------
    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    base_pred = baseline.predict(X_test)
    base_metrics = metrics(y_test, base_pred)
    print("-" * 70)
    print("3) Baseline (средна цена)")
    print("-" * 70)
    print(f"Тест: MAE={eur(base_metrics['MAE'])}, RMSE={eur(base_metrics['RMSE'])}, "
          f"R²={base_metrics['R2']:.3f}")
    print()

    # --- Cross-validation на избрания Random Forest ------------------------
    print("-" * 70)
    print("4) 5-fold крос-валидация на Random Forest (върху train)")
    print("-" * 70)
    cv_result = cross_validate(
        rf_pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        n_jobs=-1,
    )
    cv_summary = {
        "MAE_mean": float(-cv_result["test_mae"].mean()),
        "MAE_std": float(cv_result["test_mae"].std()),
        "RMSE_mean": float(-cv_result["test_rmse"].mean()),
        "R2_mean": float(cv_result["test_r2"].mean()),
        "R2_std": float(cv_result["test_r2"].std()),
    }
    print(f"MAE  = {eur(cv_summary['MAE_mean'])} ± {eur(cv_summary['MAE_std'])}")
    print(f"RMSE = {eur(cv_summary['RMSE_mean'])}")
    print(f"R²   = {cv_summary['R2_mean']:.3f} ± {cv_summary['R2_std']:.3f}")
    print()

    # --- Анализ на остатъците ---------------------------------------------
    residuals = y_test - rf_pred
    worst_idx = np.argsort(np.abs(residuals))[::-1][:5]
    worst = []
    for idx in worst_idx:
        row = X_test.iloc[int(idx)]
        worst.append(
            {
                "make": str(row["make"]),
                "model": str(row["model"]),
                "year": int(row["year"]),
                "mileage_km": int(row["mileage_km"]) if pd.notna(row["mileage_km"]) else None,
                "actual": float(y_test[int(idx)]),
                "predicted": float(rf_pred[int(idx)]),
                "error": float(residuals[int(idx)]),
            }
        )

    print("-" * 70)
    print("5) Анализ на грешките на Random Forest (тест)")
    print("-" * 70)
    print(f"Средна грешка (bias): {eur(float(residuals.mean()))}")
    print(f"Стандартно отклонение: {eur(float(residuals.std()))}")
    print(f"90% от грешките са в интервал: "
          f"[{eur(float(np.percentile(residuals, 5)))}, {eur(float(np.percentile(residuals, 95)))}]")
    print("Най-големи абсолютни грешки:")
    for item in worst:
        print(f"  {item['make']} {item['model']} ({item['year']}, "
              f"{item['mileage_km']} км): реална {eur(item['actual'])}, "
              f"прогноза {eur(item['predicted'])}, грешка {eur(item['error'])}")
    print()

    # --- Важност на признаците --------------------------------------------
    importance = feature_importance(rf_pipeline)
    print("-" * 70)
    print("6) Важност на признаците (Random Forest)")
    print("-" * 70)
    for item in importance["aggregated"]:
        bar = "█" * max(1, int(round(item["importance"] * 40)))
        print(f"  {item['feature']:<18} {item['importance']*100:5.1f}%  {bar}")
    print()

    # --- Запис на резултатите ---------------------------------------------
    results.update(
        {
            "decision_tree": {
                "best_params": dt_best_params,
                "cv_mae": dt_cv_mae,
                "test": dt_metrics,
            },
            "random_forest": {
                "best_params": rf_best_params,
                "cv_mae": rf_cv_mae,
                "test": rf_metrics,
            },
            "baseline": {"test": base_metrics},
            "cross_validation_rf": cv_summary,
            "residuals": {
                "mean": float(residuals.mean()),
                "std": float(residuals.std()),
                "p05": float(np.percentile(residuals, 5)),
                "p95": float(np.percentile(residuals, 95)),
            },
            "worst_errors": worst,
            "feature_importance": importance["aggregated"],
            "feature_importance_individual": importance["individual"],
        }
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": rf_pipeline,
            "feature_columns": FEATURES,
            "reference_year": REFERENCE_YEAR,
            "metrics": results,
        },
        model_path,
        compress=3,
    )
    print(f"Моделът е записан в: {model_path}")

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Метриките са записани в: {metrics_path}")

    if make_plots:
        make_all_plots(
            rf_pipeline,
            X_train,
            y_train,
            X_test,
            y_test,
            rf_pred,
            importance,
            {
                "Decision Tree": dt_metrics,
                "Random Forest": rf_metrics,
                "Baseline": base_metrics,
            },
            plots_dir,
        )

    print()
    print("=" * 70)
    print("ГОТОВО. Моделът е обучен и оценен.")
    print("За интерактивна прогноза: python main.py --predict")
    print("=" * 70)
    return results


# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------

def make_all_plots(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    importance: dict[str, Any],
    all_metrics: dict[str, dict[str, float]],
    plots_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Графиките са пропуснати (matplotlib не е наличен): {exc}")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

    # 1. Реална срещу прогнозна цена
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, alpha=0.65, edgecolor="white", linewidth=0.4)
    low = min(float(y_test.min()), float(y_pred.min()))
    high = max(float(y_test.max()), float(y_pred.max()))
    ax.plot([low, high], [low, high], "r--", label="Идеална прогноза")
    ax.set_xlabel("Реална цена (€)")
    ax.set_ylabel("Прогнозна цена (€)")
    ax.set_title("Random Forest: реална срещу прогнозна цена")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    # 2. Остатъци
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(residuals, bins=25, color="#4C72B0", edgecolor="white")
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Грешка (реална − прогнозна), €")
    ax.set_ylabel("Брой автомобили")
    ax.set_title("Разпределение на остатъците (тест)")
    fig.tight_layout()
    fig.savefig(plots_dir / "residuals.png", dpi=150)
    plt.close(fig)

    # 3. Важност на признаците
    items = importance["aggregated"]
    labels = [item["feature"] for item in items][::-1]
    values = [item["importance"] * 100 for item in items][::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, values, color="#55A868")
    ax.set_xlabel("Относителна важност (%)")
    ax.set_title("Кои признаци влияят най-много на цената")
    fig.tight_layout()
    fig.savefig(plots_dir / "feature_importance.png", dpi=150)
    plt.close(fig)

    # 4. Сравнение на моделите по MAE, RMSE и R²
    model_names = list(all_metrics.keys())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, metric_key, title in zip(
        axes,
        ["MAE", "RMSE", "R2"],
        ["MAE (по-малко е по-добре)", "RMSE (по-малко е по-добре)", "R² (повече е по-добре)"],
    ):
        values = [all_metrics[name][metric_key] for name in model_names]
        colors = ["#4C72B0", "#55A868", "#C44E52"]
        bars = ax.bar(model_names, values, color=colors[: len(model_names)])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        for bar, value in zip(bars, values):
            label = f"{value:.2f}" if metric_key == "R2" else eur(value)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle("Сравнение на моделите върху тестовите данни")
    fig.tight_layout()
    fig.savefig(plots_dir / "model_comparison.png", dpi=150)
    plt.close(fig)

    # 5. Малко дърво за обяснение (max_depth=3)
    try:
        pre = fitted_pipeline(pipeline).named_steps["preprocessor"]
        Xt = pre.transform(X_train)
        names = list(pre.get_feature_names_out())
        short_names = [n.replace("num__", "").replace("cat__", "") for n in names]
        shallow = DecisionTreeRegressor(max_depth=3, random_state=RANDOM_STATE)
        shallow.fit(Xt, y_train)
        fig, ax = plt.subplots(figsize=(16, 9))
        plot_tree(
            shallow,
            feature_names=short_names,
            filled=True,
            rounded=True,
            fontsize=7,
            ax=ax,
        )
        ax.set_title("Примерно Decision Tree с дълбочина 3 (за обяснение)")
        fig.tight_layout()
        fig.savefig(plots_dir / "decision_tree_depth3.png", dpi=150)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        print(f"Дървото за визуализация е пропуснато: {exc}")

    print(f"Графиките са записани в: {plots_dir}")


# ---------------------------------------------------------------------------
# Прогнози
# ---------------------------------------------------------------------------

def load_artifact(model_path: Path = DEFAULT_MODEL) -> dict[str, Any] | None:
    if not model_path.exists():
        return None
    try:
        return joblib.load(model_path)
    except Exception as exc:
        print(f"Моделът не може да бъде зареден ({exc}); ще бъде обучен наново.")
        return None


def predict_one(artifact: dict[str, Any], car: dict[str, Any]) -> tuple[float, float, float]:
    estimator: TransformedTargetRegressor = artifact["pipeline"]
    row = pd.DataFrame([{feature: car.get(feature) for feature in FEATURES}])
    pred = float(estimator.predict(row)[0])

    inner = fitted_pipeline(estimator)
    model = inner.named_steps["model"]
    if hasattr(model, "estimators_"):
        pre = inner.named_steps["preprocessor"]
        transformed = pre.transform(row)
        # Дърветата предсказват log(цена) -> връщаме в евро с expm1.
        tree_preds = np.expm1(
            np.array([tree.predict(transformed)[0] for tree in model.estimators_])
        )
        low, high = np.percentile(tree_preds, [10, 90])
        return pred, float(low), float(high)
    return pred, pred, pred


def print_prediction(car: dict[str, Any], pred: float, low: float, high: float) -> None:
    print()
    print(f"  {car['make']} {car['model']}, {car['year']} г., "
          f"{car['mileage_km']:,} км".replace(",", " "))
    print(f"  Гориво: {car['fuel_type']}, кутия: {car['transmission']}, "
          f"мощност: {car['power_hp']} к.с.")
    print(f"  Състояние: {car['condition']}, собственици: {car['owners']}, "
          f"сервизна история: {car['service_history']}")
    print(f"  → Прогнозна цена: {eur(pred)}")
    if high > low:
        print(f"    Ориентировъчен интервал (10–90% от дърветата): "
              f"{eur(low)} – {eur(high)}")


def interactive_predict(artifact: dict[str, Any]) -> None:
    print("=" * 70)
    print("ИНТЕРАКТИВНА ПРОГНОЗА")
    print("Натиснете Enter, за да използвате стойността в скобите.")
    print("=" * 70)

    defaults = SAMPLE_CARS[0]
    car: dict[str, Any] = {}

    def ask_text(label: str, key: str) -> str:
        value = input(f"{label} [{defaults[key]}]: ").strip()
        return value if value else str(defaults[key])

    def ask_int(label: str, key: str, minimum: int, maximum: int) -> int:
        while True:
            raw = input(f"{label} [{defaults[key]}]: ").strip()
            if not raw:
                return int(defaults[key])
            try:
                value = int(raw)
            except ValueError:
                print("  Моля, въведете цяло число.")
                continue
            if minimum <= value <= maximum:
                return value
            print(f"  Стойността трябва да е между {minimum} и {maximum}.")

    def ask_choice(label: str, key: str, choices: list[str]) -> str:
        while True:
            raw = input(f"{label} {choices} [{defaults[key]}]: ").strip().lower()
            if not raw:
                return str(defaults[key])
            if raw in [c.lower() for c in choices]:
                return next(c for c in choices if c.lower() == raw)
            print(f"  Изберете една от: {', '.join(choices)}")

    car["make"] = ask_text("Марка", "make")
    car["model"] = ask_text("Модел", "model")
    car["brand_segment"] = ask_int("Пазарен сегмент (1=бюджет … 4=премиум)", "brand_segment", 1, 4)
    car["year"] = ask_int("Година", "year", 1990, REFERENCE_YEAR)
    car["mileage_km"] = ask_int("Пробег (км)", "mileage_km", 0, 600000)
    car["engine_cc"] = ask_int("Двигател (куб. см, 0 за електрически)", "engine_cc", 0, 6000)
    car["fuel_type"] = ask_choice("Гориво", "fuel_type", CATEGORY_VALUES["fuel_type"])
    car["transmission"] = ask_choice("Скоростна кутия", "transmission", CATEGORY_VALUES["transmission"])
    car["power_hp"] = ask_int("Мощност (к.с.)", "power_hp", 30, 1000)
    car["body_type"] = ask_choice("Тип", "body_type", CATEGORY_VALUES["body_type"])
    car["owners"] = ask_int("Брой собственици", "owners", 1, 10)
    car["condition"] = ask_choice("Състояние", "condition", CATEGORY_VALUES["condition"])
    car["service_history"] = ask_choice(
        "Сервизна история", "service_history", CATEGORY_VALUES["service_history"]
    )

    pred, low, high = predict_one(artifact, car)
    print_prediction(car, pred, low, high)


def sample_predict(artifact: dict[str, Any]) -> None:
    print("=" * 70)
    print("ПРИМЕРНИ ПРОГНОЗИ")
    print("=" * 70)
    for car in SAMPLE_CARS:
        pred, low, high = predict_one(artifact, car)
        print_prediction(car, pred, low, high)
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Прогнозиране на цени на употребявани автомобили с Decision Tree / Random Forest"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="път до data.csv")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="път до .joblib модела")
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS, help="папка за графиките")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS, help="път до metrics.json")
    parser.add_argument("--quick", action="store_true", help="без GridSearch (по-бързо)")
    parser.add_argument("--no-plots", action="store_true", help="без графики")
    parser.add_argument("--predict", action="store_true", help="интерактивна прогноза")
    parser.add_argument("--sample", action="store_true", help="примерни прогнози")
    return parser.parse_args()


def main() -> None:
    configure_console()
    args = parse_args()

    if args.predict or args.sample:
        artifact = load_artifact(args.model)
        if artifact is None:
            print("Няма записан модел. Обучение...")
            train_and_evaluate(
                data_path=args.data,
                model_path=args.model,
                plots_dir=args.plots_dir,
                metrics_path=args.metrics,
                quick=args.quick,
                make_plots=not args.no_plots,
            )
            artifact = load_artifact(args.model)
        if artifact is None:
            raise RuntimeError("Моделът не може да бъде зареден.")
        if args.predict:
            interactive_predict(artifact)
        else:
            sample_predict(artifact)
        return

    train_and_evaluate(
        data_path=args.data,
        model_path=args.model,
        plots_dir=args.plots_dir,
        metrics_path=args.metrics,
        quick=args.quick,
        make_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
