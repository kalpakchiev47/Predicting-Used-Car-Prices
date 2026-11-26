"""
generate_data.py
----------------
Генерира синтетичен, но реалистичен набор от данни за употребявани автомобили.

Данните са създадени с фиксиран seed (42), така че всеки път се получава
един и същ файл data.csv. Цената се формира по пазарни зависимости:
възраст, пробег, марка/модел, гориво, скоростна кутия, мощност, състояние,
брой собственици и сервизна история, плюс случаен шум.

Употреба:
    python generate_data.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

SEED = 42
N_ROWS = 1000
REFERENCE_YEAR = 2026
OUTPUT = Path(__file__).with_name("data.csv")

# (make, model, new_price_eur, body_type, engines[(cc, hp)], fuel_weights)
# fuel_weights: вероятности за petrol / diesel / hybrid / electric
CARS = [
    ("VW", "Golf", 26000, "hatchback", [(1200, 90), (1400, 125), (1600, 115), (2000, 150)], (0.45, 0.45, 0.10, 0.00)),
    ("VW", "Passat", 34000, "sedan", [(1600, 120), (2000, 150), (2000, 190)], (0.30, 0.60, 0.10, 0.00)),
    ("VW", "Tiguan", 38000, "SUV", [(1400, 150), (2000, 150), (2000, 190)], (0.40, 0.50, 0.10, 0.00)),
    ("BMW", "3 Series", 45000, "sedan", [(1600, 136), (2000, 190), (3000, 265)], (0.35, 0.55, 0.10, 0.00)),
    ("BMW", "X3", 55000, "SUV", [(2000, 190), (3000, 265)], (0.30, 0.60, 0.10, 0.00)),
    ("Audi", "A4", 42000, "sedan", [(1400, 150), (2000, 150), (2000, 190), (3000, 272)], (0.35, 0.55, 0.10, 0.00)),
    ("Audi", "Q5", 52000, "SUV", [(2000, 190), (2000, 245), (3000, 286)], (0.30, 0.60, 0.10, 0.00)),
    ("Mercedes", "C-Class", 46000, "sedan", [(1600, 156), (2000, 194), (3000, 258)], (0.35, 0.55, 0.10, 0.00)),
    ("Mercedes", "GLC", 56000, "SUV", [(2000, 194), (3000, 258)], (0.30, 0.60, 0.10, 0.00)),
    ("Toyota", "Corolla", 27000, "hatchback", [(1200, 116), (1600, 132), (1800, 122)], (0.45, 0.15, 0.40, 0.00)),
    ("Toyota", "RAV4", 38000, "SUV", [(2000, 175), (2500, 178)], (0.35, 0.15, 0.50, 0.00)),
    ("Toyota", "Yaris", 20000, "hatchback", [(1000, 72), (1500, 116), (1500, 130)], (0.50, 0.10, 0.40, 0.00)),
    ("Renault", "Clio", 17000, "hatchback", [(900, 75), (1200, 90), (1500, 115)], (0.55, 0.35, 0.10, 0.00)),
    ("Renault", "Megane", 22000, "hatchback", [(1200, 100), (1500, 115), (1600, 130)], (0.50, 0.40, 0.10, 0.00)),
    ("Peugeot", "308", 23000, "hatchback", [(1200, 100), (1500, 130), (1600, 120)], (0.50, 0.40, 0.10, 0.00)),
    ("Peugeot", "3008", 32000, "SUV", [(1200, 130), (1500, 130), (2000, 180)], (0.45, 0.45, 0.10, 0.00)),
    ("Ford", "Focus", 23000, "hatchback", [(1000, 100), (1500, 120), (1500, 150), (2000, 150)], (0.50, 0.40, 0.10, 0.00)),
    ("Ford", "Kuga", 32000, "SUV", [(1500, 150), (2000, 150), (2500, 180)], (0.40, 0.50, 0.10, 0.00)),
    ("Opel", "Astra", 22000, "hatchback", [(1000, 105), (1400, 145), (1600, 110), (1600, 136)], (0.55, 0.35, 0.10, 0.00)),
    ("Opel", "Mokka", 26000, "SUV", [(1200, 130), (1400, 145), (1600, 136)], (0.50, 0.40, 0.10, 0.00)),
    ("Skoda", "Octavia", 26000, "sedan", [(1000, 115), (1400, 150), (1600, 115), (2000, 150)], (0.40, 0.50, 0.10, 0.00)),
    ("Skoda", "Superb", 34000, "sedan", [(1400, 150), (2000, 150), (2000, 190)], (0.30, 0.60, 0.10, 0.00)),
    ("Skoda", "Kodiaq", 38000, "SUV", [(1500, 150), (2000, 150), (2000, 190)], (0.35, 0.55, 0.10, 0.00)),
    ("Hyundai", "i30", 23000, "hatchback", [(1000, 100), (1400, 140), (1600, 136)], (0.50, 0.35, 0.15, 0.00)),
    ("Hyundai", "Tucson", 33000, "SUV", [(1600, 150), (1600, 180), (2000, 185)], (0.45, 0.35, 0.20, 0.00)),
    ("Kia", "Ceed", 23000, "hatchback", [(1000, 100), (1400, 140), (1600, 136)], (0.50, 0.35, 0.15, 0.00)),
    ("Kia", "Sportage", 33000, "SUV", [(1600, 150), (1600, 180), (2000, 185)], (0.45, 0.35, 0.20, 0.00)),
    ("Dacia", "Duster", 19000, "SUV", [(1000, 100), (1300, 130), (1500, 115)], (0.60, 0.30, 0.10, 0.00)),
    ("Dacia", "Sandero", 14000, "hatchback", [(900, 75), (1000, 100), (1500, 115)], (0.70, 0.25, 0.05, 0.00)),
    ("Nissan", "Qashqai", 30000, "SUV", [(1200, 115), (1600, 130), (1600, 163)], (0.45, 0.45, 0.10, 0.00)),
    ("Nissan", "Leaf", 32000, "hatchback", [(0, 110), (0, 150), (0, 217)], (0.00, 0.00, 0.00, 1.00)),
    ("Mazda", "CX-5", 34000, "SUV", [(2000, 165), (2200, 150), (2500, 194)], (0.55, 0.35, 0.10, 0.00)),
    ("Honda", "Civic", 28000, "hatchback", [(1000, 126), (1500, 182), (1600, 120)], (0.60, 0.20, 0.20, 0.00)),
    ("Seat", "Leon", 24000, "hatchback", [(1000, 115), (1400, 150), (1600, 115), (2000, 190)], (0.50, 0.40, 0.10, 0.00)),
    ("Fiat", "500", 16000, "hatchback", [(900, 85), (1200, 69), (1200, 95)], (0.70, 0.20, 0.10, 0.00)),
    ("Volvo", "XC60", 52000, "SUV", [(2000, 190), (2000, 250), (2000, 310)], (0.25, 0.50, 0.25, 0.00)),
    ("Tesla", "Model 3", 48000, "sedan", [(0, 283), (0, 351), (0, 498)], (0.00, 0.00, 0.00, 1.00)),
]

PREMIUM = {"BMW", "Audi", "Mercedes", "Volvo", "Tesla"}

# Ординален пазарен сегмент: 1 = бюджет, 2 = икономичен, 3 = среден, 4 = премиум.
# Признакът е известен от марката и помага на дърветата да различават класовете
# по-добре, отколкото 20 отделни one-hot колони за марка.
BRAND_SEGMENT = {
    "Dacia": 1, "Fiat": 1,
    "Renault": 2, "Peugeot": 2, "Opel": 2, "Ford": 2, "Skoda": 2,
    "Hyundai": 2, "Kia": 2, "Seat": 2,
    "VW": 3, "Toyota": 3, "Nissan": 3, "Mazda": 3, "Honda": 3,
    "Audi": 4, "BMW": 4, "Mercedes": 4, "Volvo": 4, "Tesla": 4,
}


def weighted_choice(rng: np.random.Generator, weights: tuple[float, ...]) -> int:
    """Връща индекс според зададени тегла."""
    p = np.asarray(weights, dtype=float)
    p = p / p.sum()
    return int(rng.choice(len(p), p=p))


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []

    for _ in range(N_ROWS):
        make, model, new_price, body, engines, fuel_weights = CARS[rng.integers(len(CARS))]

        # По-голямата част от автомобилите са на 3–12 години.
        year = int(np.clip(round(rng.normal(2017, 4.2)), 2010, 2024))
        age = REFERENCE_YEAR - year

        # Годишен пробег: премиум колите се карат малко по-малко.
        annual_km = float(np.clip(rng.normal(15000, 5000), 5000, 30000))
        if make in PREMIUM:
            annual_km *= 0.85
        mileage = int(round(age * annual_km * float(rng.lognormal(0, 0.16)) / 1000) * 1000)
        mileage = int(np.clip(mileage, 5000, 360000))

        fuel_idx = weighted_choice(rng, fuel_weights)
        fuel_type = ("petrol", "diesel", "hybrid", "electric")[fuel_idx]

        engine_cc, base_hp = engines[rng.integers(len(engines))]
        if fuel_type == "electric":
            engine_cc = 0
        power_hp = int(np.clip(round(base_hp + rng.normal(0, 8)), 60, 520))

        p_auto = 0.18 + (0.35 if make in PREMIUM else 0.0) + 0.035 * (year - 2010)
        transmission = "automatic" if rng.random() < min(p_auto, 0.90) else "manual"

        owners = int(np.clip(1 + rng.poisson(age / 6.0), 1, 5))

        # Състояние зависи от възрастта и пробега.
        if age <= 5 and mileage < 120000:
            condition = str(rng.choice(["excellent", "good", "fair"], p=[0.55, 0.40, 0.05]))
        elif age <= 10 and mileage < 210000:
            condition = str(rng.choice(["excellent", "good", "fair"], p=[0.30, 0.55, 0.15]))
        else:
            condition = str(rng.choice(["excellent", "good", "fair"], p=[0.12, 0.55, 0.33]))

        p_service = 0.92 - 0.02 * age + (0.04 if condition == "excellent" else 0.0)
        service_history = "yes" if rng.random() < float(np.clip(p_service, 0.40, 0.97)) else "no"

        # Пазарна цена: мултипликативен модел + шум.
        age_dep = 0.885 ** age
        mileage_dep = max(0.55, 1.0 - 0.0000012 * mileage)
        fuel_factor = {"petrol": 1.00, "diesel": 1.03, "hybrid": 1.12, "electric": 0.98}[fuel_type]
        transmission_factor = 1.07 if transmission == "automatic" else 1.00
        body_factor = {"hatchback": 1.00, "sedan": 1.03, "estate": 1.05, "SUV": 1.12, "coupe": 1.10}[body]
        condition_factor = {"excellent": 1.08, "good": 1.00, "fair": 0.87}[condition]
        owners_factor = {1: 1.04, 2: 1.00, 3: 0.96, 4: 0.92, 5: 0.88}[owners]
        service_factor = 1.04 if service_history == "yes" else 0.95
        power_factor = float(np.clip(1.0 + (power_hp - 120) / 1500.0, 0.88, 1.25))
        noise = float(rng.lognormal(0, 0.075))

        price = (
            new_price
            * age_dep
            * mileage_dep
            * fuel_factor
            * transmission_factor
            * body_factor
            * condition_factor
            * owners_factor
            * service_factor
            * power_factor
            * noise
        )
        price = int(max(1000, round(price / 50.0) * 50))

        rows.append(
            {
                "make": make,
                "model": model,
                "brand_segment": BRAND_SEGMENT[make],
                "year": year,
                "mileage_km": mileage,
                "engine_cc": engine_cc,
                "fuel_type": fuel_type,
                "transmission": transmission,
                "power_hp": power_hp,
                "body_type": body,
                "owners": owners,
                "condition": condition,
                "service_history": service_history,
                "price_eur": price,
            }
        )

    # Малко липсващи стойности, за да се демонстрира предобработка.
    missing_candidates = ["mileage_km", "engine_cc", "owners", "condition", "service_history"]
    for idx in rng.choice(N_ROWS, size=18, replace=False):
        column = missing_candidates[rng.integers(len(missing_candidates))]
        rows[int(idx)][column] = ""

    fieldnames = [
        "make", "model", "brand_segment", "year", "mileage_km", "engine_cc",
        "fuel_type", "transmission", "power_hp", "body_type", "owners",
        "condition", "service_history", "price_eur",
    ]
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Готово: {OUTPUT} ({len(rows)} реда)")


if __name__ == "__main__":
    main()
