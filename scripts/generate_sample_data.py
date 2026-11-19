"""Generate clearly labelled synthetic nationwide retail data for development tests.

This is NOT real Nexvia, retailer, customer, or market data and must never be
presented as a real-world benchmark or used to claim model performance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STORES = [
    ("ST001", "Delhi Central", "Delhi", "Delhi", "North"),
    ("ST002", "Chandigarh Sector 17", "Chandigarh", "Chandigarh", "North"),
    ("ST003", "Lucknow Hazratganj", "Lucknow", "Uttar Pradesh", "North"),
    ("ST004", "Mumbai Andheri", "Mumbai", "Maharashtra", "West"),
    ("ST005", "Pune Central", "Pune", "Maharashtra", "West"),
    ("ST006", "Ahmedabad West", "Ahmedabad", "Gujarat", "West"),
    ("ST007", "Bengaluru Indiranagar", "Bengaluru", "Karnataka", "South"),
    ("ST008", "Chennai T Nagar", "Chennai", "Tamil Nadu", "South"),
    ("ST009", "Hyderabad Central", "Hyderabad", "Telangana", "South"),
    ("ST010", "Kolkata Salt Lake", "Kolkata", "West Bengal", "East"),
    ("ST011", "Bhubaneswar Market", "Bhubaneswar", "Odisha", "East"),
    ("ST012", "Patna Central", "Patna", "Bihar", "East"),
    ("ST013", "Nagpur Dharampeth", "Nagpur", "Maharashtra", "Central"),
    ("ST014", "Indore Vijay Nagar", "Indore", "Madhya Pradesh", "Central"),
    ("ST015", "Bhopal New Market", "Bhopal", "Madhya Pradesh", "Central"),
    ("ST016", "Guwahati GS Road", "Guwahati", "Assam", "Northeast"),
    ("ST017", "Shillong Police Bazaar", "Shillong", "Meghalaya", "Northeast"),
    ("ST018", "Imphal City Centre", "Imphal", "Manipur", "Northeast"),
]

PRODUCTS = [
    ("SKU001", "Wheat Flour 5kg", "Staples", 260.0, 185.0, 18.0),
    ("SKU002", "Basmati Rice 5kg", "Staples", 520.0, 390.0, 12.0),
    ("SKU003", "Sunflower Oil 1L", "Grocery", 145.0, 108.0, 24.0),
    ("SKU004", "Toor Dal 1kg", "Staples", 175.0, 132.0, 15.0),
    ("SKU005", "Tea 500g", "Beverages", 245.0, 172.0, 10.0),
    ("SKU006", "Coffee 200g", "Beverages", 185.0, 128.0, 8.0),
    ("SKU007", "Bath Soap Pack", "Personal Care", 120.0, 78.0, 30.0),
    ("SKU008", "Laundry Detergent 2kg", "Household", 310.0, 225.0, 20.0),
    ("SKU009", "Shampoo 650ml", "Personal Care", 390.0, 278.0, 11.0),
    ("SKU010", "Biscuits Family Pack", "Snacks", 95.0, 61.0, 36.0),
    ("SKU011", "Packaged Juice 1L", "Beverages", 110.0, 72.0, 16.0),
    ("SKU012", "Toothpaste 150g", "Personal Care", 105.0, 69.0, 22.0),
]


def generate_dataset(
    *, start_date: str = "2025-01-01", days: int = 180, seed: int = 42
) -> pd.DataFrame:
    """Create deterministic daily observations for 18 stores and 12 SKUs."""
    if days < 30:
        raise ValueError("days must be at least 30 to provide a useful development sample")
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=days, freq="D")
    holiday_dates = {"01-26", "03-14", "04-14", "05-01"}
    rows: list[dict[str, object]] = []

    store_factors = {store[0]: float(rng.uniform(0.78, 1.28)) for store in STORES}
    sku_factors = {product[0]: float(rng.uniform(0.8, 1.25)) for product in PRODUCTS}

    for store_id, store_name, city, state, region in STORES:
        store_factor = store_factors[store_id]
        for sku_id, product_name, category, unit_price, unit_cost, base_demand in PRODUCTS:
            stock = int(rng.integers(130, 210))
            sku_factor = sku_factors[sku_id]
            lead_time = int(rng.integers(2, 9))
            supplier_id = f"SUP{int(rng.integers(1, 7)):03d}"

            for day_index, date in enumerate(dates):
                promo_flag = int(rng.random() < 0.085)
                holiday_flag = int(date.strftime("%m-%d") in holiday_dates)
                weekend_factor = 1.16 if date.dayofweek >= 5 else 1.0
                promo_factor = 1.22 if promo_flag else 1.0
                holiday_factor = 1.18 if holiday_flag else 1.0
                trend_factor = 1.0 + 0.10 * (day_index / max(days - 1, 1))
                noise = max(0.35, float(rng.normal(1.0, 0.16)))
                expected_demand = (
                    base_demand * store_factor * sku_factor * weekend_factor
                    * promo_factor * holiday_factor * trend_factor * noise
                )
                requested_units = max(0, int(round(expected_demand)))

                # Simple synthetic replenishment process to create inventory fields.
                opening_stock = stock
                replenishment_units = 0
                if opening_stock < 110:
                    replenishment_units = int(rng.integers(150, 221))
                available_units = opening_stock + replenishment_units
                units_sold = min(requested_units, available_units)
                closing_stock = available_units - units_sold
                stock = closing_stock

                discount_pct = float(rng.choice([0.0, 0.0, 0.0, 0.05, 0.10, 0.15]))
                effective_price = round(unit_price * (1 - discount_pct), 2)
                rows.append(
                    {
                        "date": date.date().isoformat(),
                        "store_id": store_id,
                        "store_name": store_name,
                        "city": city,
                        "state": state,
                        "region": region,
                        "sku_id": sku_id,
                        "product_name": product_name,
                        "category": category,
                        "units_sold": units_sold,
                        "unit_price": unit_price,
                        "unit_cost": unit_cost,
                        "discount_pct": discount_pct,
                        "promo_flag": promo_flag,
                        "holiday_flag": holiday_flag,
                        "opening_stock_units": opening_stock,
                        "replenishment_units": replenishment_units,
                        "closing_stock_units": closing_stock,
                        "lead_time_days": lead_time,
                        "supplier_id": supplier_id,
                        "net_sales_value": round(units_sold * effective_price, 2),
                    }
                )

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="data/sample/nationwide_retail_daily_sales_synthetic.csv",
        help="Output CSV path (default: bundled sample location)",
    )
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_dataset(start_date=args.start_date, days=args.days, seed=args.seed)
    frame.to_csv(output_path, index=False)
    print("Synthetic development dataset generated successfully.")
    print("IMPORTANT: This is synthetic test data, not real business data.")
    print(f"Output: {output_path.resolve()}")
    print(f"Rows: {len(frame):,} | Stores: {frame['store_id'].nunique()} | SKUs: {frame['sku_id'].nunique()}")
    print(f"Date range: {frame['date'].min()} to {frame['date'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
