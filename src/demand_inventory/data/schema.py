"""Canonical input schema for daily store-SKU sales and inventory observations."""

REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "store_id",
    "sku_id",
    "units_sold",
    "unit_price",
)

BUSINESS_KEY: tuple[str, ...] = ("date", "store_id", "sku_id")

OPTIONAL_COLUMNS: tuple[str, ...] = (
    "store_name",
    "city",
    "state",
    "region",
    "product_name",
    "category",
    "unit_cost",
    "discount_pct",
    "promo_flag",
    "holiday_flag",
    "opening_stock_units",
    "replenishment_units",
    "closing_stock_units",
    "lead_time_days",
    "supplier_id",
)

NUMERIC_COLUMNS: tuple[str, ...] = (
    "units_sold",
    "unit_price",
    "unit_cost",
    "discount_pct",
    "opening_stock_units",
    "replenishment_units",
    "closing_stock_units",
    "lead_time_days",
)

NON_NEGATIVE_COLUMNS: tuple[str, ...] = (
    "units_sold",
    "unit_price",
    "unit_cost",
    "opening_stock_units",
    "replenishment_units",
    "closing_stock_units",
    "lead_time_days",
)
