# DS-02 data dictionary — Part 1

The expected grain is **one row per date, store, and SKU**. The composite business key is `(date, store_id, sku_id)`.

| Column | Required | Meaning | Validation / notes |
|---|---:|---|---|
| `date` | Yes | Business date of the observation | Must parse as a date |
| `store_id` | Yes | Stable store identifier | Non-empty |
| `sku_id` | Yes | Stable product/SKU identifier | Non-empty |
| `units_sold` | Yes | Units sold during the date | Numeric and non-negative |
| `unit_price` | Yes | Listed selling price per unit, INR in bundled sample | Numeric and non-negative |
| `store_name` | No | Store display name | Optional enrichment |
| `city` | No | Store city | Optional enrichment |
| `state` | No | State/UT | Optional enrichment |
| `region` | No | Broad operating region | Optional enrichment |
| `product_name` | No | Product description | Optional enrichment |
| `category` | No | Product category | Optional enrichment |
| `unit_cost` | No | Unit acquisition cost | Numeric and non-negative |
| `discount_pct` | No | Discount as a decimal proportion | 0.10 means 10%; range 0–1 |
| `promo_flag` | No | Promotion indicator | 0/1 in bundled sample |
| `holiday_flag` | No | Holiday indicator | 0/1 in bundled sample |
| `opening_stock_units` | No | Stock at start of day | Numeric and non-negative |
| `replenishment_units` | No | Units received that day | Numeric and non-negative |
| `closing_stock_units` | No | Stock at end of day | Numeric and non-negative |
| `lead_time_days` | No | Replenishment lead time in days | Numeric and non-negative |
| `supplier_id` | No | Supplier identifier | Optional enrichment |
| `net_sales_value` | No | Units sold × price after discount | Present in bundled synthetic sample |

## Real-world ingestion guidance

Real source systems may use different field names, currency, date conventions, SKU identifiers, return handling, and stock definitions. Map source fields explicitly before validation. Do not assume `units_sold` always equals unconstrained customer demand: stockouts can censor observed sales. Preserve raw source files and document transformations.
