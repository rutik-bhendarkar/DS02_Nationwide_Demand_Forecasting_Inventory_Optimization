# Part 2 — Exploratory Data Analysis Findings

> **Data source:** Bundled synthetic development dataset; not real retailer data.
> This report is descriptive. It does not evaluate forecasting accuracy or prove causal effects.

## Dataset overview

- Rows: **38,880**
- Columns: **21**
- Date coverage: **2025-01-01 to 2025-06-29** (180 calendar days)
- Stores: **18**
- SKUs: **12**
- Regions: **6**

## Demand and sales

- Total observed units sold: **882,002**
- Average units per row: **22.69**
- Median units per row: **20.00**
- Daily demand average: **4,900.01 units**
- Sales value basis: **provided net_sales_value column**
- Total analysis sales value: **162,440,552.25**

## Data quality and coverage

- Missing cells: **0**
- Duplicate date-store-SKU keys: **0**
- Store-SKU series analyzed for coverage: **216**
- Series with at least one missing calendar day: **0**

## Outlier review

- Flagged observations: **590** (1.5175% of rows)
- Method: 1.5 × IQR fences calculated separately for each store_id + sku_id series.
- Interpretation: Statistical flags for review, not proof of data errors. Promotions, holidays, stock availability and local events may explain unusual demand.

## Interpretation cautions

- This dataset is a development sample; results are not evidence about actual nationwide retail demand.
- Observed sales may be lower than true demand when inventory is unavailable (censored demand).
- Calendar patterns are descriptive only. Promotion and holiday relationships are not causal estimates.
- Outlier flags should be reviewed with promotion, holiday, stock and source-system context before any removal.
- Forecast model training and time-based model evaluation are reserved for Parts 4 and 5.

## Generated outputs

See the `outputs/eda/` folder for CSV analysis tables and the interactive HTML dashboard, and `reports/part2_eda_summary.json` for the machine-readable summary.
