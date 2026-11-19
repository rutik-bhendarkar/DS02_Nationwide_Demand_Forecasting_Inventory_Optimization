from demand_inventory.data.ingestion import inspect_sales_csv, load_sales_csv


def test_bundled_sample_dataset_is_valid(project_root) -> None:
    path = project_root / "data/sample/nationwide_retail_daily_sales_synthetic.csv"
    frame, report = inspect_sales_csv(path)
    assert report.is_valid, report.errors
    assert len(frame) > 10_000
    assert frame["store_id"].nunique() == 18
    assert frame["sku_id"].nunique() == 12
    assert frame["region"].nunique() >= 5


def test_load_sales_csv_returns_dataframe(project_root) -> None:
    path = project_root / "data/sample/nationwide_retail_daily_sales_synthetic.csv"
    frame = load_sales_csv(path)
    assert not frame.empty
    assert {"date", "store_id", "sku_id", "units_sold"}.issubset(frame.columns)
