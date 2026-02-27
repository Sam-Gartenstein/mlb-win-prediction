import pandas as pd


def missing_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "total_na_cells": int(df.isna().sum().sum()),
        "rows_with_any_na": int(df.isna().any(axis=1).sum()),
        "cols_with_any_na": int((df.isna().sum() > 0).sum()),
    }


def assert_no_missing(df: pd.DataFrame, name: str) -> None:
    total_na = int(df.isna().sum().sum())
    print(f"{name}: total missing cells = {total_na}")
    assert total_na == 0, f"{name}: found {total_na} missing cells"