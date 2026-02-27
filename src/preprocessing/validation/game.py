import pandas as pd


def qc_missing_halves(season_df: pd.DataFrame):
    key = "game_id" if "game_id" in season_df.columns else "game_pk"

    half = season_df["inning_topbot"].astype("string").str.lower()

    has_top = (half == "top").groupby(season_df[key]).any()
    has_bot = (half == "bot").groupby(season_df[key]).any()

    qc = (
        pd.DataFrame({"has_top": has_top, "has_bot": has_bot})
          .reset_index()
          .rename(columns={key: "game_id"})
    )
    qc["missing_top"] = ~qc["has_top"]
    qc["missing_bot"] = ~qc["has_bot"]
    qc["all_top_or_all_bot"] = qc["missing_top"] | qc["missing_bot"]

    # Helpful counts
    summary = {
        "total_games": len(qc),
        "missing_top": int(qc["missing_top"].sum()),
        "missing_bot": int(qc["missing_bot"].sum()),
        "all_top_or_all_bot": int(qc["all_top_or_all_bot"].sum()),
    }
    return qc, summary



