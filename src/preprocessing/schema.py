import pandas as pd


ROLL_COMPONENTS = ("IP", "H", "BB", "HBP", "K", "HR")
WINDOWS = ("3G", "7G")
ROLES = ("starter", "bullpen")

def drop_rolled_component_cols(
    df: pd.DataFrame,
    windows: tuple[str, ...] = WINDOWS,
    roles: tuple[str, ...] = ROLES,
    components: tuple[str, ...] = ROLL_COMPONENTS,
) -> pd.DataFrame:
    """
    Drop rolled count/component columns like roll_3D_starter_IP, roll_7D_bullpen_HR, etc.
    Keeps rate columns like roll_3D_starter_WHIP / K9 / HR9 / FIP.
    Returns a NEW df.
    """
    to_drop = []
    for w in windows:
        for role in roles:
            for comp in components:
                col = f"roll_{w}_{role}_{comp}"
                if col in df.columns:
                    to_drop.append(col)

    return df.drop(columns=to_drop, errors="ignore")


