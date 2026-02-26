import numpy as np
import pandas as pd

#%%
def one_hot_encode(df, drop_first=False, dummy_na=False, prefix_sep="_"):
    """
    One-hot encode categorical columns in a DataFrame.
    :param df:
    :param drop_first:
    :param dummy_na:
    :param prefix_sep:
    :return: encoded DataFrame and a mapping of original categories to new column names
    """
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    out = pd.get_dummies(
        df,
        columns=cat_cols,
        drop_first=drop_first,
        dummy_na=dummy_na,
        prefix_sep=prefix_sep
    )

    cat_map = {}
    for c in cat_cols:
        # categories (including NaN as a literal label if dummy_na=True)
        vals = df[c].astype("object")
        if dummy_na:
            vals = vals.where(vals.notna(), other="nan")

        categories = pd.Index(vals.unique()).astype(str).tolist()

        # build expected dummy col names that actually exist
        mapping = {}
        for v in categories:
            col_name = f"{c}{prefix_sep}{v}"
            if col_name in out.columns:
                mapping[v] = col_name

        cat_map[c] = mapping

    return out, cat_map