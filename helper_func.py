import numpy as np
import pandas as pd

#%%
def one_hot_encode(df,
                   drop_first: bool = False,
                   dummy_na: bool = False,
                   prefix_sep: str = "_",):
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

def convert_to_numeric(df:pd.DataFrame,
                       drop_nan=True,)-> pd.DataFrame:
    """Convert all object columns in the DataFrame to numeric, coercing errors to NaN."""
    not_accepted = df.select_dtypes(include=['object']).columns
    for col in not_accepted:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df.dropna() if drop_nan else df

def get_descriptors(data_set:pd.DataFrame,
                    mol_col:list,
                    save_dir: str , #= './descriptors' Path to save the descriptor files
                    save_file: bool = True, # Whether to save the updated DataFrame with descriptors as a CSV file
                    output_df: bool = True, # Whether to return the updated DataFrame with descriptors
                    )->pd.DataFrame | None:
    """Helper function to calculate and save molecular descriptors for specified columns in a DataFrame."""
    from functools import lru_cache
    from descriptastorus.descriptors import rdNormalizedDescriptors
    from tqdm.auto import tqdm
    import os

    @lru_cache(maxsize=None)
    def get_cached_descriptors(mol_input): return generator.calculateMol(mol_input, None)
    generator = rdNormalizedDescriptors.RDKit2DNormalized()

    if save_file: os.makedirs(save_dir, exist_ok=True)

    # Create a copy of the dataset to avoid modifying the original
    df = data_set.copy()
    for col in mol_col:
        print(f"Processing {col}...")
        x = np.stack([get_cached_descriptors(m) for m in tqdm.tqdm(df[col].tolist())])
        print(f"{col} shape: {x.shape}")
        if save_file: np.save(f'{save_dir}/descriptors_for_{col}.npy', x)
        df = pd.concat([df, pd.DataFrame(x, columns=[f"{col}_desc_{i}" for i in range(x.shape[1])])], axis=1)

    # Save the updated DataFrame with descriptors
    if save_file: df.to_csv(f'{save_dir}/dataset_with_descriptors.csv', index=False)
    return df if output_df else None


def get_smiles(df: pd.DataFrame,
               print_errors: bool = False)->pd.DataFrame:
    """Helper function to fetch SMILES strings for chemical components in a DataFrame using PubChemPy, with caching and error handling.
    :param df: Input DataFrame containing 'Component 1' and 'Component 2' columns.
    :param print_errors: Whether to print error messages during fetching.
    :return: DataFrame with added 'Smiles 1' and 'Smiles' columns containing the SMILES strings for the respective components."""

    import time
    from functools import lru_cache
    import pubchempy as pcp
    from rdkit import Chem

    @lru_cache(maxsize=None)
    def obtain_smile(name, retries=3, print_errors=print_errors):
        for i in range(retries):
            try:
                compounds = pcp.get_compounds(name, 'name')
                if compounds:
                    return compounds[0].isomeric_smiles
                return None
            except Exception as e:
                if "SSL" in str(e) or "EOF" in str(e):
                    print(f"SSL Error for {name}, retrying ({i + 1}/{retries})...") if print_errors else None
                    time.sleep(2)  # Give the connection a moment to breathe
                    continue
                print(f"Permanent Error fetching {name}: {e}") if print_errors else None
                break
        return None

    # Apply the cached function to get SMILES for both components
    df['Smiles 1'] = df['Component 1'].apply(obtain_smile)
    df['Smiles 2'] = df['Component 2'].apply(obtain_smile)

    # Replace empty strings with NaN across the specific columns
    df['Smiles 1'] = df['Smiles 1'].replace('', np.nan)
    df['Smiles 2'] = df['Smiles 2'].replace('', np.nan)

    # Check for empty/NaN values in 'Smiles 1' and 'Smiles 2'
    empty_count = (df['Smiles 1'].isnull() | df['Smiles 2'].isnull()).sum()
    print(f"Number of rows with empty/NaN 'Smiles 1' or 'Smiles 2': {empty_count}")

    # Drop rows with empty/NaN values in 'Smiles 1' or 'Smiles 2'
    if empty_count > 0:
        df = df.dropna(subset=['Smiles 1', 'Smiles 2'])
        print(f"New shape after dropping: {df.shape}")

    df['mol1'] = df['Smiles 1'].apply(Chem.MolFromSmiles)
    df['mol2'] = df['Smiles 2'].apply(Chem.MolFromSmiles)

    return df