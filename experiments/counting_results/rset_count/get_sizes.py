# %%
import pandas as pd 
import numpy as np

# %%
def compression_sizes(i): 
    df = pd.read_csv(f'{i}.csv')
    df_old = pd.read_csv(f'../rset_count_old/{i}.csv')
    print(df['dataset'].iloc[0])
    string_trivial = f"${df['rset_size_no_trivial_extensions'].mean():.0f} \\pm {df['rset_size_no_trivial_extensions'].std()/ np.sqrt(len(df)):0.0f}$"
    string_base = f"${df['rset_size'].mean():0.0f} \\pm {df['rset_size'].std()/ np.sqrt(len(df)):0.0f}$"
    string_us = f"${df['rset_size_unique'].mean():.0f} \\pm {df['rset_size_unique'].std()/ np.sqrt(len(df)):0.0f}$"
    string_us_no_cont = f"${df_old['rset_size_unique'].mean():.0f} \\pm {df_old['rset_size_unique'].std()/ np.sqrt(len(df_old)):0.0f}$"
    print(f"{string_trivial} & {string_base} & {string_us_no_cont} & {string_us} \\\\ %")

# %%
# relevant indices are 12, 13, 14, 16 (compas, wine, wisc, fico without the unneeded class balancing)
for i in [12, 13, 14, 16]:
    compression_sizes(i)


