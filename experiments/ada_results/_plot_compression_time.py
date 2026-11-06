# aggregate all files into one
import pandas as pd
import numpy as np
import glob

DATASET_TO_NAME = {
    "irissetosa.csv": r"Iris\_setosa",
    "irisvirginica.csv": r"Iris\_virginica",
    "irisversicolor.csv": r"Iris\_versicolor",
    "spiral.csv": "Spiral",
    "tic-tac-toe.csv": "Tic-Tac-Toe",
    "compascomplete.csv": "COMPAS",
    "wisconsin.csv": "Wisconsin",
    "ficocomplete.csv": "FICO",
    "browardgeneral2y.csv": "Broward",
    "winequality.csv": "Wine Quality",
    "couponfull.csv": "Coupon",
    "netherlands.csv": "Netherlands",
}

baselines = glob.glob("../tuning/*.csv")
df_list_baselines = [pd.read_csv(f) for f in baselines]
df_baselines = pd.concat(df_list_baselines, ignore_index=True)
grouped_baselines = df_baselines.groupby(['dataset', 'model'])[['auc', 'accuracy']].agg(['mean', 'std', 'count']).reset_index()
grouped_baselines.to_csv("timing_grouped_baselines.csv", index=False)
def get_baseline_stats(dataset, model, metric = 'auc'):
    return grouped_baselines[(grouped_baselines['dataset'] == dataset) & (grouped_baselines['model'] == model)][metric][['mean', 'std', 'count']].values[0]

models = df_baselines['model'].unique()
models = sorted(models, key = lambda m: (10 if 'AUC' in m else (1 if 'RandomForest' in m else 0), m))
models = ['AdaBoost', 'AdaBoost_3']
datasets = df_baselines['dataset'].unique()

# Make latex table; each row is dataset, each column is model, value is mean +- std/np.sqrt(count)
with open("baseline_stats_table.tex", "w") as f:
    f.write("\\begin{tabular}{l" + "c" * len(models) + "}\n")
    f.write("Dataset & " + " & ".join([m.replace("_", "") for m in models]) + " \\\\\n")
    f.write("\\hline\n")
    for dataset in datasets:
        row = [dataset.replace("_", "")]
        for model in models:
            mean, std, count = get_baseline_stats(dataset, model)
            row.append(f"{mean:.3f} $\\pm$ {std/np.sqrt(count):.3f}")
        f.write(" & ".join(row) + " \\\\\n")
    f.write("\\end{tabular}\n")
models = df_baselines['model'].unique()
models = sorted(models, key = lambda m: (10 if 'AUC' in m else (1 if 'RandomForest' in m else 0), m))


# Make latex table; each row is dataset, each column is model, value is mean +- std/np.sqrt(count)
with open("acc_baseline_stats_table.tex", "w") as f:
    f.write("\\begin{tabular}{l" + "c" * len(models) + "}\n")
    f.write("Dataset & " + " & ".join([m.replace("_", "") for m in models]) + " \\\\\n")
    f.write("\\hline\n")
    for dataset in datasets:
        row = [DATASET_TO_NAME[dataset.replace("_", "")]]
        for model in models:
            mean, std, count = get_baseline_stats(dataset, model, metric='accuracy')
            row.append(f"{mean:.3f} $\\pm$ {std/np.sqrt(count):.3f}")
        f.write(" & ".join(row) + " \\\\\n")
    f.write("\\end{tabular}\n")

# aggregate results from compress_ada_nobin
compress_ada_nobin = glob.glob("compress_ada_nobin/*.csv")
df_list_compress_ada_nobin = [pd.read_csv(f) for f in compress_ada_nobin]
df_compress_ada_nobin = pd.concat(df_list_compress_ada_nobin, ignore_index=True)
df_compress_ada_nobin['tree_compression'] = df_compress_ada_nobin['num_unique_trees']/df_compress_ada_nobin['num_trees']
df_compress_ada_nobin['leaf_compression'] = df_compress_ada_nobin['sparsified_node_count']/df_compress_ada_nobin['pre_PE_node_count']
grouped_compress_ada_nobin = df_compress_ada_nobin.groupby(['dataset', 'depth'])[['tree_compression', 'leaf_compression', 'total_compression_time', 'pairwise_equivalence_check_time']].agg(['mean', 'std', 'count']).reset_index()


def get_ada_stats(dataset, binning = False, full_solver = False, depth = 3, metric = 'tree_compression'):
    grouped = grouped_compress_ada_nobin
    try: 
        return grouped[(grouped['dataset'] == dataset) & (grouped['depth'] == depth)][metric][['mean', 'std', 'count']].values[0]
    except IndexError:
        return (np.nan, np.nan, 1)


# table needs to be as follows: 
# each row is a dataset
# columns correspond to compress_ada_nobin leaf compression, compress_ada_bin leaf compression, compress_ada_nobin tree compression, compress_ada_bin tree compression

for depth in [3]:
    with open(f"ada_compression_stats_table_depth_{depth}.tex", "w") as f:
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("Dataset & Ada (full compression) & Ada (Just removing duplicates) \\\\\n")
        f.write("\\hline\n")
        for dataset in datasets:
            row = [DATASET_TO_NAME[dataset.replace("_", "")]]
            for compression_type in ['leaf_compression', 'tree_compression']:
                for binning in [False]:
                    mean, std, count = get_ada_stats(dataset, binning, depth=depth, metric=compression_type)
                    row.append(f"{mean:.3f} $\\pm$ {std/np.sqrt(count):.3f}")
            f.write(" & ".join(row) + " \\\\\n")
        f.write("\\end{tabular}\n")
    # now do runtime table:
    with open(f"ada_compression_time_table_depth_{depth}.tex", "w") as f:
        f.write("\\begin{tabular}{lcc}\n")
        f.write("Dataset & Ada (full compression) & Ada (Just removing duplicates) \\\\\n")
        f.write("\\hline\n")
        for dataset in datasets:
            row = [DATASET_TO_NAME[dataset.replace("_", "")]]
            for compression_type in ['total_compression_time', 'pairwise_equivalence_check_time']:
                for binning in [False]:
                    mean, std, count = get_ada_stats(dataset, binning, depth=depth, metric=compression_type)
                    row.append(f"{mean:.3f} $\\pm$ {std/np.sqrt(count):.3f}")
            f.write(" & ".join(row) + " \\\\\n")
        f.write("\\end{tabular}\n")
