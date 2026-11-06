import pandas as pd
import numpy as np

all_res = []
for i in range(400):
    try:
        all_res.append(pd.read_csv(f"experiments/hashing_results/{i}.csv"))
    except:
        continue
res = pd.concat(all_res, axis=0)
res = res.fillna("None")
res = res[res['rashomon_ignore_trivial_extensions'] == True]

# plot EfficientContinuousMissTree time_new_compression vs EfficientContinuousMissTree time_to_eval_model_set when hashing_type is None vs EfficientContinuousMissTree time_to_eval_model_set when hashing_type is training_data, with x axis on log scale representing rset size and y axis on log scale representing these values (time), and color by hashing_type 
import matplotlib.pyplot as plt
hashing_sample_count = 100

plt.figure(figsize=(10,4))
res_tmp = res
    
res_none = res_tmp[res_tmp["hashing_type"] == "None"]
rset_size = res_none[res_none["tree_type"] == "EfficientContinuousMissTree"]['rset_size']
unique_rset_size = res_none[res_none["tree_type"] == "EfficientContinuousMissTree"]['rset_size_unique']

new_compression_time = res_none[res_none["tree_type"] == "EfficientContinuousMissTree"]["time_new_compression"] + res_none[res_none["tree_type"] == "EfficientContinuousMissTree"]["time_to_eval_model_list"]
pairwise_baseline = res_none[res_none["tree_type"] == "EfficientContinuousMissTree"]["time_to_eval_model_set"]
comparison_var_counts = []
comparison_runtimes_dnf = []
reference_points = res_none[res_none["tree_type"] == "EfficientContinuousMissTree"]
for i in range(len(new_compression_time)):
    dataset = reference_points["dataset"].iloc[i]
    depth = reference_points["depth"].iloc[i]
    lammy = reference_points["lambda"].iloc[i]
    fold = reference_points["fold"].iloc[i]
    trivial = reference_points["rashomon_ignore_trivial_extensions"].iloc[i]
    eps = reference_points["eps"].iloc[i]
    tree_type = 'DNF'
    row = res_tmp[(res_tmp["dataset"] == dataset) & (res_tmp["depth"] == depth) & (res_tmp["lambda"] == lammy) & (res_tmp["tree_type"] == tree_type) & (res_tmp["fold"] == fold)& (res_tmp["rashomon_ignore_trivial_extensions"] == trivial) & (res_tmp["eps"] == eps)]
    if row.shape[0] == 0:
        runtime = pd.NA
        var_count = pd.NA
    else:
        runtime = row["time_to_eval_model_list"].iloc[0] + row["time_to_eval_model_set"].iloc[0]
        var_count = row['largest_unique_splits'].iloc[0]
        if row['rset_size'].iloc[0] != reference_points['rset_size'].iloc[i]:
            assert False, f"Rset size mismatch for dataset {dataset} depth {depth} lambda {lammy} fold {fold}"
    comparison_var_counts.append(var_count)
    comparison_runtimes_dnf.append(runtime)

plt.scatter(rset_size * unique_rset_size, new_compression_time, label="Ours")
plt.scatter(rset_size * unique_rset_size, pairwise_baseline, label="Pairwise Baseline", color='tab:orange')
NA_mask = pd.isna(comparison_runtimes_dnf) | pd.isna(comparison_var_counts)
var_counts = pd.Series(comparison_var_counts)[~NA_mask].reset_index(drop=True)
runtimes_no_na = pd.Series(comparison_runtimes_dnf)[~NA_mask].reset_index(drop=True)
unique_rset_size_no_na = unique_rset_size[~NA_mask].reset_index(drop=True)
rset_size_no_na = rset_size[~NA_mask].reset_index(drop=True)
easy_mask = var_counts <= 7
hard_mask = var_counts > 7
# plot dnf baseline
plt.scatter(rset_size_no_na[easy_mask] * unique_rset_size_no_na[easy_mask], runtimes_no_na[easy_mask], label="QM Compression (each tree <8 unique splits)", color = 'green')
plt.scatter(rset_size_no_na[hard_mask] * unique_rset_size_no_na[hard_mask], runtimes_no_na[hard_mask], label="QM Compression (some tree >8 unique splits)", color = 'purple')
plt.scatter(rset_size[~NA_mask] * unique_rset_size[~NA_mask], pd.Series(new_compression_time)[~NA_mask], color='tab:blue')
legend = plt.legend(loc='lower right', fontsize='large')#(0.2, 0.5))
texts = legend.get_texts()
texts[0].set_weight('bold')

plt.xlabel("Original Rset Size * Compressed Rset Size", fontsize='x-large')
plt.xlabel("Original Rset Size * Compressed Rset Size", fontsize='x-large')
plt.ylabel("Time (s)", fontsize='x-large')
# plt.title("Ours vs Pairwise Baseline", fontsize='x-large')
# plt.title("Ours vs QM, Partitioned by Tree Complexity", fontsize='x-large')
plt.title('Log Runtime vs |Rset| * |Compressed Rset|')
# set x axis font size to large
plt.tick_params(axis='x', labelsize='large')
plt.tick_params(axis='x', labelsize='large')
plt.yscale('log')
plt.tight_layout()
plt.savefig(f'experiments/time_comp_product.png')
plt.savefig(f'experiments/time_comp_product.pdf')