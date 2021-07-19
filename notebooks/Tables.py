# flake8: noqa

# In[1]:


import os
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd

# In[9]:


datasets = ["Adult", "COMPAS", "Give_Me_Some_Credit"]
ml_methods = ["ANN", "Linear"]

cf_ind = ["ar", "as", "cem", "dice", "gs", "wachter"]
cf_dep = [
    "ar-lime",
    "clue",
    "dice_vae",
    "cem-vae",
    "face-knn",
    "face-epsilon",
    "revise",
]
cf_methods = cf_ind + cf_dep

measurements = [
    "ell0",
    "ell1",
    "ell2",
    "ell-inf",
    "redundancy",
    "violation",
    "ynn",
    "success",
    "avgtime",
]


# In[10]:


results = []

# for d in datasets:
#     for clf in ml_methods:
#         for cf in cf_methods:
#             filepath = f"../Results/{d}/{clf}/{cf}.csv"
#
#             if path.exists(filepath):
#                 df_result = pd.read_csv(filepath, index_col=0)
#             else:
#                 df_result = pd.DataFrame(
#                     [[np.NaN] * len(measurements)], columns=measurements
#                 )
#
#             df_result["data"] = d.lower()
#             df_result["clf"] = clf.lower()
#             df_result["cf_method"] = cf.lower().replace("-", "_")
#
#             results.append(df_result)

path = "C:/Users/sbiel/Downloads/results_for_csv"
files = [f for f in listdir(path) if isfile(join(path, f))]

for file in files:
    df = pd.read_csv(os.path.join(path, file))
    results.append(df)
df = pd.concat(results).reset_index(drop=True)
df["Recourse_Method"] = df["Recourse_Method"].apply(
    lambda x: x.lower().replace("-", "_")
)

# df = pd.concat(results)


# In[11]:


def approach(method):
    a = "unknown"
    if method in [m.lower().replace("-", "_") for m in cf_ind]:
        a = "independence"
    elif method in [m.lower().replace("-", "_") for m in cf_dep]:
        a = "dependence"

    return a


df["approach"] = df["Recourse_Method"].apply(approach)


# In[21]:


groupers = ["Dataset", "Recourse_Method", "ML_Model"]

df_agg = df.groupby(groupers, dropna=False, as_index=False).agg(
    count=("Distance_1", "count"),
    ell0=("Distance_1", "mean"),
    ell1=("Distance_2", "mean"),
    ell2=("Distance_3", "mean"),
    ell_inf=("Distance_4", "mean"),
    redundancy=("Redundancy", "mean"),
    redundancy_std=("Redundancy", "std"),
    violation=("Constraint_Violation", "mean"),
    violation_std=("Constraint_Violation", "std"),
    ynn=("y-Nearest-Neighbours", "first"),
    success=("Success_Rate", "first"),
    avgtime=("Average_Time", "first"),
    approach=("approach", "first"),
)

drop_cols = ["ell0", "ell1", "ell2", "ell_inf"]
df_agg = df_agg.drop(drop_cols, axis="columns")

a = df_agg.copy()


# ### Formatting

# In[22]:


df_agg = a

datasets = {"give_me_some_credit": "GMC", "adult": "Adult", "compas": "COMPAS"}
df_agg["Dataset"] = df_agg["Dataset"].replace(datasets)

cf_methods = {
    "cem_vae": "\\texttt{CEM-VAE}",
    "face_eps": "\\texttt{FACE--EPS}",
    "face_knn": "\\texttt{FACE--KNN}",
    "dice_vae": "\\texttt{EB--CF}",
    "clue": "\\texttt{CLUE}",
    "as": "\\texttt{AS}",
    "ar": "\\texttt{AR(--LIME)}",
    "cem": "\\texttt{CEM}",
    "dice": "\\texttt{DICE}",
    "gs": "\\texttt{GS}",
}
df_agg["Recourse_Method"] = df_agg["Recourse_Method"].replace(cf_methods)
# df_agg["cf_method"] = df_agg["cf_method"].str.upper()

df_agg["count"] = df_agg["count"].astype(int)

cols = {
    "count": "$n$",
    "ynn": "\textit{yNN}",
    "redundancy": "redund.",
    "redundancy_std": "redund_std.",
    "violation": "violation",
    "violation_std": "violation_std",
    "success": "success",
    "avgtime": "$\overline{t}(s)$",
}

df_agg = df_agg.rename(cols, axis="columns")


df_dep = df_agg[df_agg["approach"] == "dependence"].drop("approach", axis="columns")
df_ind = df_agg[df_agg["approach"] == "independence"].drop("approach", axis="columns")


# In[23]:


def latex_table(df):

    df = df.melt(
        id_vars=["Dataset", "Recourse_Method", "ML_Model"], var_name="measurement"
    )

    df = df.pivot_table(
        index=["Dataset", "Recourse_Method"], columns=["ML_Model", "measurement"]
    )

    df = df.T.reindex(cols.values(), level="measurement").T

    df = df.droplevel(0, axis=1)
    df = df.droplevel(0, axis=1)
    df[cols["count"]] = df[cols["count"]].astype(int)

    return df


df_latex = latex_table(df_dep)
latex_table(df_ind)


# In[24]:


df_latex = latex_table(df_dep)
latex = df_latex.to_latex(
    float_format="{:0.2f}".format,
    na_rep="--",
    # sparsify=False,
    multirow=True,
    multicolumn=True,
    escape=False,
    #         bold_rows=True,
    label="method_comparison_dependend",
    caption="Dependent methods results",
)

print(latex)


# In[ ]:


# In[ ]:


# In[ ]:
