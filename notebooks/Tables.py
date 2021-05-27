# flake8: noqa

# In[1]:


from os import path

import numpy as np
import pandas as pd

# In[9]:


datasets = ["Adult", "COMPAS", "Give_Me_Some_Credit"]
ml_methods = ["ANN", "Linear"]

cf_ind = ["ar", "as", "cem", "dice", "gs"]
cf_dep = ["ar-lime", "clue", "dice_vae", "cem-vae", "face-knn", "face-eps"]
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

for d in datasets:
    for clf in ml_methods:
        for cf in cf_methods:
            filepath = f"../Results/{d}/{clf}/{cf}.csv"

            if path.exists(filepath):
                df_result = pd.read_csv(filepath, index_col=0)
            else:
                df_result = pd.DataFrame(
                    [[np.NaN] * len(measurements)], columns=measurements
                )

            df_result["data"] = d.lower()
            df_result["clf"] = clf.lower()
            df_result["cf_method"] = cf.lower().replace("-", "_")

            results.append(df_result)

df = pd.concat(results)


# In[11]:


def approach(method):
    a = "unknown"
    if method in [m.lower().replace("-", "_") for m in cf_ind]:
        a = "independence"
    elif method in [m.lower().replace("-", "_") for m in cf_dep]:
        a = "dependence"

    return a


df["approach"] = df["cf_method"].apply(approach)


# In[21]:


groupers = ["data", "cf_method", "clf"]

df_agg = df.groupby(groupers, dropna=False, as_index=False).agg(
    count=("ell0", "count"),
    ell0=("ell0", "mean"),
    ell1=("ell1", "mean"),
    ell2=("ell2", "mean"),
    ell_inf=("ell-inf", "mean"),
    redundancy=("redundancy", "mean"),
    violation=("violation", "mean"),
    ynn=("ynn", "first"),
    success=("success", "first"),
    avgtime=("avgtime", "first"),
    approach=("approach", "first"),
)

drop_cols = ["ell0", "ell1", "ell2", "ell_inf"]
df_agg = df_agg.drop(drop_cols, axis="columns")

a = df_agg.copy()


# ### Formatting

# In[22]:


df_agg = a

datasets = {"give_me_some_credit": "GMC", "adult": "Adult", "compas": "COMPAS"}
df_agg["data"] = df_agg["data"].replace(datasets)

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
df_agg["cf_method"] = df_agg["cf_method"].replace(cf_methods)
# df_agg["cf_method"] = df_agg["cf_method"].str.upper()

df_agg["count"] = df_agg["count"].astype(int)

cols = {
    "count": "$n$",
    "ynn": "\textit{yNN}",
    "redundancy": "redund.",
    "violation": "violation",
    "success": "success",
    "avgtime": "$\overline{t}(s)$",
}

df_agg = df_agg.rename(cols, axis="columns")


df_dep = df_agg[df_agg["approach"] == "dependence"].drop("approach", axis="columns")
df_ind = df_agg[df_agg["approach"] == "independence"].drop("approach", axis="columns")


# In[23]:


def latex_table(df):

    df = df.melt(id_vars=["data", "cf_method", "clf"], var_name="measurement")

    df = df.pivot_table(index=["data", "cf_method"], columns=["clf", "measurement"])

    df = df.T.reindex(cols.values(), level="measurement").T

    df = df.droplevel(0, axis=1)
    df = df.droplevel(0, axis=1)
    df[cols["count"]] = df[cols["count"]].astype(int)

    return df


df_latex = latex_table(df_dep)
latex_table(df_ind)


# In[24]:


df_latex = latex_table(df_ind)
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
