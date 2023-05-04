from os import listdir
from os.path import join

import pandas as pd
import seaborn as sns

data = pd.read_csv(join("results", sorted(listdir("./results"))[-1]))
items = data.groupby(["model", "task1", "task2"], dropna=False).size().reset_index().drop(0, axis=1).values

for model, task1, task2 in items:
    print(model, task1, task2)
    filtered = data[(data["model"] == model) & (data["task1"] == task1) &
                    (data["task2"] == task2 if not pd.isnull(task2) else data["task2"].isnull())]

    g = sns.FacetGrid(filtered, col="target", row="metric", hue="phase", sharex=False, sharey="row", margin_titles=True)
    g.map(sns.lineplot, "epoch", "value")
    g.add_legend()
    g.savefig(f"graphs/{model}_{task1}_{task2}.png")
