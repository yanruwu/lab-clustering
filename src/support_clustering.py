from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn_extra.cluster import KMedoids

import pandas as pd
from collections import Counter


def search_best_model(data, random_state = None, sort_metric = "silhouette_score"):
    results = {"model" : [], "distance" : [], "silhouette_score" : [], "davies_bouldin_score" : [], "n_clusters" : [], "cardinality" : []}
    dist = ["euclidean", "cosine", "manhattan", "chebyshev"]
    for k in range(2,16):
        for d in dist:
            if d == "euclidean":
                model = KMeans(random_state=random_state, n_clusters=k)
                results["model"].append("Kmeans")
            else:
                model = KMedoids(metric = d, random_state=random_state, n_clusters=k)
                results["model"].append("Kmedoids")
            model.fit(data)
            labels = model.labels_
            results["distance"].append(d)
            results["silhouette_score"].append(silhouette_score(X = data, labels=labels))
            results["davies_bouldin_score"].append(davies_bouldin_score(X = data, labels=labels))
            results["n_clusters"].append(k)
            counts = Counter(labels)
            results["cardinality"].append(counts)
    results_df =  pd.DataFrame(results)
    if sort_metric == "silhouette_score":
        results_df = results_df.sort_values(by = sort_metric, ascending=False).reset_index(drop = True)
    elif sort_metric == "davies_bouldin_score":
        results_df = results_df.sort_values(by = sort_metric, ascending=True).reset_index(drop = True)
    return results_df