import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sys


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df.pop('is_non_optimal')
    X = df.copy()
    X = pd.get_dummies(X)
    print(f'length of X is {len(X)}')
    # #numer is the DataFrame that holds all of X's numerical variables
    numer = X[['result']]
    # #cater is the DataFrame that holds all of X's categorical variables
    cater = X[[c for c in X.columns.to_list() if c != 'result']]

    scaler = StandardScaler()
    numer = pd.DataFrame(scaler.fit_transform(numer))
    numer.columns = ['result_Scaled']

    X = pd.concat([numer, cater], axis=1, join='inner')
    return X


def kmeans(X: pd.DataFrame, nb_clusters: int) -> np.ndarray:
    kmeans = KMeans(n_clusters=nb_clusters)
    kmeans.fit(X)

    clusters = kmeans.predict(X)

    return clusters


def plot_clustering(df: pd.DataFrame, cluster_column:str, filename: str) -> None:
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection='3d')

    problems = df['problem'].unique().tolist()
    print(f'{len(problems)} problems.')
    searches = df['search'].unique().tolist()
    print(f'{len(searches)} searches.')
    heuristics = df['heuristic'].unique().tolist()
    print(f'{len(heuristics)} heuristics.')
    clusters = df[cluster_column].unique().tolist()
    print(f'{len(clusters)} clusters.')

    for c in clusters:
        points = []
        for i in range(len(searches)):
            for j in range(len(heuristics)):
                for k in range(len(problems)):
                    if not df.index[(df.problem==problems[i]) & (df.search==searches[i]) & (df.heuristic==heuristics[j]) & (df[cluster_column]==c)].empty:
                        points.append((i, j, k))
                        # cluster = df.at[df.index[(df.problem==problems[i]) & (df.search==searches[i]) & (df.heuristic==heuristics[j])][0], cluster_column]
        print(f'{len(points)} points for cluster {c}.')
        if points != []:
            ax.scatter3D(*zip(*points), label=f'cluster {c}')

    # ax.set_xlabel('X axis: searches')
    ax.set_xticks(ticks=np.arange(len(searches)), labels=searches, rotation=45)
    # ax.set_ylabel('Y axis: searches')
    ax.set_yticks(ticks=np.arange(len(heuristics)), labels=heuristics)
    ax.set_zlabel('Z axis: problems')
    # ax.set_zlim(0, max)
    ax.legend()
    fig.savefig(filename)


def main():
    if len(sys.argv) != 3:
        print('wrong number of arguments.')
        return

    df = pd.read_csv(f'{sys.argv[1]}.csv')
    nb_clusters = int(sys.argv[2])

    X = process_dataframe(df)

    # df['kmeans1'] = kmeans(X, nb_clusters)
    # df['kmeans2'] = kmeans(X, nb_clusters)
    # plot_clustering(df, 'kmeans1', 'kmeans1.png')
    # plot_clustering(df, 'kmeans2', 'kmeans2.png')

    for i in range(1, 10):
        df[f'kmeans{i}'] = kmeans(X, nb_clusters)
        plot_clustering(df, f'kmeans{i}', f'kmeans{i}.png')

    # df.to_csv(f'{sys.argv[1]}_clustered.csv', index=0)

if __name__ == '__main__':
    main()