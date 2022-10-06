import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations


OPTIMAL_RESULTS_FILENAME = 'optimal_results.csv'


def add_oracle_results(df: pd.DataFrame) -> None:
    """
    Add the oracle results in the dataframe. In place.
    """
    oracle_results_df = pd.read_csv(OPTIMAL_RESULTS_FILENAME)[['problem', 'result']]
    is_non_optimal = []
    for row in df.iterrows():
        is_non_optimal.append(1 if row[1].result != oracle_results_df.loc[oracle_results_df.problem==row[1].problem].result.values[0] else 0)
    df.insert(len(df.columns), 'is_non_optimal', is_non_optimal, True)


def plot_similarities(df: pd.DataFrame, configurations: list[tuple[str, str]], filename: str) -> None:
    """
    Plots a heatmap where the similarity of each pair of configurations is given.
    """
    problems = df['problem'].unique().tolist()
    df = df.set_index(['problem', 'search', 'heuristic'])
    labels = [f'{s}_{h}' for (s, h) in configurations]
    n = len(labels)
    results = np.zeros((n, n))
    for p in problems:
        for i in range(n - 1):
            for j in range(i + 1, n):
                (s1, h1) = configurations[i]
                (s2, h2) = configurations[j]
                res1 = df.at[(p, s1, h1), 'result']
                res2 = df.at[(p, s2, h2), 'result']
                # results[i, j] += abs(res1 - res2) / res1
                if res1 == res2:
                    results[i, j] += 1
    # plotting
    data = 100 * (results / len(problems))
    fig, ax = plt.subplots(figsize=(18, 14))
    im = ax.imshow(data)
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f'{data[i, j]:.1f}', ha='center', va='center', color='w')
    ax.set_title('Percentage of identical solutions for each mutant')
    fig.tight_layout()
    plt.savefig(filename, dpi=200)


def compute_mutation_coverages(df: pd.DataFrame, configurations: list[tuple[str, str]]) -> list[tuple[str, str, float]]:
    """
    Computes the 'private' mutation coverages of the input configurations (using the dataframe).
    A score for a configuration is computed as the percentage of problems for which it is non optimal.
    The list returned is composed of tuples (search, heuristic, score).
    """
    mutation_coverages = []
    for (s, h) in configurations:
        mutation_coverages.append([s, h, 100 * len(df.loc[(df.search==s) & (df.heuristic==h) & (df.is_non_optimal==1)]) / len(df.loc[(df.search==s) & (df.heuristic==h)])])
        # print(f'{s}_{h} has a private mutation coverage of {mutation_coverages[-1]:.1f}')
    return mutation_coverages


def compute_configurations(df: pd.DataFrame) -> list[tuple[str, str]]:
    """"
    Returns the list of all the configurations of the dataframe.
    """
    configurations = []
    for s in df['search'].unique().tolist():
        for h in df['heuristic'].unique().tolist():
            if not df.loc[(df.search==s) & (df.heuristic==h)].empty:
                configurations.append((s, h))
    print(f'{len(configurations)} configurations found.')
    return configurations


def remove_configurations(df: pd.DataFrame, configurations: list[tuple[str, str]]) -> pd.DataFrame:
    """
    Returns a dataframe where all the data related to the given configurations has been removed.
    """
    dfs = []
    for s in df['search'].unique().tolist():
        for h in df['heuristic'].unique().tolist():
            if not df.loc[(df.search==s) & (df.heuristic==h)].empty and (s, h) not in configurations:
                dfs.append(df.loc[(df.search==s) & (df.heuristic==h)])
    return pd.concat(dfs, ignore_index=True)


def uniform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retuns a dataframe reduced to problems passed by all configurations.
    """
    problems = df['problem'].unique().tolist()
    n = len(compute_configurations(df))
    problems_with_missing_data = [p for p in problems if len(df.loc[df.problem==p]) != n]
    print(f'{len(problems_with_missing_data)} problems with missing data found.')
    df = df.loc[~df.problem.isin(problems_with_missing_data)]
    return df


def filter_dataframe(df: pd.DataFrame, configurations: list[tuple[str, str]], threshold: float) -> tuple[pd.DataFrame, list[tuple[str, str], list[tuple[str, str]]]]:
    """
    Returns a dataframe with only the configurations that are mutated enough (given the threshold input)
    """
    valid_configurations = []
    filtered_configurations = []
    # sorts every configuration
    for [s, h, c] in compute_mutation_coverages(df, configurations):
        if c > threshold:
            valid_configurations.append((s, h))
        else:
            filtered_configurations.append((s, h))
    print(f'{len(valid_configurations)} configurations left after filtration ({threshold}%).')
    return remove_configurations(df, filtered_configurations), valid_configurations, filtered_configurations


def plot_results_per_problem(df: pd.DataFrame, filename: str) -> None:
    """
    Saves a scatter plot where the x axis is the problems and the y axis the length of the solutions. The size of each point scales with the number identical solutions.
    """
    problems = df['problem'].unique().tolist()
    nb_problems = len(problems)
    x = []
    y = []
    s = []
    c = []
    for i in range(nb_problems):
        sub_df = df.loc[df.problem==problems[i]]
        results = np.unique(sub_df['result'].tolist(), return_counts=True)
        y += results[0].tolist()
        s += results[1].tolist()
        x += [(i + 1) for _ in range(len(results[0].tolist()))]
        color = 'b' if i % 2 == 0 else 'r'
        c += [color for _ in range(len(results[0].tolist()))]
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(x, y, (np.array(s) * 10).tolist(), c)
    ax.set_xticks(np.arange(1, nb_problems + 1), labels=problems, rotation=45)
    ax.set_yscale('log')
    plt.savefig(filename)


def rank_configurations(df: pd.DataFrame, configurations: list[tuple[str, str]], problems: list[str]=None) -> tuple[list[tuple[str, str], list[str]]]:
    """
    Ranks each configuration of the dataframe with the mean of their relative performance a the given list of problems.
    If no list of problem is given, it considers all the problems of the dataframe.
    """
    oracle_results = pd.read_csv(OPTIMAL_RESULTS_FILENAME)[['problem', 'result']].set_index('problem').result
    ranks = {c: [] for c in configurations}
    if problems == None:
        problems = df['problem'].unique().tolist()
    for problem in problems:
        problem_df = df.loc[df.problem==problem]
        oracle_result = oracle_results[problem]
        for row in problem_df.iterrows():
            if (row[1].search, row[1].heuristic) in configurations:
                ranks[(row[1].search, row[1].heuristic)].append(100 * (row[1].result - oracle_result) / oracle_result)

    means = []
    for (s, h) in configurations:
        means.append((np.mean(ranks[(s, h)]), s, h))
    means.sort()

    return [(s, h) for (_, s, h) in means], [r for (r, _, _) in means]


def all_elements(list1, list2) -> bool:
    for e in list1:
        if e not in list2:
            return False
    return True


def select_problems(df: pd.DataFrame, configurations: list[tuple[str, str]], target: int) -> dict[int, tuple[list[tuple[str, str]], list[str]]]:
    """
    Function that returns a set of (configurations, problems), indexed by the number of configurations (higher or equal to the target), and for which
    they are all non optimal on the associated problems. So, the tuples returned represents a good test bench for MT.
    """
    print(f'target is {target}.\t\t\t\t\t\t(dataframe length is {len(df)})')
    problems = [p for p in df['problem'].unique().tolist() if len(df.loc[(df.problem==p) & (df.is_non_optimal==1)]) >= target]
    df = df.loc[(df.problem.isin(problems)) & (df.is_non_optimal==1)]
    print(f'{len(problems)} problems that satisfies the target found.\t\t(dataframe length is now {len(df)})')
    configuration_mut_scores = [len(df.loc[(df.search==s) & (df.heuristic==h)]['problem'].unique().tolist()) for (s, h) in configurations]
    configs = []
    for i in range(len(configurations)):
        if configuration_mut_scores[i] >= target:
            configs.append(configurations[i])
    df = pd.concat([df.loc[(df.search==s) & (df.heuristic==h)] for (s, h) in configs], ignore_index=True)
    print(f'{len(configs)} configurations that satisfies the target found.\t(dataframe length is now {len(df)})')
    results = {}
    for i in range(len(configs), target, -1):
        best_i = 0
        problems_i = []
        config_i = []
        sub_configs = [list(c) for c in combinations(configs, i)]
        print(f'{len(sub_configs)} sub_configs of size {i} found.')
        for sub_config in sub_configs:
            # if sub_config is included in the previous result, then we can reduce the dataframe by considering the previous results
            if results.get(i + 1) and all_elements(sub_config, results[i + 1][0]):
                tmp_problems = results[i + 1][1].copy()
                reduced_problems = [p for p in problems if p not in tmp_problems]
                tmp = pd.concat([df.loc[(df.search==s) & (df.heuristic==h) & (df.problem.isin(reduced_problems))] for (s, h) in sub_config], ignore_index=True)
                for p in reduced_problems:
                    if len(tmp.loc[tmp.problem==p]) == len(sub_config):
                        tmp_problems.append(p)
            else:
                tmp = pd.concat([df.loc[(df.search==s) & (df.heuristic==h)] for (s, h) in sub_config], ignore_index=True)
                tmp_problems = []
                for p in problems:
                    if len(tmp.loc[tmp.problem==p]) == len(sub_config):
                        tmp_problems.append(p)
            ##### commun code of the loop ######### #TODO: proper code by using helper functions
            tmp_score = len(tmp_problems)
            if  tmp_score == len(problems):
                print('a sub_config that is valid for all problems has been found.')
                return problems, sub_config
            if tmp_score > best_i:
                print(f'{tmp_score} problems covered.')
                best_i = tmp_score
                config_i = sub_config
                problems_i = tmp_problems
        results[i] = (config_i, problems_i)
    return results


def main_example():
    """
    Example of use.
    """
    return
    # reads the results and adds to the dataframe the oracle values
    df = pd.read_csv('results.csv', header=0)
    add_oracle_results(df)
    # retrieves from the dataframe the configurations
    configurations = compute_configurations(df)
    # filters the dataframe by considering only mutated enough configurations
    df1, config1, filtered_configs = filter_dataframe(df, configurations, 60)
    # print(f'filtered configurations: {filtered_configs}')
    # makes a new usable dataframe by reducing it to problems passed by all mutants
    df1 = uniform_dataframe(df1)
    config1 = compute_configurations(df1)
    # uses this function to have different set of configurations for which they are all non optimal for the associated set of problems
    results = select_problems(df1, config1, 18)
    ################### then it's up to me to decide which trade off to do and then export the dataframe ###################
    sub_problems = results[19][1]
    sub_configs = results[19][0]
    print(f'new set of problems: {sub_problems}')
    df2 = pd.concat([df1.loc[(df1.search==s) & (df1.heuristic==h) & (df1.problem.isin(sub_problems))] for (s, h) in sub_configs], ignore_index=True)
    ranked_configs, _ = rank_configurations(df2, sub_configs, sub_problems)
    print(ranked_configs)


if __name__ == '__main__':
    main_example()

# exec(open('results_helper.py').read())
df = pd.read_csv('results.csv', header=0)
df = uniform_dataframe(df)
configs_to_rank = []
problems_to_use_for_ranking = []
ranked_configs, ranks = rank_configurations(df, configs_to_rank, problems_to_use_for_ranking)
print(len(ranked_configs), ranked_configs)
plot_similarities(df, compute_configurations(df), 'similarities.png')