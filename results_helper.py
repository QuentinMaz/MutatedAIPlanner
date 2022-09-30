import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations


OPTIMAL_RESULTS_FILENAME = 'optimal_results.csv'


def visualise(df: pd.DataFrame, configurations: list[tuple[str, str]], filename: str):
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection='3d')

    df.pop('is_non_optimal')
    problems = df['problem'].unique().tolist()

    results = []
    max = 60
    for i in range(len(problems)):
        results.append([])
        for j in range(len(configurations)):
            s, h = configurations[j]
            if df.index[(df.problem==problems[i]) & (df.search==s) & (df.heuristic==h)].empty:
                results[i].append(np.nan)
            else:
                result = min(max, df.at[df.index[(df.problem==problems[i]) & (df.search==s) & (df.heuristic==h)][0], 'result'])
                results[i].append(result)
    # counts from 0 to len(configurations) from 0 to len(problems)
    xdata = [i for i in range(len(problems)) for _ in range(len(configurations))]
    # counts from 0 to len(configurations) len(problems) times
    ydata = [i for _ in range(len(problems)) for i in range(len(configurations))]
    zdata = [results[i][j] for i, j in zip(xdata, ydata)]
    ax.scatter3D(xdata, ydata, zdata, c=ydata)

    # ax.set_xlabel('X axis: problems')
    ax.set_xticks(ticks=np.arange(len(problems)), labels=problems, rotation=45, fontsize=8)
    # ax.set_ylabel('Y axis: configurations')
    ax.set_yticks(ticks=np.arange(len(configurations)), labels=[f'{s}_{h}' for s, h in configurations])
    ax.set_zlabel('Z axis: lengths of plan')
    ax.set_zlim(0, max)

    fig.savefig(filename, dpi=200)


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
    labels = []
    results = []
    for s1, h1 in configurations:
            labels.append(f'{s1}_{h1}')
            row_res = []
            for s2, h2 in configurations:
                nb_same_results = 0
                sh_df1 = df.loc[(df.search==s1) & (df.heuristic==h1)]
                if len(sh_df1['problem'].unique().tolist()) == 0:
                    row_res.append(np.nan)
                else:
                    sh_df2 = df.loc[(df.search==s2) & (df.heuristic==h2)]
                    for problem in df['problem'].unique().tolist():
                        if not sh_df1.loc[sh_df1.problem==problem].empty and not sh_df2.loc[sh_df2.problem==problem]['result'].empty:
                            nb_same_results += sh_df1.loc[sh_df1.problem==problem]['result'].values[0] == sh_df2.loc[sh_df2.problem==problem]['result'].values[0]
                    row_res.append(100 * nb_same_results / len(sh_df1['problem'].unique().tolist()))
                    # print(f'{s1}_{h1} and {s1}_{h2} share {nb_same_results} results.')
            results.append(row_res)
    data = np.array(results)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(data)
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f'{data[i, j]:.1f}%', ha='center', va='center', color='w')
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


def filter_dataframe_strict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe reduced to the cases where every result is non optimal
    """
    return df[df.is_non_optimal==1]


def aux1(df: pd.DataFrame, configurations: list[tuple[str, str]], problems: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """
    It returns a tuple composed of the list of problems given in parameter and the list of configurations that are non optimal on all the given problems.
    """
    mutants = []
    for problem in problems:
        mutants.append(list(zip(df.loc[(df.problem==problem) & (df.is_non_optimal==1)]['search'], df.loc[(df.problem==problem) & (df.is_non_optimal==1)]['heuristic'])))

    results = []
    for (s, h) in configurations:
        keep_configuration = True
        for m in mutants:
            if (s, h) not in m:
                keep_configuration = False
                break
        if keep_configuration:
            results.append((s, h))

    return problems, results


def aux2(df: pd.DataFrame, configurations: set[tuple[str, str]], problems: list[str]) -> tuple[list[str], set[tuple[str, str]]]:
    """
    Recursive function that computes at each call the intersection between the given set of configurations and the one that are non optimal for the problem poped from the list of problems.
    It returns the list of problems to be tested as well as the non optimal configurations. That is, if the list of problems returned is empty, then the returned configurations are all non optimal for the initial list of problems.
    Otherwise, it returns the list of untested problems and the last non empty set of configurations that have been non optimal until poping up the last problem.
    """
    if problems == []:
        return problems, configurations
    else:
        problem = problems.pop()
        intersection = configurations.intersection(list(zip(df.loc[(df.problem==problem) & (df.is_non_optimal==1)]['search'], df.loc[(df.problem==problem) & (df.is_non_optimal==1)]['heuristic'])))
        if len(intersection) == 0:
            return problems, configurations
        else:
            print(intersection, len(problems))
            return aux2(df, intersection, problems)


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


def rank_configurations(df: pd.DataFrame, configurations: list[tuple[str, str]], problems: list[str]=None) -> list[tuple[str, str]]:
    """
    Ranks each configuration of the dataframe with the mean of their relative performance a the given list of problems.
    If no list of problem is given, it considers all the problems of the dataframe.
    """
    ranks = {c: [] for c in configurations}
    if problems == None:
        problems = df['problem'].unique().tolist()
    for problem in problems:
        problem_df = df.loc[df.problem==problem].sort_values(by='result', ignore_index=True)
        for row in problem_df.iterrows():
            ranks[(row[1].search, row[1].heuristic)].append(row[0])

    means = []
    for (s, h) in configurations:
        means.append((np.mean(ranks[(s, h)]), s, h))
    means.sort()

    return [(s, h) for (_, s, h) in means]


def all_elements(list1, list2) -> bool:
    result = True
    for e in list1:
        if e not in list2:
            result = False
            break
    return result


# TODO: to fix !
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
                tmp_problems = results[i + 1][1]
                reduced_problems = [p for p in problems if p not in tmp_problems]
                reduced_df = df.loc[df.problem.isin(reduced_problems)]
                tmp = pd.concat([reduced_df.loc[(reduced_df.search==s) & (reduced_df.heuristic==h)] for (s, h) in sub_config], ignore_index=True)
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


# TODO: to fix !
def select_problems2(df: pd.DataFrame, configurations: list[tuple[str, str]], target: int) -> dict[int, tuple[list[tuple[str, str]], list[str]]]:
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
    for i in range(target, len(configs), 1):
        best_i = 0
        problems_i = []
        config_i = []
        sub_configs = [list(c) for c in combinations(configs, i)]
        print(f'{len(sub_configs)} sub_configs of size {i} found.')
        for sub_config in sub_configs:
            tmp = pd.concat([df.loc[(df.search==s) & (df.heuristic==h)] for (s, h) in sub_config], ignore_index=True)
            sub_problems = [p for p in problems if p not in results[i - 1][1]] if i != target else problems
            tmp_problems = results[i - 1][1] if i != target else []
            for p in sub_problems:
                if len(tmp.loc[tmp.problem==p]) == len(sub_config):
                    tmp_problems.append(p)
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


# exec(open('results_helper.py').read())
# reads the results and adds to the dataframe the oracle values
df = pd.read_csv('fresults.csv', header=0)
add_oracle_results(df)
# retrieves from the dataframe the configurations
configurations = compute_configurations(df)
# filters the dataframe by considering only mutated enough configurations
df1, config1, filtered_configs = filter_dataframe(df, configurations, 60)
print(f'filtered configurations: {filtered_configs}')
df1 = uniform_dataframe(df1)
config1 = compute_configurations(df1)
results = select_problems2(df1, config1, 18)
################### then it's up to me to decide which trade off to do and then export the dataframe ###################
# result = results[20]
# df2 = df1.loc[df1.problem.isin(result[1])]
# df2 = pd.concat([df2.loc[(df2.search==s) & (df2.heuristic==h)] for (s, h) in result[0]], ignore_index=True)
# df2.to_csv('plot4.csv', index=0)
# plot_results_per_problem(df, 'fplot33.png')
# plot_similarities(df2, result[0], 'plot4.png')
# visualise(df1, config1, 'fplot60strict.png')

# sub_problems = ['pegsol08', 'psr-small05', 'psr-small09', 'sokoban02', 'blocks09', 'miconic03', 'openstacks01', 'blocks06', 'pegsol09', 'satellite01', 'miconic04', 'miconic05', 'transport01', 'depot01', 'newspapers02', 'psr-small02', 'blocks01', 'blocks08', 'travel05', 'rovers02', 'pegsol06', 'tpp05', 'miconic02', 'psr-small06', 'psr-small07', 'tpp04', 'newspapers03', 'blocks07']
# sub_configs = [('f2', 'hdiff'), ('f2', 'hlength'), ('f2', 'hnba'), ('f3', 'hlength'), ('f3', 'hi'), ('f3', 'hnba'), ('f4', 'hmax'), ('f4', 'hdiff'), ('f4', 'hlength'), ('f4', 'hi'), ('f4', 'hg'), ('f4', 'hnba'), ('f5', 'hdiff'), ('f5', 'hlength'), ('f5', 'hnba')]
# sub_df = pd.concat([df.loc[(df.search==s) & (df.heuristic==h) & (df.is_non_optimal==1)] for (s, h) in sub_configs], ignore_index=True)
# print(len(pd.concat([df.loc[(df.search==s) & (df.heuristic==h) & (df.is_non_optimal==1) & (df.problem.isin(sub_problems))] for (s, h) in sub_configs], ignore_index=True)))
# for p in sub_problems:
#     sub_df = df.loc[(df.problem==p) & (df.is_non_optimal==1)]
#     non_optimal_configs_for_p = list(zip(sub_df['search'], sub_df['heuristic']))
#     suspected_optimal_configs = [(s, h) for (s, h) in sub_configs if (s, h) not in non_optimal_configs_for_p]
#     if suspected_optimal_configs != []:
#         print(p, suspected_optimal_configs)

# sub_problems2 = ['miconic02', 'psr-small05', 'psr-small09', 'sokoban02', 'blocks09', 'miconic03', 'openstacks01', 'blocks06', 'pegsol09', 'satellite01', 'miconic04', 'transport01', 'depot01', 'newspapers02', 'psr-small02', 'blocks08', 'psr-small06', 'psr-small07', 'pegsol06', 'tpp04', 'blocks02', 'blocks07', 'gripper02', 'blocks05']
# sub_configs2 = [('f2', 'hdiff'), ('f2', 'hlength'), ('f2', 'hg'), ('f2', 'hnba'), ('f3', 'hmax'), ('f3', 'hdiff'), ('f3', 'hlength'), ('f3', 'hi'), ('f3', 'hg'), ('f3', 'hnba'), ('f4', 'hmax'), ('f4', 'hdiff'), ('f4', 'hlength'), ('f4', 'hi'), ('f4', 'hg'), ('f4', 'hnba'), ('f5', 'hdiff'), ('f5', 'hlength'), ('f5', 'hi'), ('f5', 'hg')]
# sub_df2 = pd.concat([df.loc[(df.search==s) & (df.heuristic==h) & (df.is_non_optimal==1)] for (s, h) in sub_configs2], ignore_index=True)
# print(len(pd.concat([df.loc[(df.search==s) & (df.heuristic==h) & (df.is_non_optimal==1) & (df.problem.isin(sub_problems2))] for (s, h) in sub_configs2], ignore_index=True)))
# for p in sub_problems2:
#     sub_df = df.loc[(df.problem==p) & (df.is_non_optimal==1)]
#     non_optimal_configs_for_p = list(zip(sub_df['search'], sub_df['heuristic']))
#     suspected_optimal_configs = [c for c in sub_configs2 if c not in non_optimal_configs_for_p]
#     if suspected_optimal_configs != []:
#         print(p, suspected_optimal_configs,
#             'BUT:',
#             [(s, h) for (s, h) in suspected_optimal_configs if not df.loc[(df.search==s) & (df.heuristic==h) & (df.problem==p)].empty])