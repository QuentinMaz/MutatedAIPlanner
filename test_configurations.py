import multiprocessing
import subprocess
import time
import os
import re
import pandas as pd
import argparse
import json


HEURISTICS = {
    'h_max': 'hmax',
    'h_diff': 'hdiff',
    'h_state_length': 'hlength',
    'h_distance_with_i': 'hi',
    'h_distance_with_g': 'hg',
    'h_nb_actions': 'hnba'
}


def build_main() -> None:
    """
    Builds the planner (main.exe) and removes all the artifacts that are created during the process.
    """
    compile_command = 'sicstus --goal "compile(main), save_program(\'main.sav\'), halt."'
    build_command = 'cd "C:\Program Files (x86)\Microsoft Visual Studio\\2019\Community\VC\Auxiliary\Build" && vcvars64.bat && cd "C:\\Users\Quentin\Documents\\5INFO\Simula\MutatedAIPlanner" && spld --output=main.exe --static main.sav'
    try:
        subprocess.run(compile_command, shell=True, stdout=subprocess.DEVNULL)
        subprocess.run(build_command, shell=True, stdout=subprocess.DEVNULL)
        for artifact in ['main.sav', 'main.pdb', 'main.ilk', 'main.exp', 'main.lib']:
            os.remove(artifact)
    except:
        print('something went wrong')


def run_configuration(search: str, heuristic: str, domain_filename: str, problem_filename: str, output: str, timeout: int) -> 'tuple[str, str, str, float]':
    """
    Runs a configuration, defined by a search and a heuristic, and returns the execution time.
    """
    problem = re.compile('.+/(.+)_.+_.+.txt').match(output).group(1)

    if os.path.exists(output):
        print(f'{output} already exists. Execution canceled.')
        return (problem, search, heuristic, 1.0)

    command = f'main.exe mutated_astar-{search} {heuristic} {domain_filename} {problem_filename} {output}'
    print(command)
    start_time = time.time()
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, timeout=timeout)
    except:
        print(f'something went wrong for {search}_{heuristic} on {problem}.')
    return (problem, search, heuristic, time.time() - start_time)


def get_arguments(searches: 'list[str]', heuristics: 'list[str]', folder: str, nb_problems: int, timeout: int) -> 'tuple[list[str],list[tuple[str, str, str, str, str, int]]]':
    problems = []
    args = []
    tmp_args = []
    domains = os.listdir(folder)
    for domain in domains:
        single_domain = 'domain.pddl' in os.listdir(f'{folder}/{domain}')
        for i in range(1, nb_problems):
                pn = f'{domain}0{i}'
                dfp = f'{folder}/{domain}/domain.pddl' if single_domain else f'{folder}/{domain}/domain0{i}.pddl'
                pfp = f'{folder}/{domain}/task0{i}.pddl'
                tmp_args.append([dfp, pfp, pn])
                problems.append(pn)
    for [d, p, pname] in tmp_args:
        for s in searches:
            for h in heuristics:
                args.append([s, h, d, p, f'tmp/{pname}_{s}_{HEURISTICS[h]}.txt', timeout])
    return problems, args


def get_arguments_from_problems(searches: 'list[str]', heuristics: 'list[str]', folder: str, problems: 'list[str]', timeout: int) -> 'list[tuple[str, str, str, str, str, int]]':
    args = []
    tmp_args = []
    regex = re.compile('.+(\d)')
    domains = os.listdir(folder)
    for domain in domains:
        single_domain = 'domain.pddl' in os.listdir(f'{folder}/{domain}')
        for problem in [p for p in problems if domain in p]:
                i = regex.match(problem).group(1)
                dfp = f'{folder}/{domain}/domain.pddl' if single_domain else f'{folder}/{domain}/domain0{i}.pddl'
                pfp = f'{folder}/{domain}/task0{i}.pddl'
                tmp_args.append([dfp, pfp, problem])
    for [d, p, pname] in tmp_args:
        for s in searches:
            for h in heuristics:
                args.append([s, h, d, p, f'tmp/{pname}_{s}_{HEURISTICS[h]}.txt', timeout])
    return args


def log_results(results: 'list[tuple[str, str, str, float]]', timeout: int) -> None:
    logs = {
        'problem': [],
        'search': [],
        'heuristic': [],
        'execution_time': []
    }
    for p, s, h, e in results:
        logs['problem'].append(p)
        logs['search'].append(s)
        logs['heuristic'].append(HEURISTICS[h])
        logs['execution_time'].append(e if e < timeout else 0)
    pd.DataFrame.from_dict(logs).to_csv('logs.csv', index=0)
    print('logs exported (logs.csv).')


def summarise_results(problems: 'list[str]', searches: 'list[str]', heuristics: 'list[str]', results_filename: str, keep_files=False):
    result_files = [f for f in os.listdir('tmp')]
    # columns-ordered dictionnary to export results as .csv file
    csv_results = {
        'problem': [],
        'search': [],
        'heuristic': [],
        'result': []
    }
    # dicitonnary to export results as a .json file. The structure is 2 level dictionnaries:
    #   - the first keys describe each configuration possible
    #   - the associated values are dictionnaries where each key is the name of a problem and its value the length of the result
    json_results = {}
    for search in searches:
        for heuristic in heuristics:
            json_results[f'{search}_{HEURISTICS[heuristic]}'] = {}
            for problem in problems:
                result_file = f'{problem}_{search}_{HEURISTICS[heuristic]}.txt'
                if result_file in result_files:
                    f = open(f'tmp/{result_file}', 'r')
                    result = len(f.readlines())
                    f.close()
                    csv_results['problem'].append(problem)
                    csv_results['search'].append(search)
                    csv_results['heuristic'].append(HEURISTICS[heuristic])
                    csv_results['result'].append(result)
                    json_results[f'{search}_{HEURISTICS[heuristic]}'][problem] = result
                    if not keep_files:
                        os.remove(f'tmp/{result_file}')
    # exports the results
    pd.DataFrame.from_dict(csv_results).to_csv(f'{results_filename}.csv', index=0)
    f = open(f'{results_filename}.json', 'w')
    f.write(json.dumps(json_results, sort_keys=True))
    f.close()
    print(f'results exported ({results_filename}.csv).')


def main():
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('-pf', '--problem_folder', dest='problem_folder', type=str, default='benchmarks')
    argparser.add_argument('-np', '--number_problems', dest='nb_problems', type=int, default=10)
    argparser.add_argument('-t', '--timeout', dest='timeout', type=int, default=110)
    argparser.add_argument('-o', '--result_filename', dest='results_filename', type=str, default='results')
    argparser.add_argument('-log', '--export_logs', dest='log', default=False, action='store_true')
    argparser.add_argument('-k', '--keep_results', dest='keep_results', default=False, action='store_true')
    args = argparser.parse_args()

    searches = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
    heuristics = ['h_max', 'h_diff', 'h_state_length', 'h_distance_with_i', 'h_distance_with_g', 'h_nb_actions']

    # problems, my_args = get_arguments(searches, heuristics, args.problem_folder, args.nb_problems, args.timeout)

    # problems solved by the prolog oracle
    # problems = ['airport01', 'airport02', 'airport03', 'airport04', 'airport05', 'airport06', 'airport07', 'blocks01', 'blocks02', 'blocks03', 'blocks04', 'blocks05', 'blocks06', 'blocks07', 'blocks08', 'blocks09', 'depot01', 'gripper01', 'gripper02', 'logistics06', 'logistics08', 'miconic01', 'miconic02', 'miconic03', 'miconic04', 'miconic05', 'openstacks01', 'parcprinter01', 'parcprinter02', 'pegsol01', 'pegsol02', 'pegsol03', 'pegsol04', 'pegsol05', 'pegsol06', 'pegsol07', 'pegsol09', 'psr-small01', 'psr-small02', 'psr-small03', 'psr-small04', 'psr-small05', 'psr-small06', 'psr-small07', 'psr-small08', 'psr-small09', 'rovers01', 'rovers02', 'rovers04', 'satellite01', 'sokoban01', 'sokoban02', 'sokoban03', 'tpp01', 'tpp02', 'tpp03', 'tpp04', 'transport01', 'transport02', 'woodworking01']
    # problems solved by the pyperplan oracle
    # problems = ['airport01', 'airport02', 'airport03', 'airport04', 'airport05', 'airport06', 'airport07', 'airport08', 'blocks01', 'blocks02', 'blocks03', 'blocks04', 'blocks05', 'blocks06', 'blocks07', 'blocks08', 'blocks09', 'depot01', 'depot02', 'gripper01', 'gripper02', 'gripper03', 'gripper04', 'logistics01', 'logistics02', 'logistics03', 'logistics04', 'logistics05', 'logistics06', 'logistics08', 'miconic01', 'miconic02', 'miconic03', 'miconic04', 'miconic05', 'miconic06', 'movie01', 'movie02', 'movie03', 'movie04', 'movie05', 'movie06', 'movie07', 'movie08', 'movie09', 'newspapers01', 'newspapers02', 'newspapers03', 'openstacks01', 'openstacks02', 'openstacks03', 'parcprinter01', 'parcprinter02', 'parcprinter03', 'parcprinter04', 'pegsol01', 'pegsol02', 'pegsol03', 'pegsol04', 'pegsol05', 'pegsol06', 'pegsol07', 'pegsol08', 'pegsol09', 'psr-small01', 'psr-small02', 'psr-small03', 'psr-small04', 'psr-small05', 'psr-small06', 'psr-small07', 'psr-small08', 'psr-small09', 'rovers01', 'rovers02', 'rovers03', 'rovers04', 'satellite01', 'satellite02', 'satellite03', 'scanalyzer01', 'sokoban01', 'sokoban02', 'sokoban03', 'sokoban04', 'sokoban05', 'sokoban06', 'sokoban07', 'sokoban09', 'tpp01', 'tpp02', 'tpp03', 'tpp04', 'tpp05', 'transport01', 'transport02', 'travel02', 'travel03', 'travel04', 'travel05', 'travel06', 'travel07', 'travel08', 'woodworking01', 'woodworking02']
    # plot3.csv
    # problems = ['blocks06', 'blocks07', 'blocks08', 'blocks09', 'depot01', 'gripper01', 'gripper02', 'miconic02', 'miconic03', 'miconic04', 'openstacks01', 'pegsol05', 'pegsol06', 'pegsol09', 'psr-small02', 'psr-small05', 'psr-small06', 'psr-small07', 'psr-small09', 'satellite01', 'sokoban02', 'tpp04', 'transport01']

    problems = ['airport01', 'airport02', 'airport03', 'airport04', 'airport05', 'airport06', 'airport07', 'blocks01', 'blocks02', 'blocks03', 'blocks04', 'blocks05', 'blocks06', 'depot01', 'gripper01', 'gripper02', 'miconic01', 'miconic02', 'miconic03', 'miconic04', 'movie01', 'movie02', 'movie03', 'movie04', 'movie05', 'movie06', 'movie07', 'movie08', 'movie09', 'newspapers02', 'openstacks01', 'parcprinter01', 'parcprinter02', 'pegsol01', 'pegsol02', 'pegsol03', 'pegsol04', 'pegsol05', 'pegsol06', 'pegsol07', 'psr-small01', 'psr-small02', 'psr-small03', 'psr-small04', 'psr-small05', 'psr-small06', 'psr-small07', 'psr-small08', 'psr-small09', 'satellite01', 'sokoban03', 'tpp01', 'tpp02', 'tpp03', 'tpp04', 'transport01', 'travel02', 'travel03', 'travel06']
    my_args = get_arguments_from_problems(searches, heuristics, args.problem_folder, problems, args.timeout)
    print(f'{len(problems)} planning problems found. {len(my_args)} tasks computed.')

    pool = multiprocessing.Pool(processes=3)
    results = pool.starmap(run_configuration, my_args, chunksize=10)

    # if args.log:
    #     log_results(results, args.timeout)

    summarise_results(problems, searches, heuristics, args.results_filename, args.keep_results)


if __name__ == '__main__':
    if 'main.exe' not in os.listdir():
        build_main()
    main()