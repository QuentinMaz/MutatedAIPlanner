import subprocess
import os
import time
import pandas as pd
import re
import json

HEURISTICS = {
    'h_max': 'hmax',
    'h_diff': 'hdiff',
    'h_state_length': 'hlength',
    'h_distance_with_i': 'hi',
    'h_distance_with_g': 'hg'
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


def run_configuration(search: str, heuristic: str, domain_filename: str, problem_filename: str, output: str, timeout: int) -> float:
    """
    Runs a configuration, defined by a search and a heuristic, and returns the execution time.
    """
    command = f'main.exe {search} {heuristic} {domain_filename} {problem_filename} {output}'
    print(command)
    start_time = time.time()
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, timeout=timeout)
    except:
        print('something went wrong.')
        return 0
    return time.time() - start_time


def run_problems(problem_folder: str, search: str, heuristic: str, nb_problems: int, timeout: int, export_logs=False) -> 'list[str]':
    logs = {
        'problem': [],
        'search': [],
        'heuristic': [],
        'execution_time': []
    }
    problems = []
    domains = os.listdir(problem_folder)
    for domain in domains:
        single_domain = 'domain.pddl' in os.listdir(f'{problem_folder}/{domain}')
        for i in range(1, nb_problems):
                problem = f'{domain}0{i}'
                execution_time = run_configuration(search, heuristic,
                    f'{problem_folder}/{domain}/domain.pddl' if single_domain else f'{problem_folder}/{domain}/domain0{i}.pddl',
                    f'{problem_folder}/{domain}/task0{i}.pddl',
                    f'tmp/{problem}_{search}_{HEURISTICS[heuristic]}.txt', timeout)
                print(execution_time)
                # gathers result only if the planner didn't timeout
                if execution_time < timeout and execution_time != 0:
                    problems.append(problem)
                    if export_logs:
                        logs['problem'].append(problem)
                        logs['search'].append(search)
                        logs['heuristic'].append(heuristic)
                        logs['execution_time'].append(execution_time if execution_time < timeout else 0)
    pd.DataFrame.from_dict(logs).to_csv(f'logs_{search}_{HEURISTICS[heuristic]}_{timeout}.csv', index=0)
    summarise_results(problems, search, heuristic)
    return problems


def summarise_results(problems: 'list[str]', search: str, heuristic: str, remove_files=True):
    result_files = [f for f in os.listdir('tmp')]
    regex = re.compile('([a-z0-9-]+)_([a-z]+)_([a-z]+).txt')
    # results for csv exporting
    csv_results = {
        'problem': [],
        'search': [],
        'heuristic': [],
        'result': []
    }
    # results for json exporting
    json_results = {}
    for result_file in result_files:
        m = regex.match(result_file)
        if m != None:
            p = m.group(1)
            s = m.group(2)
            h = m.group(3)
            # only picks up the result of interest
            if p in problems and s == search and h == HEURISTICS[heuristic]:
                f = open(f'tmp/{result_file}', 'r')
                result = len(f.readlines())
                f.close()
                csv_results['problem'].append(p)
                csv_results['search'].append(s)
                csv_results['heuristic'].append(h)
                csv_results['result'].append(result)
                json_results[p] = result
        if remove_files:
            os.remove(f'tmp/{result_file}')
    pd.DataFrame.from_dict(csv_results).to_csv(f'{search}_{heuristic}.csv', index=0)
    f = open(f'{search}_{heuristic}.json', 'w')
    f.write(json.dumps(json_results, sort_keys=True))
    f.close()

if __name__ == '__main__':
    if 'main.exe' not in os.listdir():
        build_main()
    problems = run_problems('experiments', 'astar', 'h_max', 11, 110)
