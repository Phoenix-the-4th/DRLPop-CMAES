from iohinspector import *
from matplotlib import pyplot as plt
import os
import numpy as np
import json
import pandas as pd
np.random.seed(0)


def success_rates(folder):
    # folder/path/.json files
    paths = [os.path.join(folder, i) for i in os.listdir(folder) if os.path.isdir(os.path.join(folder, i))]
    ls = {i: {j: 0 for j in [2, 3, 5, 10, 20, 40]} for i in paths}  # successes
    le = {i: {j: 0 for j in [2, 3, 5, 10, 20, 40]} for i in paths}  # evals
    for path in paths:
        for fname in os.listdir(path):
            if fname.endswith('.json'):
                dat = json.load(open(os.path.join(path, fname), 'r'))
                for scenario in dat['scenarios']:
                    dim = scenario['dimension']
                    for run in scenario['runs']:
                        ls[path][dim] += run['success']
                        le[path][dim] += run['evals']/(24*15*dim)

    df = pd.DataFrame(ls).sort_index().T
    df['sum'] = df.sum(1)
    print(df.sort_index())  # successes

    df2 = pd.DataFrame(le).sort_index().T
    df2['avg'] = df2.mean(1)
    print(df2.sort_index().astype(int)) # avg evals/dimension


manager = DataManager()
manager.add_folders([os.path.join('results', i) for i in os.listdir('results')])
selections = manager.select()
df = selections.load(include_meta_data=True)
functions = dict(zip(df['function_name'], df['function_id']))



def aocc_table_5():
    a = get_aocc(metrics.utils.transform_fval(df.filter(df['dimension'] == 5)), 50000)
    a = a.pivot(index='function_name', columns='algorithm_name', values='AOCC')
    a = a.T
    a['avg'] = a.mean(axis=1)
    a = a.T
    a["best"] = a.idxmax(axis=1)
    a = a.sort_values('function_name', key= lambda x: x.map(functions))
    a.to_latex('temp.txt')

def aocc_table_averaged():
    lista = [get_aocc(metrics.utils.transform_fval(df.filter(df['dimension'] == d)), d * 10000).pivot(index='function_name', columns='algorithm_name', values='AOCC') for d in [2, 3, 5, 10, 20, 40]]
    combined = sum(lista)/6
    combined = combined.T
    combined['avg'] = combined.mean(axis=1)
    combined = combined.T
    combined["best"] = combined.idxmax(axis=1)
    combined.sort_values('function_name', key= lambda x: x.map(functions)).to_latex('temp.txt')

def saveecdfs():
    plot_ecdf(df)
    plt.savefig(f'graphs/ecdfs/ecdf.png')    
    for fname in functions:
        plot_ecdf(df.filter(df['function_name'] == fname))
        plt.savefig(f'graphs/ecdfs/ecdf_{functions[fname]}_{fname}.png')

def savefixbudget():
    for fname in functions:
        plot_single_function_fixed_budget(df.filter(df['function_name'] == fname))
        plt.savefig(f'graphs/fb/fb_{functions[fname]}_{fname}.png')

def savefixtarget():
    for fname in functions:
        plot_single_function_fixed_target(df.filter(df['function_name'] == fname))
        plt.savefig(f'graphs/ft/ft_{functions[fname]}_{fname}.png')



if __name__ == "__main__":
    success_rates('results')