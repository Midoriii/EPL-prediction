import ast
import pandas as pd
import sys
import os

# Get the params dictionary
file = open("eval/grid_results_params.txt", "r")

contents = file.read()
dictionary_params = ast.literal_eval(contents)
file.close()

#print(dictionary_params)

# Get the accuracy results
dictionary_acc = pd.read_csv("eval/grid_results.csv", header=None, index_col=0, squeeze=True).to_dict()

#print(dictionary_acc)

# Get the matching params to the top 15 results
with open('eval/grid_results_combined_top15.txt', 'w') as f:
    for x in list(dictionary_acc)[0:15]:
        print(x, file=f)
        print(dictionary_params['params ' + x], file=f)
