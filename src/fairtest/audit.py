import csv
import numpy as np
import os
from random import random
import argparse
import pandas as pd
from scipy import stats
import re # import the regular expressions module
# from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool 
import statistical_parity_generator as spg

import fairtest.utils.prepare_data as prepare
from fairtest import Testing, train, test, report, DataSource

parser = argparse.ArgumentParser()
parser.add_argument('--directory', '-d', type=str, default='data/generator/statistical_parity', help='output folder')
parser.add_argument('--settings', '-s', type=str, default='settings.csv', help='file path for experiment settings')
parser.add_argument('--runs', '-r', type=int, default=100, help='number of trials')
parser.add_argument('--threads', '-t', type=int, default=10, help='number of threads')

def run(settings):
  # Extract Settings
  exp = settings['title']
  m = int(settings['columns'])
  n = int(settings['samples'])
  biased = False if settings['biased'] == 'False' else True
  eps = float(settings['epsilon'])
  p_y_A = float(settings['proby'])
  p_a = float(settings['proba'])
  p = float(settings['prob'])


  OUTPUT_DIR = "{}/output".format(directory)
  # Initializing parameters for experiment
  EXPL = []
  SENS = ['A']
  TARGET = 'O'
  output_filename = "{}/output/report_{}_output.csv".format(directory, exp)

  with open(output_filename, 'w') as f:
    f.write("lower,upper,pval\n")

    for _ in range(num_trials):
        # Generate Dataset
      dataset = spg.generate_dataset(exp, m, n, biased, eps, p_y_A, p_a, p)
      columns = ['X{}'.format(str(i)) for i in range(m)] + ['A', 'O']
      # quick data processing

      df = pd.DataFrame(data=dataset, columns=columns)

      data_source = DataSource(df)

      # Instantiate the experiment
      inv = Testing(data_source, SENS, TARGET, EXPL, random_state=0)

      # Train the classifier
      train([inv])

      # Evaluate on the testing set
      test([inv])

      # Create the report
      report([inv], "output_{}".format(exp), OUTPUT_DIR)

      input_filename = "{}/output/report_output_{}.txt".format(directory, exp)

      with open(input_filename, 'rt') as in_file:  # Open file for reading of text data.
        contents = in_file.read()
        pval_pattern = "p\-value = \d*\.?\d*e[\-|\+]?\d* ; CORR = \[\-?\d*\.\d*, \-?\d*\.\d*\]"
        pval_match = re.findall(pval_pattern, contents)

        index = len(pval_match) - 1
        selected = pval_match[index]

        intervals = re.findall(r'-?\d*\.\d*', selected)
        pval = re.findall(r'\d*\.?\d*e[\-|\+]\d*', selected)
        if len(pval) > 0 and len(intervals) > 0:
            f.write(",".join(intervals + pval) + "\n")
        else:
            print(pval)
            exit()  

  exp_name, exp_trial = exp.split("-")
  results_filename = "{}/results/{}_results.csv".format(directory, exp_name)
  results = pd.read_csv(output_filename)
  param = settings['parameter']
  results_columns = [param, 'FP']

  fp_count = results[results.pval < 0.05].count()['pval']

  write_header = False
  if not os.path.exists(results_filename):
    write_header = True
  results_file = open(results_filename, "a")

  if write_header:
    results_file.write(','.join(results_columns) + '\n')

  results_file.write(','.join([settings[param], str(fp_count)]) + '\n')      

if __name__ == '__main__':
  args = parser.parse_args()
  directory = args.directory
  settings_filename = args.settings
  num_trials = int(args.runs)
  num_threads = int(args.threads)

  all_experiments = csv.DictReader(open(settings_filename))

  pool = ThreadPool(num_threads)
  pool.map(run, all_experiments)
  pool.close() 
  pool.join()