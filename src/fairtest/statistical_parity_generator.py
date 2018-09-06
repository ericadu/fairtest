import sys
import numpy as np
from random import random
import argparse
import pandas as pd
from scipy import stats
import re # import the regular expressions module
import os.path

import fairtest.utils.prepare_data as prepare
from fairtest import Testing, train, test, report, DataSource

parser = argparse.ArgumentParser()
parser.add_argument('--columns', '-c', type=int, default=1, help='number of extra columns. needs to be greater than or equal to 0')
parser.add_argument('--samples', '-s', type=int, default=10000, help='number of samples')
parser.add_argument('--row', '-r', type=str, default='random', help='what type of row. select from the following list [random, ok, corr]')
parser.add_argument('--epsilon', '-e', type=float, default=0.01, help='epsilon statistical parity')
parser.add_argument('--directory', '-d', type=str, default='data/generator/statistical_parity', help='output folder')
parser.add_argument('--probability', '-p', type=float, default=0.5, help='probability variable')
parser.add_argument('--prob0', type=float, default=0.0)
parser.add_argument('--prob1', type=float, default=0.0)

def get_cols():
  return ['outcome', 'protected'] + ['f{}'.format(i) for i in range(num_columns)]

# P[outcome = 1 | x = 1] = 0.5, P[outcome = 1 | x = 0] = 0.5
def get_random_row():
  return [1 if random() < prob else 0] + [1 if random() < 0.5 else 0 for _ in range(num_columns + 1)]

# P[outcome = 1 | y = 1] = 0.8, P[outcome = 1 | y = 0] = 0.3, y is an unprotected attribute
# P[outcome = 1 | x = 1] = 0.5, P[outcome = 1 | x = 0] = 0.5, x is a protected attribute
def get_biased_by_nonprotected_row():
  row = get_random_row()
  if num_columns < 1:
    print("Pick a different row type or increase number of columns to something greater than 0.")
    exit()

  attribute = row[2]

  row[0] = 1 if (attribute and random() < prob0) or (not attribute and random() < prob1) else 0
  return row

def get_corr_row():
  row = get_random_row()
  if num_columns < 1:
    print("Pick a different row type or increase number of columns to something greater than 0.")
    exit()
  attribute = row[2]

  p = random()
  flipped = 0 if attribute == 1 else 1
  if p < prob:
    row[0] = attribute
  else:
    row[0] = flipped
  return row

def check_statistical_parity(data):
  group_by_protected = data.groupby(['protected', 'outcome']).size()
  total_absent = data.protected.value_counts()[0]
  total_present = data.protected.value_counts()[1]

  p_positive_absent = float(group_by_protected[0][1]) / total_absent
  p_positive_present = float(group_by_protected[1][1]) / total_present
  epsilon = abs(p_positive_absent - p_positive_present)
  if epsilon < 0.01:
    # print("Satisfies statistical parity, epsilon = {}".format(str(epsilon)))
    return True
  else:
    # print("Generating a different dataset, epsilon = {}.".format(str(epsilon)))
    return False

def get_row():
  if row_type == 'random':
    return get_random_row()
  elif row_type == 'ok':
    return get_biased_by_nonprotected_row()
  elif row_type == 'corr':
    return get_corr_row()
  else:
    print("Row type {} is invalid".format(row_type))
    exit()

def validate_args():
  if num_samples < 0:
    print("Need a positive number of samples.") and exit()

  if num_columns < 0:
    print("Need a positive number of columns.") and exit()

  if epsilon > 0.5:
    print("Pick a meaningful epsilon less than 0.5.") and exit()

if __name__ == '__main__':
  args = parser.parse_args()
  directory = args.directory
  num_columns = args.columns
  num_samples = args.samples
  row_type = args.row
  epsilon = args.epsilon
  prob = args.probability

  # For ok bias
  prob0 = args.prob0
  prob1 = args.prob1

  correlation = 0
  protected_correlation = 0

  input_data = np.zeros((num_samples, num_columns + 2)) 
  test_data = np.zeros((num_samples, num_columns + 2)) 

  column_names = get_cols()

  satisfies = False
  while not satisfies:
    for i in range(num_samples):
      input_data[i,:] = get_row()

    input_df = pd.DataFrame(data=input_data, index=range(0,num_samples),columns=column_names)
    pearsons = stats.pearsonr(input_df.f0.values, input_df.outcome.values)
    pearsons_protected = stats.pearsonr(input_df.protected.values, input_df.outcome.values)

    satisfies = check_statistical_parity(input_df)

    if row_type == 'corr':
      if pearsons[1] < 0.05:
        correlation = pearsons[0]
        protected_correlation = pearsons_protected[0]
      else:
        satisfies = False

  validate_args()
  # filename = '{}/{}.csv'.format(directory, row_type)

  # with open(filename, 'w') as f:
  #   column_names = get_cols(num_columns)
  #   f.write(','.join(column_names) + '\n')
  #   for _ in range(num_samples):
  #     row = get_row()
  #     f.write(','.join([str(i) for i in row]) + '\n')

  OUTPUT_DIR = "{}/{}".format(directory, prob)
  # Initializing parameters for experiment
  EXPL = []
  SENS = ['protected']
  TARGET = 'outcome'

  data_source = DataSource(input_df)

  # Instantiate the experiment
  inv = Testing(data_source, SENS, TARGET, EXPL, random_state=0)

  # Train the classifier
  train([inv])

  # Evaluate on the testing set
  test([inv])

  # Create the report
  report([inv], "{}_output".format(row_type), OUTPUT_DIR)

  input_filename = "{}/{}/report_{}_output.txt".format(directory, prob, row_type)

  matched = None
  with open(input_filename, 'rt') as in_file:  # Open file for reading of text data.
      contents = in_file.read()
      #pattern = re.compile(r"/^CI = \[\-?\d*\.\d*, \-?\d*\.\d*\]$/") # compile the regex "\bd\w*r\b" into a pattern object
      pattern = "CI = \[\-?\d*\.\d*, \-?\d*\.\d*\]"
      match = re.search(pattern, contents)
      matched = match.group()

      pval_pattern = "p\-value = \d*\.?\d*e\-?\d*"
      pval_match = re.search(pval_pattern, contents)
      pval_matched = pval_match.group()

  if matched != None and pval_matched != None:
      intervals = re.findall(r'-?\d*\.\d*', matched)
      pval = re.findall(r'\d*\.?\d*e\-\d*', pval_matched)
      if len(pval) > 0:
          pval_value = pval[0]
          intervals.append(str(float(pval_value)))
      else:
          print(pval)
          exit()

      output_filename = "{}/report_{}_output.csv".format(directory, row_type)
      write_header = False
      if not os.path.exists(output_filename):
          write_header = True
      with open(output_filename, 'a') as f:
          if write_header:
              f.write("p,lower,upper,pval,corr,protected_corr,protected_conf\n")
          f.write(",".join([str(prob)] + intervals + [str(correlation), str(protected_correlation), str(pearsons_protected[1])]) + "\n")
  else:
      print("No match.")
