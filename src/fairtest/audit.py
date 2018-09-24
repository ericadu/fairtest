import csv
import numpy as np
import os
from random import random
import argparse
import pandas as pd
from scipy import stats
import re # import the regular expressions module
import statistical_parity_generator as spg
import counterfactual_generator as cg

import fairtest.utils.prepare_data as prepare
from fairtest import Testing, train, test, report, DataSource

parser = argparse.ArgumentParser()
parser.add_argument('--directory', '-d', type=str, default='data/generator/statistical_parity', help='output folder')
parser.add_argument('--settings', '-s', type=str, help='settings split by comma')

def check_settings(expected, actual):
  p_biased, p_unbiased = actual[5:7]

  biased = expected[-1]
  p = p_biased if biased == True else p_unbiased
  x_corr, a_corr = actual[7:]

  expected_values = [round(i, 1) for i in expected[:6]]
  actual_values = [round(i, 1) for i in list(actual[:5]) + [p]]

  if expected_values != actual_values:
    return False

  if (biased and p_biased != 0.5 and round(x_corr, 1) == 0) or (not biased and round(x_corr, 1) != 0):
      return False

  return True

def run(settings):
  # Extract Settings
  exp = settings['title']
  m = int(settings['columns'])
  n = int(settings['samples'])
  biased = False if settings['biased'] == 'False' else True
  eps = float(settings['epsilon'])
  delta = float(settings['delta'])
  # p_y_A = float(settings['proby'])
  # p_a = float(settings['proba'])
  p = float(settings['p'])


  OUTPUT_DIR = "{}/output".format(directory)
  # Initializing parameters for experiment
  EXPL = []
  SENS = ['A']
  TARGET = 'O'
  output_filename = "{}/output/report_{}_output.csv".format(directory, exp)
  validation_filename = "{}/validation/{}.csv".format(directory, exp)

  write_vf_header = False
  if not os.path.exists(validation_filename):
    write_vf_header = True
  vf = open(validation_filename, "a")
  if write_vf_header:
    # vf.write('m,n,eps,p_y_A,p_a,p_biased,p_unbiased,x_corr,a_corr\n')
    vf.write('m,n,delta,eps\n')

  write_output_header = False
  if not os.path.exists(output_filename):
    write_output_header = True
  f = open(output_filename, "a")

  if write_output_header:
    # f.write('lower,upper,pval,checked\n')
    f.write('lower,upper,pval\n')

    # Generate Dataset
  # df = spg.generate_dataset(exp, m, n, biased, eps, p_y_A, p_a, p)
  # validated = spg.validate_dataset(df)
  # checked = check_settings([m, n, eps, p_y_A, p_a, p, biased], validated)
  df = cg.generate_dataset(m, n, biased, delta, p)
  validated = cg.validate_dataset(df)
  vf.write(','.join([str(round(i, 4)) for i in validated]) + '\n')

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
        # f.write(",".join(intervals[1:] + [str(float(pval[0])), str(checked)]) + "\n")
        f.write(",".join(intervals[1:] + [str(float(pval[0]))]) + "\n")
    else:
        print(pval)
        exit()     

if __name__ == '__main__':
  args = parser.parse_args()
  directory = args.directory
  settings_values = args.settings.split(",")
  # settings_labels = ['title','columns','samples','biased','epsilon','proby','proba', 'prob','parameter']
  settings_labels = ['title','columns','samples','biased','delta','epsilon','p','parameter']
  if settings_values != settings_labels:
    settings = dict(zip(settings_labels, settings_values))
    run(settings)


