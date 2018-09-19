import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--directory', '-d', type=str, default='data/generator/statistical_parity', help='output folder')
parser.add_argument('--settings', '-s', type=str, help='settings split by comma')   

if __name__ == '__main__':
  args = parser.parse_args()
  directory = args.directory
  settings_values = args.settings.split(",")
  settings_labels = ['title','columns','samples','biased','epsilon','proby','proba', 'prob','parameter']
  if settings_values != settings_labels:
    settings = dict(zip(settings_labels, settings_values))

    exp_name, exp_trial = settings['title'].split("-")
    output_filename = "{}/output/report_{}_output.csv".format(directory, settings['title'])
    results_filename = "{}/results/{}_results.csv".format(directory, exp_name)
    results = pd.read_csv(output_filename)

    results_filename = "{}/results/{}_results.csv".format(directory, exp_name)
    validation_results_filename = "{}/results/validation.csv".format(directory, exp_name)

    # Log in overall experiment
    if True in results.checked.values:
      checked_true = results.checked.value_counts()[True]
    else:
      checked_true = 0

    vrf = open(validation_results_filename, "a")
    vrf.write("{},{}\n".format(exp, str(checked_true / 100.0)))
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