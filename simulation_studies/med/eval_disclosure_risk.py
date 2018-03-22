import numpy as np
import pdb

filenames = ['logs_test/presence_risk.txt', 'logs_test/attribute_risk']

for f in filenames:
    risk = np.loadtxt(open(f, 'rb'), delimiter=',')
    mean_sensitivity, mean_precision = np.mean(risk, axis=0)
    std_sensitivity, std_precision = np.std(risk, axis=0)
    print('\nEvaluated file: {}'.format(f))
    print('  Sensitivity (mean +- std) = {:.4f} +- {:.4f}'.format(
        mean_sensitivity, std_sensitivity))
    print('  Precision (mean +- std) = {:.4f} +- {:.4f}'.format(
        mean_precision, std_precision))
