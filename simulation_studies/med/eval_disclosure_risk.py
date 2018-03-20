import numpy as np
import pdb

filename = 'logs_data1_18k/disclosure_risk.txt'
risk = np.loadtxt(open(filename, 'rb'), delimiter=',')
mean_sensitivity, mean_precision = np.mean(risk, axis=0)
std_sensitivity, std_precision = np.std(risk, axis=0)
print('Evaluated file: {}'.format(filename))
print('Sensitivity (mean +- std) = {:.4f} +- {:.4f}'.format(
    mean_sensitivity, std_sensitivity))
print('Precision (mean +- std) = {:.4f} +- {:.4f}'.format(
    mean_precision, std_precision))
