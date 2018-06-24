from __future__ import absolute_import

import numpy as np
import bump_calculator
import pylab as plt

plt.ioff()

data = np.loadtxt('SN2007af_r.dat')

results = bump_calculator.get_bump_flux(data[:, 0], np.exp(-data[:, 1]),
                                        np.exp(-data[:, 1]) * data[:, 2])

x_pred = np.arange(-30, 100, 0.1)
y_pred, y_pred_var = results.gp_model._raw_predict(x_pred[:, None])
y_pred = y_pred.squeeze()
y_pred_var = y_pred_var.squeeze()

plt.scatter(results.x, results.y_norm, s=10)
plt.errorbar(results.x, results.y_norm, yerr=results.y_norm_err,
             fmt='+', color='b')
plt.plot(x_pred, y_pred, color='k')
plt.fill_between(x_pred, y_pred - np.sqrt(y_pred_var),
                 y_pred + np.sqrt(y_pred_var), color='b', alpha=0.1)
plt.axvline(results.bump_time, color='b')
plt.fill_betweenx(np.arange(*plt.ylim()),
                  results.bump_time - 3 * results.bump_time_err,
                  results.bump_time + 3 * results.bump_time_err,
                  alpha=0.1)
plt.xlabel('$\mathrm{Time}~(days)$')
plt.ylabel('$\mathrm{Flux}$')
plt.show()
