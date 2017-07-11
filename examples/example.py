from __future__ import absolute_import

import numpy as np
import bump_calculator
import pylab as plt
plt.ioff()

data = np.loadtxt('SN2007af_r.dat')

results = bump_calculator.get_bump(data[:, 0], data[:, 1], data[:, 2])
print(results)

results.gp_model.plot()
plt.axvline(results.bump_time, color='b')

plt.ylim(-data[:, 1].ptp() - 0.5, 0.5)
plt.fill_betweenx(np.arange(*plt.ylim()),
                  results.bump_time - 3 * results.bump_time_err,
                  results.bump_time + 3 * results.bump_time_err,
                  alpha=0.1)
plt.show()
