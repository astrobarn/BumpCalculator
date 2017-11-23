from collections import namedtuple

import numpy as np

import GPy

BumpResults = namedtuple('BumpResults', ('bump_time', 'bump_time_err',
                                         'probability', 'number_points',
                                         'gp_model'))


def _get_bump(x, y, kernel):
    m = GPy.models.GPRegression(x, y, kernel)
    m.optimize()

    x_pred = np.arange(13, 40, 0.1)[:, np.newaxis]
    y_pred = m.predict(x_pred)[0].squeeze()

    grad = np.abs(np.diff(y_pred).squeeze() / np.diff(x_pred.squeeze()))
    grad_ = np.diff(y_pred).squeeze() / np.diff(x_pred.squeeze())
    xgrad_ = 0.5 * (x_pred[:-1] + x_pred[1:])
    secgrad_ = np.diff(grad_).squeeze() / np.diff(xgrad_.squeeze())

    if grad.min() < 0.015 and grad_.max() > 0:
        maybe_bump = True
    else:
        maybe_bump = False

    for i in range(1, len(grad) - 2):
        if grad_[i - 1] > 0 and grad_[i + 1] < 0:
            bump_time = float(x_pred[i])
            is_bump = maybe_bump and bump_time > x.min() and bump_time < x.max()
            return bump_time, is_bump, m

        elif secgrad_[i - 1] > 0 and secgrad_[i + 1] < 0:
            bump_time = float(x_pred[i])
            is_bump = maybe_bump and bump_time > x.min() and bump_time < x.max()
            return bump_time, is_bump, m

    bump_time = float(x_pred[np.argmin(grad)])
    is_bump = maybe_bump and bump_time > x.min() and bump_time < x.max()

    return bump_time, is_bump, m


def get_bump(jd, mag, emag, t_max=None, n_bootstrap=15):
    """
    Compute the position of the bump
    :param jd: times, 1D array.
    :param mag: magnitudes, 1D array.
    :param emag: errors, 1D array.
    :param t_max: time of max, float
    :param n_bootstrap: number of times to repeat booststraping, int.
    :return: tuple:
        - Best estimation of the time of bump
        - Error (std) of the time
        - Probability of bump (number of boostrapped realisations that showed bump)
        - Number of points
        - GP model
    """

    jd = np.asarray(jd)
    mag = -np.asarray(mag)
    err = np.asarray(emag)

    norm = np.nanmax(mag)

    if t_max is None:
        t_max = jd[np.nanargmax(mag)]

    jd = jd - t_max
    valid = np.logical_and(jd < 100, jd > -30)
    valid = np.logical_and(valid, ~np.isnan(mag))

    if valid.sum() <= 5:
        raise RuntimeError('Not enough points')

    jd = jd[valid]
    mag = mag[valid] - norm
    err = err[valid]
    count_points_in_bump = np.sum(np.logical_and(jd > 13, jd < 40))

    x = jd[:, np.newaxis]
    y = mag[:, np.newaxis]
    kernel = GPy.kern.Matern32(1)

    bump_time, is_bump, gp_model = _get_bump(x, y, kernel)
    gp_model = gp_model.copy()  # Save a copy of the model to be reusable

    time_list = []
    is_bump_list = [is_bump]
    for _ in range(n_bootstrap):
        noise = np.random.uniform(low=-1, high=1, size=y.shape)
        y_ = y + noise * err[:, np.newaxis]

        t_noise, is_bump_noise, _ = _get_bump(x, y_, kernel)
        time_list.append(t_noise)
        is_bump_list.append(is_bump_noise)

    time_list = np.asarray(time_list)
    is_bump_list = np.asarray(is_bump_list)

    return BumpResults(bump_time, time_list.std(), is_bump_list.mean(),
                       count_points_in_bump, gp_model)
