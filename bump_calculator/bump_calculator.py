from collections import namedtuple

import numpy as np

import GPy

BumpResults = namedtuple('BumpResults', ('bump_time', 'bump_time_err',
                                         'probability_bump',
                                         'probability_shoulder',
                                         'number_points',
                                         'gp_model', 'x', 'y_norm',
                                         'y_norm_err',
                                         'x_samples', 'y_samples'))


def _get_bump_flux(x, y, error, kernel):
    # m = GPy.models.GPRegression(x, y, kernel)

    m = GPy.models.GPHeteroscedasticRegression(x, y, kernel)
    m['.*het_Gauss.variance'] = np.abs(error)
    m.het_Gauss.variance.fix()
    m.optimize()

    x_pred = np.arange(13, 40, 0.1)[:, np.newaxis]
    y_pred = m._raw_predict(x_pred)[0].squeeze()

    grad = np.diff(y_pred).squeeze() / np.diff(x_pred.squeeze())
    abs_grad = np.abs(grad)
    xgrad_ = 0.5 * (x_pred[:-1] + x_pred[1:])
    secgrad_ = np.diff(grad).squeeze() / np.diff(xgrad_.squeeze())

    # No bump nor shoulder, default fallback
    bump_time = np.nan
    is_bump = False
    is_shoulder = False

    for i in range(1, len(abs_grad) - 2):
        time = float(x_pred[i])
        if time < x.min() or time > x.max():
            continue

        # Change from positive to negative derivative (top of inverted U).
        # It is a bump, report and exit
        if grad[i] >= 0 and grad[i + 1] < 0:
            bump_time = time
            is_bump = True
            is_shoulder = False
            return bump_time, is_bump, is_shoulder, m, x_pred, y_pred

        # Or change from positive to negative curvature (like -x^3)
        # It is a shoulder, continue scanning
        elif secgrad_[i] >= 0 and secgrad_[i + 1] < 0 and not is_shoulder:
            bump_time = time
            is_shoulder = True
    return bump_time, is_bump, is_shoulder, m, x_pred, y_pred


def get_bump_flux(jd, flux, error, t_max=None, n_bootstrap=15):
    """
    Compute the position of the bump
    :param jd: times, 1D array.
    :param flux: flux, 1D array.
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
    flux = np.asarray(flux)
    error = np.asarray(error)

    if t_max is None:
        t_max = jd[np.nanargmax(flux)]

    jd = jd - t_max
    valid = np.logical_and(jd < 100, jd > -30)
    valid = np.logical_and(valid, ~np.isnan(flux))

    if valid.sum() <= 5:
        raise RuntimeError('Not enough points')

    jd = jd[valid]
    flux = flux[valid]
    error = error[valid]

    # Normalisation
    norm = 1 / error.mean()
    flux *= norm
    error *= norm

    count_points_in_bump = np.sum(np.logical_and(jd > 13, jd < 40))

    x = jd[:, np.newaxis]
    y = flux[:, np.newaxis]
    err = error[:, np.newaxis]

    kernel = GPy.kern.Matern32(1, variance=11000., lengthscale=25)

    bump_time, is_bump, is_shoulder, gp_model, this_x, this_y = _get_bump_flux(
        x, y, err, kernel)
    gp_model = gp_model.copy()  # Save a copy of the model to be reusable

    time_list = []
    is_bump_list = [is_bump]
    is_shoulder_list = [is_shoulder]
    y_list = [this_y]
    for _ in range(n_bootstrap):
        noise = np.random.uniform(low=-1, high=1, size=y.shape)
        y_ = y + noise * error[:, np.newaxis]

        t_noise, is_bump_noise, is_shoulder_noise, _, _, this_y = _get_bump_flux(
            x, y_, err, kernel)
        time_list.append(t_noise)
        is_bump_list.append(is_bump_noise)
        is_shoulder_list.append(is_shoulder_noise)
        y_list.append(this_y)

    time_list = np.asarray(time_list)

    return BumpResults(bump_time, np.nanstd(time_list),
                       np.mean(is_bump_list), np.mean(is_shoulder_list),
                       count_points_in_bump, gp_model, jd, flux, error,
                       this_x.squeeze(), np.asarray(y_list))
