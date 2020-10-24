import numpy as np
import scipy as sp
import scipy.stats as stats

# duration of alive subscriptions
censored = np.array(
    [133.65, 10.26, 0.24, 3.87, 23.84, 25.91, 41.83, 137.805, 0.985, 100.39, 14.9, 18.72, 29.65, 13.11, 26.71, 22.64,
     179.985, 9.6, 37.61, 144.53, 18.855, 80.865, 88.56, 21.955, 73.945, 10.365])
# duration of completed subscriptions
uncensored = np.array(
    [55.31, 47.03, 0.44, 190.41, 80.07, 0.77, 23.93, 151.72, 33.09, 10.9, 140.41, 209.49, 21.38, 40.18, 99.26, 167.52,
     16.75, 109.77, 18.07, 90.23, 233.68, 27.09, 42.35, 109.06, 181.86, 24.5, 66.08, 19.25])


# Log likelihoods for censored data
def log_likelihood_weibull(args):
    shape, scale = args
    val = stats.weibull_min.logpdf(uncensored, shape, loc=0, scale=scale).sum() + stats.weibull_min.logsf(censored,
                                                                                                          shape, loc=0,
                                                                                                          scale=scale).sum()
    return -val  # sign is flipped so we can use a minimizer


def log_likelihood_lomax(args):
    shape, scale = args
    val = stats.lomax.logpdf(uncensored, shape, loc=0, scale=scale).sum() + stats.lomax.logsf(censored, shape, loc=0,
                                                                                              scale=scale).sum()
    return -val


res_weibl = sp.optimize.minimize(log_likelihood_weibull, [1, 1], bounds=((0.001, 1000000), (0.001, 1000000)))
res_lomax = sp.optimize.minimize(log_likelihood_lomax,   [1, 1], bounds=((0.001, 1000000), (0.001, 1000000)))


print("weibull shape", res_weibl.x[0], ", scale=", res_weibl.x[1])
print("weibull mean", stats.weibull_min.mean(res_weibl.x[0], scale=res_weibl.x[1]))
print("weibull median", stats.weibull_min.median(res_weibl.x[0], scale=res_weibl.x[1]))

print("lomax shape", res_lomax.x[0], ", scale=", res_lomax.x[1])
print("lomax mean", stats.lomax.mean(res_lomax.x[0], scale=res_lomax.x[1]))
print("lomax median", stats.lomax.median(res_lomax.x[0], scale=res_lomax.x[1]))