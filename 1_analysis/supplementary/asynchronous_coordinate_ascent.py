import numpy as np
import matplotlib.pylab as plt


def simulate_updates(beta,
                     max_iter=100,
                     parallel=False,
                     parallel_after=0,
                     r=0.5,
                     alternate=False,
                     rho=1.):
    q = np.zeros(2)
    eta = np.zeros(2)
    eta_diff = np.zeros(2)

    track = {
        'q': [],
        'eta': [],
        'eta_diff': []
    }

    for i in range(max_iter):

        parallel_cond = parallel and i >= parallel_after
        if parallel_cond and alternate:
            if i % 2 == 0:
                parallel_cond = not parallel_cond

        # Update the first variant:
        new_eta = beta[0] - q[0]
        eta_diff[0] = rho * new_eta - rho * eta[0]
        eta[0] += eta_diff[0]
        new_q_1 = q[1] + r * eta_diff[0]
        if not parallel_cond:
            q[1] = new_q_1

        # Update the second variant:

        new_eta = beta[1] - q[1]
        eta_diff[1] = rho * new_eta - rho * eta[1]
        eta[1] += eta_diff[1]
        q[0] += r * eta_diff[1]

        if parallel_cond:
            q[1] = new_q_1

        track['q'].append(q.copy())
        track['eta'].append(eta.copy())
        track['eta_diff'].append(eta_diff.copy())

    for k, v in track.items():
        track[k] = np.array(v)

    return track


np.random.seed(7209)

beta = np.random.normal(size=2)

res = simulate_updates(beta, parallel=False, r=.9, )
res_parallel = simulate_updates(beta, parallel=True, r=.9, rho=1.)

for k in res.keys():
    plt.plot(np.arange(res[k].shape[0]), res[k][:, 0], c='blue', label='serial')
    plt.plot(np.arange(res[k].shape[0]), res[k][:, 1], c='blue', label='serial')
    plt.plot(np.arange(res_parallel[k].shape[0]), res_parallel[k][:, 0], c='red', label='parallel')
    plt.plot(np.arange(res_parallel[k].shape[0]), res_parallel[k][:, 1], c='red', label='parallel')
    plt.title(k)
    plt.legend()
    plt.show()
