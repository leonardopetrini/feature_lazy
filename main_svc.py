import os
import argparse
import torch
import time
import signal
from functools import partial
import subprocess

import math
import numpy as np
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from scipy import spatial

from dataset import get_binary_dataset
from kernels import compute_kernels

class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler():   # Custom signal handler
    raise TimeoutException


def kernel(args, xtr, xte):

    dtrtr = torch.cdist(xtr, xtr, p=2)
    dtetr = torch.cdist(xte, xtr, p=2)

    if args.kernel == "laplace":
        return (- dtrtr / args.sigma).exp(), (- dtetr / args.sigma).exp()
    if args.kernel == "gaussian":
        return (-0.5 * (dtrtr / args.sigma).pow(2)).exp(), (-0.5 * (dtetr / args.sigma).pow(2)).exp()
    # pay attention with the next part
    if args.kernel == "feature":
        predicate = lambda a: a.ptr == 2000
        filename = 'sphere_qhingeL2'
        for _, f0, f, args, d in load_data_with_f(
            filename, predicate=predicate, t=-1):
            ktrtr, ktetr, _ = compute_kernels(f, xtr, xte)
            break
    if args.kernel == "ntk":
        predicate = lambda a: a.ptr == 2000
        filename = 'sphere_qhingeL2'
        for _, f0, f, args, d in load_data_with_f(
            filename, predicate=predicate, t=-1):
            ktrtr, ktetr, _ = compute_kernels(f, xtr, xte)
            break
    if args.kernel == "symfeature":

def func(clf, ktetr):
    '''
    :param clf: SVC classifier from sklearn.svm
    :param ktetr: Gram matrix computed at (testing points) x (training set)
    :return: f(x) = sum_\mu alpha_mu K(x, x_mu) + bias
    '''

    # compute charges
    alpha = torch.tensor(clf.dual_coef_[0])
    
    return ktetr[:, clf.support_] @ alpha + clf.intercept_


def samesign(a, b):
    return a * b > 0


def bisect_oncircle(K, clf, xtr, rlow, rhigh, t, tolerance=None, max_iter=100):
    '''
    Find a zero, angularly, for SVC
    '''

    xl = torch.tensor([[rlow * t.cos(), rlow * t.sin()]])
    xh = torch.tensor([[rhigh * t.cos(), rhigh * t.sin()]])

    kh, kl = K(xtr, xh), K(xtr, xl)

    assert not samesign(func(clf, kl), func(clf, kh)), 'The function sould have opposite signs at the exterma of the considered interval'

    for i in range(max_iter):
        midpoint = (rlow + rhigh) / 2.
        xm = torch.tensor([[midpoint * t.cos(), midpoint * t.sin()]])
        km = K(xtr, xm)
        if samesign(func(clf, kl), func(clf, km)):
            rlow = midpoint
        else:
            rhigh = midpoint
        if tolerance is not None and abs(rhigh - rlow) < tolerance:
            break
    return midpoint


def add_SV(args, clf, xtr, ytr, K):
    r = []
    dalpha = []
    R = 2 ** .5

    for i in range(args.num_probes):

        xi = torch.randn(1) / 10
        theta = torch.randn(1) * 2 * math.pi
        xn = torch.tensor([[(R + xi) * theta.cos(), (R + xi) * theta.sin()]])
        yn = xi.sign()

        xtrn = torch.cat((xtr, xn), dim=0)
        ytrn = torch.cat((ytr, yn.view(1)))

        ktrtrn = K(xtrn, xtrn)

        clfn = svm.SVC(C=args.C, kernel="precomputed", max_iter=-1)
        clfn.fit(ktrtrn, ytrn)
        assert sum(clfn.support_ == args.ptr), 'The added probing point is not a SV'

        alpha = torch.zeros(xtr.shape[0])
        alpha[clf.support_] = torch.tensor(clf.dual_coef_[0])

        alphan = torch.zeros(xtr.shape[0])
        alphan[clfn.support_[clfn.support_ != args.ptr]] = torch.tensor(clfn.dual_coef_[0, clfn.support_ != args.ptr])

        if args.angular_distance:
            utr = xtr / xtr.norm(dim=1, keepdim=True)
            un = xn / xn.norm(dim=1, keepdim=True)

            ang_dist = torch.acos(utr @ un.t())[:, 0].sort()
            distances, indices = ang_dist.values, ang_dist.indices
        else:
            tree = spatial.cKDTree(xtr)
            distances, indices = tree.query(xn, k=args.ptr)
            distances, indices = distances[0], indices[0]

        dalpha.append((alpha - alphan).abs()[indices])
        r.append(distances)

    dalpha = torch.stack(dalpha)
    r = torch.stack([torch.tensor(ri) for ri in r])

    n_bins = 50
    bins = torch.logspace(np.log10(1e-3), np.log10(math.pi), n_bins)
    dalphas = torch.zeros(n_bins,)
    count = torch.zeros(n_bins,)

    indices = np.digitize(r, bins)

    for i, row in enumerate(indices):
        for j, bin_idx in enumerate(row):
            dalphas[bin_idx] += dalpha[i, j]
            count[bin_idx] += 1

    return bins, dalphas.div(count)


def run(args):

    res = dict()

    tic = time.time()
    xtr, ytr, xte, yte = init(args)
    print("Generated data")
    K = partial(kernel, args)

    ktrtr = K(xtr, xtr)

    print('Training...')
    clf = svm.SVC(C=args.C, kernel="precomputed", max_iter=-1)
    clf.fit(ktrtr, ytr)

    res["clf"] = clf

    # compute the test error
    ktetr = K(xtr, xte)
    ote = func(clf, ktetr)
    res['test_error'] = ((ote * yte) < 0).sum().float().div(args.pte)

    tac = time.time()
    res["wall"] = round(tac - tic, 2)

    print('Test error: {:.02f}'.format(res['test_error']))

    if args.compute_all:
        # compute rc
        x = xtr[clf.support_]
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(x.numpy())
        distances, _ = nbrs.kneighbors(x.numpy())
        res['rc'] = distances[:, 1].mean()

        # compute S(k)
        R = args.R
        eps = args.Reps  # delta around R to look for zeros
        na = args.num_angles  # number of angles to look at

        ts = torch.linspace(-math.pi, math.pi, na)
        delta = torch.zeros(na,)
        try:
            for i, t in enumerate(ts):
                delta[i] += bisect_oncircle(K, clf, xtr, R - eps, R + eps, t, tolerance=1e-5, max_iter=100) - R
            res['delta'] = delta
            res['S'] = delta.rfft(1).norm(dim=1).pow(2)
        except:
            res['delta'] = None
            res['S'] = None

        # probe disturbance with additional SV
        try:
            res['r_probe'], res['dalpha'] = add_SV(args, clf, xtr, ytr, K)
        except:
            print("No probing, error!!!")
            res['r_probe'], res['dalpha'] = None, None

    yield res


def init(args):
    torch.backends.cudnn.benchmark = True
    if args.dtype == 'float64':
        torch.set_default_dtype(torch.float64)
    if args.dtype == 'float32':
        torch.set_default_dtype(torch.float32)

    [(xte, yte, ite), _, (xtr, ytr, itr)] = get_binary_dataset(
        args.dataset,
        (args.pte, 20, args.ptr),
        (args.seed_testset + args.pte, 0, args.seed_trainset + args.ptr),
        args.d,
        (0, 0),
        args.device,
        torch.get_default_dtype()
    )
    return xtr, ytr, xte, yte


def main():
    git = {
        'log': subprocess.getoutput('git log --format="%H" -n 1 -z'),
        'status': subprocess.getoutput('git status -z'),
    }

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--dtype", type=str, default='float64')

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ptr", type=int, required=True)
    parser.add_argument("--pte", type=int, default=1000)
    parser.add_argument("--seed_trainset", type=int, default=0)
    parser.add_argument("--seed_testset", type=int, default=0)

    parser.add_argument("--test", type=int, default=0)

    parser.add_argument("--d", type=int, required=True)

    parser.add_argument("--kernel", type=str, default="laplace")
    parser.add_argument("--sigma", type=float, default=1.)
    parser.add_argument("--C", type=float, default=1e20)
    parser.add_argument("--max_time", type=float, default=600)

    parser.add_argument("--compute_all", type=int, default=0)

    parser.add_argument("--R", type=float, default=1.414213)
    parser.add_argument("--Reps", type=float, default=1.5)
    parser.add_argument("--num_angles", type=int, default=5000)

    # testing the minimal disturbance hypothesis probing solution with new SV
    parser.add_argument("--num_probes", type=int, default=100)
    parser.add_argument("--angular_distance", type=int, default=0)

    parser.add_argument("--pickle", type=str, required=True)
    args = parser.parse_args()

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.max_time)

    torch.save(args, args.pickle)
    saved = False
    try:
        for res in run(args):
            res['git'] = git
            res['args'] = args
            with open(args.pickle, 'wb') as f:
                torch.save(args, f)
                torch.save(res, f)
                saved = True
    except:
        if not saved:
            os.remove(args.pickle)
        raise


if __name__ == "__main__":
    main()
