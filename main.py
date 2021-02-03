# pylint: disable=C, R, bare-except, arguments-differ, no-member, undefined-loop-variable, not-callable
import argparse
import copy
import os
import subprocess
from functools import partial
from time import perf_counter

import torch

from arch import CV, FC, FixedAngles, FixedWeights, FixedBetas, FixedNorm, Wide_ResNet, Conv1d, CrownInit, MFAngles
from arch.fc import LinearNetwork
from arch.fa2 import LinearFANetwork
from arch.mnas import MnasNetLike, MNISTNet
from arch.swish import swish
from dataset import get_binary_dataset
from dynamics import train_kernel, train_regular, loglinspace
from kernels import compute_kernels, kernel_intdim, eigenvectors

import numpy as np

def loss_func(args, o, y):
    if args.loss == 'hinge':
        return (args.loss_margin - args.alpha * o * y).relu() / args.alpha ** args.loss_over_alpha_power
    if args.loss == 'softhinge':
        sp = partial(torch.nn.functional.softplus, beta=args.loss_beta)
        return sp(args.loss_margin - args.alpha * o * y) / args.alpha ** args.loss_over_alpha_power
    if args.loss == 'qhinge':
        return 0.5 * (args.loss_margin - args.alpha * o * y).relu().pow(2) / args.alpha ** args.loss_over_alpha_power
    if args.loss == 'linear':
        return (args.loss_margin - args.alpha * o * y) / args.alpha ** args.loss_over_alpha_power
    if args.loss == 'hinge_pt':
        # Divide by true p(t) the loss instead of total p. Should speed-up dynamics towards the end
        # add a rescaling to test loss
        pt = (args.alpha * o * y < args.loss_margin).sum()
        return (args.loss_margin - args.alpha * o * y).relu() / args.alpha ** args.loss_over_alpha_power * (args.ptr / pt)


def loss_func_prime(args, f, y):
    if args.loss == 'hinge':
        return -((args.loss_margin - args.alpha * f * y) > 0).double() * y / args.alpha ** (args.loss_over_alpha_power - 1)
    if args.loss == 'softhinge':
        return -torch.sigmoid(args.loss_beta * (args.loss_margin - args.alpha * f * y)) * y / args.alpha ** (args.loss_over_alpha_power - 1)
    if args.loss == 'qhinge':
        return -(args.loss_margin - args.alpha * f * y).relu() * y / args.alpha ** (args.loss_over_alpha_power - 1)
    if args.loss == 'linear':
        return - y.double() / args.alpha ** (args.loss_over_alpha_power - 1)
    if args.loss == 'hinge_L2':
        raise NotImplementedError

class SplitEval(torch.nn.Module):
    def __init__(self, f, size):
        super().__init__()
        self.f = f
        self.size = size

    def forward(self, x):
        return torch.cat([self.f(x[i: i + self.size]) for i in range(0, len(x), self.size)])


def run_kernel(prefix, args, ktrtr, ktetr, ktete, xtr, ytr, xte, yte):
    assert args.f0 == 1

    assert ktrtr.shape == (len(xtr), len(xtr))
    assert ktetr.shape == (len(xte), len(xtr))
    assert ktete.shape == (len(xte), len(xte))
    assert len(yte) == len(xte)
    assert len(ytr) == len(xtr)

    tau = args.tau_over_h_kernel * args.h
    if args.tau_alpha_crit is not None:
        tau *= min(1, args.tau_alpha_crit / args.alpha)

    margin = 0

    checkpoint_generator = loglinspace(args.ckpt_step, args.ckpt_tau)
    checkpoint = next(checkpoint_generator)

    wall = perf_counter()
    dynamics = []
    for state, otr, alpha, _velo, grad in train_kernel(ktrtr, ytr, tau, partial(loss_func_prime, args), args.max_dgrad, args.max_dout / args.alpha):
        save_outputs = args.save_outputs
        save = stop = False

        if state['step'] == checkpoint:
            checkpoint = next(checkpoint_generator)
            save = True
        if torch.isnan(otr).any():
            save = stop = True
        if wall + args.max_wall_kernel < perf_counter():
            save = save_outputs = stop = True
        mind = (args.alpha * otr * ytr).min().item()
        if mind > margin:
            margin += 0.5
            save = save_outputs = True
        if mind > args.stop_margin:
            save = save_outputs = stop = True
        if args.train_kernel == 0:
            save = save_outputs = stop = True

        if not save:
            continue

        state['grad_norm'] = grad.norm().item()
        state['wall'] = perf_counter() - wall

        state['train'] = {
            'loss': loss_func(args, otr, ytr).mean().item(),
            'aloss': args.alpha * loss_func(args, otr, ytr).mean().item(),
            'err': (otr * ytr <= 0).double().mean().item(),
            'nd': (args.alpha * otr * ytr < args.loss_margin).long().sum().item(),
            'mind': (args.alpha * otr * ytr).min().item(),
            'maxd': (args.alpha * otr * ytr).max().item(),
            'dfnorm': otr.pow(2).mean().sqrt().item(),
            'alpha_norm': alpha.norm().item(),
            'outputs': otr.detach().cpu() if save_outputs else None,
            'labels': ytr if save_outputs else None,
        }

        # if len(xte) > len(xtr):
        #     from hessian import gradient
        #     a = gradient(f(xtr) @ alpha, f.parameters())
        #     ote = torch.stack([gradient(f(x[None]), f.parameters()) @ a for x in xte])
        # else:
        ote = ktetr @ alpha

        state['test'] = {
            'loss': loss_func(args, ote, yte).mean().item(),
            'aloss': args.alpha * loss_func(args, ote, yte).mean().item(),
            'err': (ote * yte <= 0).double().mean().item(),
            'nd': (args.alpha * ote * yte < args.loss_margin).long().sum().item(),
            'mind': (args.alpha * ote * yte).min().item(),
            'maxd': (args.alpha * ote * yte).max().item(),
            'dfnorm': ote.pow(2).mean().sqrt().item(),
            'outputs': ote.detach().cpu() if save_outputs else None,
            'labels': yte if save_outputs else None,
        }

        print(("[{prefix}] [i={d[step]:d} t={d[t]:.2e} wall={d[wall]:.0f}] [dt={d[dt]:.1e} dgrad={d[dgrad]:.1e} dout={d[dout]:.1e}]" + \
               " [train aL={d[train][aloss]:.2e} err={d[train][err]:.2f} nd={d[train][nd]}/{ptr} mind={d[train][mind]:.3f}]" + \
               " [test aL={d[test][aloss]:.2e} err={d[test][err]:.2f}]").format(prefix=prefix, d=state, ptr=len(xtr), pte=len(xte)), flush=True)
        dynamics.append(state)

        if (args.ptr - state["train"]["nd"]) / args.ptr > args.stop_frac:
            stop = True

        out = {
            'dynamics': dynamics,
            'kernel': None,
        }

        if stop:
            out['kernel'] = {
                'train': {
                    'value': ktrtr.detach().cpu() if args.store_kernel == 1 else None,
                    'diag': ktrtr.diag().detach().cpu(),
                    'mean': ktrtr.mean().item(),
                    'std': ktrtr.std().item(),
                    'norm': ktrtr.norm().item(),
                    'intdim': kernel_intdim(ktrtr),
                    'eigenvectors': eigenvectors(ktrtr, ytr),
                },
                'test': {
                    'value': ktete.detach().cpu() if args.store_kernel == 1 else None,
                    'diag': ktete.diag().detach().cpu(),
                    'mean': ktete.mean().item(),
                    'std': ktete.std().item(),
                    'norm': ktete.norm().item(),
                    'intdim': kernel_intdim(ktete),
                    'eigenvectors': eigenvectors(ktete, yte),
                },
            }

        yield out
        if stop:
            break


def run_regular(args, f0, xtr, ytr, xte, yte):

    with torch.no_grad():
        ote0 = f0(xte)
        otr0 = f0(xtr)

    if args.f0 == 0:
        ote0 = torch.zeros_like(ote0)
        otr0 = torch.zeros_like(otr0)

    tau = args.tau_over_h * args.h
    if args.tau_alpha_crit is not None:
        tau *= min(1, args.tau_alpha_crit / args.alpha)

    best_test_error = 1
    wall_best_test_error = perf_counter()
    tmp_outputs_index = -1
    margin = 0

    checkpoint_generator = loglinspace(args.ckpt_step, args.ckpt_tau)
    checkpoint = next(checkpoint_generator)

    wall = perf_counter()
    dynamics = []

    for state, f, otr, _otr0, grad, _bi in train_regular(f0, xtr, ytr, tau,
                                                         partial(loss_func, args), args.l2_decay, bool(args.f0),
                                                         args.chunk, args.bs, args.max_dgrad, args.max_dout / args.alpha):
        save_outputs = args.save_outputs
        save = stop = False

        if state['step'] == checkpoint:
            checkpoint = next(checkpoint_generator)
            save = True
        if torch.isnan(otr).any():
            save = stop = True
        if wall + args.max_wall < perf_counter():
            save = save_outputs = stop = True
        if args.wall_max_early_stopping is not None and wall_best_test_error + args.wall_max_early_stopping < perf_counter():
            save = save_outputs = stop = True
        if len(otr) == len(xtr):
            mind = (args.alpha * otr * ytr).min().item()
            if mind > margin:
                margin += 0.5
                save = save_outputs = True
            if mind > args.stop_margin:
                save = save_outputs = stop = True
        if (args.ptr - (args.alpha * otr * ytr < args.stop_margin).long().sum().item()) / args.ptr > args.stop_frac:
            save = save_outputs = stop = True

        if not save:
            continue

        if len(otr) < len(xtr):
            with torch.no_grad():
                otr = f(xtr) - otr0

            mind = (args.alpha * otr * ytr).min().item()
            if mind > margin:
                margin += 0.5
                save = save_outputs = True
            if mind > args.stop_margin:
                save = save_outputs = stop = True

        with torch.no_grad():
            ote = f(xte) - ote0

        test_err = (ote * yte <= 0).double().mean().item()
        if test_err < best_test_error:
            if tmp_outputs_index != -1:
                dynamics[tmp_outputs_index]['train']['outputs'] = None
                dynamics[tmp_outputs_index]['train']['labels'] = None
                dynamics[tmp_outputs_index]['test']['outputs'] = None
                dynamics[tmp_outputs_index]['test']['labels'] = None

            best_test_error = test_err
            wall_best_test_error = perf_counter()
            if not save_outputs:
                tmp_outputs_index = len(dynamics)
                save_outputs = True

        state['grad_norm'] = grad.norm().item()
        state['wall'] = perf_counter() - wall
        state['norm'] = sum(p.norm().pow(2) for p in f.parameters()).sqrt().item()
        state['dnorm'] = sum((p0 - p).norm().pow(2) for p0, p in zip(f0.parameters(), f.parameters())).sqrt().item()

        if args.arch == 'fc':
            def getw(f, i):
                return torch.cat(list(getattr(f.f, "W{}".format(i))))
            state['wnorm'] = [getw(f, i).norm().item() for i in range(f.f.L + 1)]
            state['dwnorm'] = [(getw(f, i) - getw(f0, i)).norm().item() for i in range(f.f.L + 1)]
            W = [getw(f, i) for i in range(2)]
            W0 = [getw(f0, i) for i in range(2)]
            if args.save_weights:
                assert args.L == 1
                state['w'] = [W[0][:, j].pow(2).mean().sqrt().item() for j in range(args.d)]
                state['dw'] = [(W[0][:, j] - W0[0][:, j]).pow(2).mean().sqrt().item() for j in range(args.d)]
                state['beta'] = W[1].pow(2).mean().sqrt().item()
                state['dbeta'] = (W[1] - W0[1]).pow(2).mean().sqrt().item()
                if args.bias:
                    B = getattr(f.f, "B0")
                    B0 = getattr(f0.f, "B0")
                    state['b'] = B.pow(2).mean().sqrt().item()
                    state['db'] = (B - B0).pow(2).mean().sqrt().item()
            if stop or args.save_attractors > 0:
                sign_on_xtr = (xtr @ (W[0].t() / xtr.size(1) ** 0.5) + B).sign()
                unique_attractors = np.unique(sign_on_xtr.detach().cpu().numpy(), axis=1, return_counts=True)[1]
                state['attractors _number'] = unique_attractors.shape[0]
                if args.save_attractors > 1:
                    state['attractors_degeneracy'] = unique_attractors
                    
        if stop or args.save_state == 1:
            state['state'] = copy.deepcopy(f.state_dict())
        else:
            state['state'] = None

        if args.fa != 'backprop':
            state['weight_fa'] = [l.weight_fa for l in f.f.linear]

        state['train'] = {
            'loss': loss_func(args, otr, ytr).mean().item(),
            'aloss': args.alpha * loss_func(args, otr, ytr).mean().item(),
            'err': (otr * ytr <= 0).double().mean().item(),
            'nd': (args.alpha * otr * ytr < args.loss_margin).long().sum().item(),
            'mind': (args.alpha * otr * ytr).min().item(),
            'maxd': (args.alpha * otr * ytr).max().item(),
            'dfnorm': otr.pow(2).mean().sqrt(),
            'fnorm': (otr + otr0).pow(2).mean().sqrt(),
            'outputs': otr if save_outputs else None,
            'labels': ytr if save_outputs else None,
        }
        state['test'] = {
            'loss': loss_func(args, ote, yte).mean().item(),
            'aloss': args.alpha * loss_func(args, ote, yte).mean().item(),
            'err': test_err,
            'nd': (args.alpha * ote * yte < args.loss_margin).long().sum().item(),
            'mind': (args.alpha * ote * yte).min().item(),
            'maxd': (args.alpha * ote * yte).max().item(),
            'dfnorm': ote.pow(2).mean().sqrt(),
            'fnorm': (ote + ote0).pow(2).mean().sqrt(),
            'outputs': ote if save_outputs else None,
            'labels': yte if save_outputs else None,
        }
        print(
            (
                "[i={d[step]:d} t={d[t]:.2e} wall={d[wall]:.0f}] " + \
                "[dt={d[dt]:.1e} dgrad={d[dgrad]:.1e} dout={d[dout]:.1e}] " + \
                "[train aL={d[train][aloss]:.2e} err={d[train][err]:.2f} " + \
                "nd={d[train][nd]}/{p} mind={d[train][mind]:.3f}] " + \
                "[test aL={d[test][aloss]:.2e} err={d[test][err]:.2f}]"
            ).format(d=state, p=len(ytr)),
            flush=True
        )

        if (args.ptr - state["train"]["nd"]) / args.ptr > args.stop_frac:
            stop = True

        if stop:
            state['test']['outputs'] = ote
            state['test']['labels'] = yte
            state['state'] = copy.deepcopy(f.state_dict())

        dynamics.append(state)

        out = {
            'dynamics': dynamics,
        }

        yield f, out
        if stop:
            break


def run_exp(args, f0, xtr, ytr, xtk, ytk, xte, yte):
    run = {
        'args': args,
        'N': sum(p.numel() for p in f0.parameters()),
        'finished': False,
    }
    wall = perf_counter()

    if args.init_features_ptr == 1:
        parameters = [p for n, p in f0.named_parameters() if 'W{}'.format(args.L) in n or 'classifier' in n]
        assert parameters
        kernels = compute_kernels(f0, xtr, xte[:len(xtk)], parameters)
        for out in run_kernel('init_features_ptr', args, *kernels, xtr, ytr, xte[:len(xtk)], yte[:len(xtk)]):
            run['init_features_ptr'] = out

            if perf_counter() - wall > 120:
                wall = perf_counter()
                yield run
        del kernels

    if args.delta_kernel == 1 or args.init_kernel == 1:
        init_kernel = compute_kernels(f0, xtk, xte[:len(xtk)])

    if args.init_kernel == 1:
        for out in run_kernel('init_kernel', args, *init_kernel, xtk, ytk, xte[:len(xtk)], yte[:len(xtk)]):
            run['init_kernel'] = out

    if args.init_kernel_ptr == 1:
        init_kernel_ptr = compute_kernels(f0, xtr, xte[:len(xtk)])
        for out in run_kernel('init_kernel_ptr', args, *init_kernel_ptr, xtr, ytr, xte[:len(xtk)], yte[:len(xtk)]):
            run['init_kernel_ptr'] = out
        del init_kernel_ptr

    if args.delta_kernel == 1:
        init_kernel = (init_kernel[0].cpu(), init_kernel[2].cpu())
    elif args.init_kernel == 1:
        del init_kernel

    if args.regular == 1:
        if args.running_kernel:
            it = iter(args.running_kernel)
            al = next(it)
        else:
            al = -1
        for f, out in run_regular(args, f0, xtr, ytr, xte, yte):
            run['regular'] = out
            if out['dynamics'][-1]['train']['aloss'] < al * out['dynamics'][0]['train']['aloss']:
                if args.init_kernel_ptr == 1:
                    assert len(xtk) >= len(xtr)
                    running_kernel = compute_kernels(f, xtk[:len(xtr)], xte[:len(xtk)])
                    for kout in run_kernel('kernel_ptr {}'.format(al), args, *running_kernel, xtk[:len(xtr)], ytk[:len(xtr)], xte[:len(xtk)], yte[:len(xtk)]):
                        out['dynamics'][-1]['kernel_ptr'] = kout
                    del running_kernel
                if args.init_features_ptr == 1:
                    parameters = [p for n, p in f.named_parameters() if 'W{}'.format(args.L) in n or 'classifier' in n]
                    assert parameters
                    assert len(xtk) >= len(xtr)
                    running_kernel = compute_kernels(f, xtk[:len(xtr)], xte[:len(xtk)], parameters)
                    for kout in run_kernel('features_ptr {}'.format(al), args, *running_kernel, xtk[:len(xtr)], ytk[:len(xtr)], xte[:len(xtk)], yte[:len(xtk)]):
                        out['dynamics'][-1]['features_ptr'] = kout
                    del running_kernel

                # out['dynamics'][-1]['state'] = copy.deepcopy(f.state_dict())

                try:
                    al = next(it)
                except:
                    al = 0

            if perf_counter() - wall > 120:
                wall = perf_counter()
                yield run
        yield run

        if args.delta_kernel == 1 or args.final_kernel == 1:
            final_kernel = compute_kernels(f, xtk, xte[:len(xtk)])
            if args.final_kernel_ptr == 1:
                ktktk, ktetk, ktete = final_kernel
                ktktk = ktktk[:len(xtr)][:, :len(xtr)]
                ktetk = ktetk[:, :len(xtr)]
                final_kernel_ptr = (ktktk, ktetk, ktete)

        elif args.final_kernel_ptr == 1:
            final_kernel_ptr = compute_kernels(f, xtk[:len(xtr)], xte[:len(xtk)])

        if args.final_kernel == 1:
            for out in run_kernel('final_kernel', args, *final_kernel, xtk, ytk, xte[:len(xtk)], yte[:len(xtk)]):
                run['final_kernel'] = out

                if perf_counter() - wall > 120:
                    wall = perf_counter()
                    yield run
            if args.delta_kernel == 0:
                del final_kernel

        if args.final_kernel_ptr == 1:
            assert len(xtk) >= len(xtr)
            for out in run_kernel('final_kernel_ptr', args, *final_kernel_ptr, xtk[:len(xtr)], ytk[:len(xtr)], xte[:len(xtk)], yte[:len(xtk)]):
                run['final_kernel_ptr'] = out

                if perf_counter() - wall > 120:
                    wall = perf_counter()
                    yield run
            del final_kernel_ptr

        if args.delta_kernel == 1:
            final_kernel = (final_kernel[0].cpu(), final_kernel[2].cpu())
            run['delta_kernel'] = {
                'train': (init_kernel[0] - final_kernel[0]).norm().item(),
                'test': (init_kernel[1] - final_kernel[1]).norm().item(),
            }
            del init_kernel, final_kernel

        if args.stretch_kernel == 1:
            assert args.save_weights
            lam = [x["w"][0] / torch.tensor(x["w"][1:]).float().mean() for x in run['regular']["dynamics"]]
            frac = [(args.ptr - x["train"]["nd"]) / args.ptr for x in run['regular']["dynamics"]]
            for _lam, _frac in zip(lam, frac):
                if _frac > 0.1:
                    lam_star = _lam
                    break
            _xtr = xtr.clone()
            _xte = xte.clone()
            _xtr[:, 1:] = xtr[:, 1:] / lam_star
            _xte[:, 1:] = xte[:, 1:] / lam_star
            stretch_kernel = compute_kernels(f0, _xtr, _xte)
            for out in run_kernel('stretch_kernel', args, *stretch_kernel, _xtr, ytr, _xte, yte):
                run['stretch_kernel'] = out

                if perf_counter() - wall > 120:
                    wall = perf_counter()
                    yield run
            del stretch_kernel

        if args.final_features == 1:
            parameters = [p for n, p in f.named_parameters() if 'W{}'.format(args.L) in n or 'classifier' in n]
            assert parameters
            kernels = compute_kernels(f, xtk, xte[:len(xtk)], parameters)
            for out in run_kernel('final_features', args, *kernels, xtk, ytk, xte[:len(xtk)], yte[:len(xtk)]):
                run['final_features'] = out

                if perf_counter() - wall > 120:
                    wall = perf_counter()
                    yield run
            del kernels

        if args.final_features_ptr == 1:
            parameters = [p for n, p in f.named_parameters() if 'W{}'.format(args.L) in n or 'classifier' in n]
            assert parameters
            assert len(xtk) >= len(xtr)
            kernels = compute_kernels(f, xtk[:len(xtr)], xte[:len(xtk)], parameters)
            for out in run_kernel('final_features_ptr', args, *kernels, xtk[:len(xtr)], ytk[:len(xtr)], xte[:len(xtk)], yte[:len(xtk)]):
                run['final_features_ptr'] = out

                if perf_counter() - wall > 120:
                    wall = perf_counter()
                    yield run
            del kernels

        if args.final_headless == 1:
            parameters = [p for n, p in f.named_parameters() if not 'f.W0.' in n and not 'f.conv_stem.w' in n]
            assert len(xtk) >= len(xtr)
            kernels = compute_kernels(f, xtk, xte[:len(xtk)], parameters)
            for out in run_kernel('final_headless', args, *kernels, xtk, ytk, xte[:len(xtk)], yte[:len(xtk)]):
                run['final_headless'] = out

                if perf_counter() - wall > 120:
                    wall = perf_counter()
                    yield run
            del kernels

        if args.final_headless_ptr == 1:
            parameters = [p for n, p in f.named_parameters() if not 'f.W0.' in n and not 'f.conv_stem.w' in n]
            assert len(xtk) >= len(xtr)
            kernels = compute_kernels(f, xtk[:len(xtr)], xte[:len(xtk)], parameters)
            for out in run_kernel('final_headless_ptr', args, *kernels, xtk[:len(xtr)], ytk[:len(xtr)], xte[:len(xtk)], yte[:len(xtk)]):
                run['final_headless_ptr'] = out

                if perf_counter() - wall > 120:
                    wall = perf_counter()
                    yield run
            del kernels

    run['finished'] = True
    yield run


def init(args):
    torch.backends.cudnn.benchmark = True
    if args.dtype == 'float64':
        torch.set_default_dtype(torch.float64)
    if args.dtype == 'float32':
        torch.set_default_dtype(torch.float32)

    [(xte, yte, ite), (xtk, ytk, itk), (xtr, ytr, itr)] = get_binary_dataset(
        args.dataset,
        (args.pte, args.ptk, args.ptr),
        (args.seed_testset + args.pte, args.seed_kernelset + args.ptk, args.seed_trainset + args.ptr),
        args.d,
        (args.data_param1, args.data_param2),
        args.device,
        torch.get_default_dtype()
    )

    torch.manual_seed(0)

    if args.act == 'relu':
        _act = torch.relu
    elif args.act == 'tanh':
        _act = torch.tanh
    elif args.act == 'softplus':
        _act = torch.nn.functional.softplus
    elif args.act == 'swish':
        _act = swish
    else:
        raise ValueError('act not specified')

    def __act(x):
        b = args.act_beta
        return _act(b * x) / b
    factor = __act(torch.randn(100000, dtype=torch.float64)).pow(2).mean().rsqrt().item()

    def act(x):
        return __act(x) * factor

    _d = abs(act(torch.randn(100000, dtype=torch.float64)).pow(2).mean().rsqrt().item() - 1)
    assert _d < 1e-2, _d

    torch.manual_seed(args.seed_init + hash(args.alpha) + args.ptr)

    if args.arch == 'fc':
        assert args.L is not None
        xtr = xtr.flatten(1)
        xtk = xtk.flatten(1)
        xte = xte.flatten(1)
        f = FC(xtr.size(1), args.h, 1, args.L, act, args.bias, args.last_bias, args.var_bias)

    elif args.arch == 'fc_fa':
        assert args.L is not None
        xtr = xtr.flatten(1)
        xtk = xtk.flatten(1)
        xte = xte.flatten(1)
        # f = FC_fa(xtr.size(1), args.h, 1, args.L, act, args.bias, args.last_bias, args.fa)
        if args.fa == 'backprop':
            f = LinearNetwork(xtr.size(1), args.h, act, args.bias)
        else:
            f = LinearFANetwork(xtr.size(1), args.h, act, args.bias, args.fa)

    elif args.arch == 'cv':
        assert args.bias == 0
        f = CV(xtr.size(1), args.h, L1=args.cv_L1, L2=args.cv_L2, act=act, h_base=args.cv_h_base,
               fsz=args.cv_fsz, pad=args.cv_pad, stride_first=args.cv_stride_first)
    elif args.arch == 'resnet':
        assert args.bias == 0
        f = Wide_ResNet(xtr.size(1), 28, args.h, act, 1, args.mix_angle)
    elif args.arch == 'mnas':
        assert args.act == 'swish'
        f = MnasNetLike(xtr.size(1), args.h, 1, args.cv_L1, args.cv_L2, dim=xtr.dim() - 2)
    elif args.arch == 'mnist':
        assert args.dataset == 'mnist'
        f = MNISTNet(xtr.size(1), args.h, 1, act)
    elif args.arch == 'crown_init':
        f = CrownInit(args.d, args.h, act, args.bias)
    elif args.arch == 'fixed_weights':
        f = FixedWeights(args.d, args.h, act, args.bias)
    elif args.arch == 'fixed_angles':
        f = FixedAngles(args.d, args.h, act, args.bias)
    elif args.arch == 'fixed_betas':
        f = FixedBetas(args.d, args.h, act, args.bias)
    elif args.arch == 'fixed_norm':
        f = FixedNorm(args.d, args.h, act, args.bias)
    elif args.arch == 'mf_angles':
        f = MFAngles(args.d, args.h, act, args.bias)
    elif args.arch == 'conv1d':
        f = Conv1d(args.d, args.h, act, args.bias)
    else:
        raise ValueError('arch not specified')

    f = SplitEval(f, args.chunk)
    f = f.to(args.device)

    return f, xtr, ytr, itr, xtk, ytk, itk, xte, yte, ite


def execute(args):
    f, xtr, ytr, itr, xtk, ytk, itk, xte, yte, ite = init(args)

    torch.manual_seed(0)
    for run in run_exp(args, f, xtr, ytr, xtk, ytk, xte, yte):
        run['dataset'] = {
            'test': ite,
            'kernel': itk,
            'train': itr,
        }
        yield run


def main():
    git = {
        'log': subprocess.getoutput('git log --format="%H" -n 1 -z'),
        'status': subprocess.getoutput('git status -z'),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--dtype", type=str, default='float64')

    parser.add_argument("--seed_init", type=int, default=0)
    parser.add_argument("--seed_testset", type=int, default=0, help="determines the testset, will affect the kernelset and trainset as well")
    parser.add_argument("--seed_kernelset", type=int, default=0, help="determines the kernelset, will affect the trainset as well")
    parser.add_argument("--seed_trainset", type=int, default=0, help="determines the trainset")

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ptr", type=int, required=True)
    parser.add_argument("--ptk", type=int, default=0)
    parser.add_argument("--pte", type=int)
    parser.add_argument("--d", type=int)
    parser.add_argument("--whitening", type=int, default=1)
    parser.add_argument("--data_param1", type=int,
                        help="Sphere dimension if dataset = Cylinder."
                        "Total number of cells, if dataset = sphere_grid. "
                        "n0 if dataset = signal_1d.")
    parser.add_argument("--data_param2", type=float,
                        help="Stretching factor for non-spherical dimensions if dataset = cylinder."
                        "Number of bins in theta, if dataset = sphere_grid.")

    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--act", type=str, required=True)
    parser.add_argument("--act_beta", type=float, default=1.0)
    parser.add_argument("--bias", type=float, default=0)
    parser.add_argument("--last_bias", type=float, default=0)
    parser.add_argument("--var_bias", type=float, default=1)
    parser.add_argument("--L", type=int)
    parser.add_argument("--h", type=int, required=True)
    parser.add_argument("--mix_angle", type=float, default=45)
    parser.add_argument("--cv_L1", type=int, default=2)
    parser.add_argument("--cv_L2", type=int, default=2)
    parser.add_argument("--cv_h_base", type=float, default=1)
    parser.add_argument("--cv_fsz", type=int, default=5)
    parser.add_argument("--cv_pad", type=int, default=1)
    parser.add_argument("--cv_stride_first", type=int, default=1)

    parser.add_argument("--fa", type=str, default="backprop")

    parser.add_argument("--init_kernel", type=int, default=0)
    parser.add_argument("--init_kernel_ptr", type=int, default=0)
    parser.add_argument("--regular", type=int, default=1)
    parser.add_argument('--running_kernel', nargs='+', type=float)
    parser.add_argument("--final_kernel", type=int, default=0)
    parser.add_argument("--final_kernel_ptr", type=int, default=0)
    parser.add_argument("--final_headless", type=int, default=0)
    parser.add_argument("--final_headless_ptr", type=int, default=0)
    parser.add_argument("--init_features_ptr", type=int, default=0)
    parser.add_argument("--final_features", type=int, default=0)
    parser.add_argument("--final_features_ptr", type=int, default=0)
    parser.add_argument("--train_kernel", type=int, default=1)
    parser.add_argument("--store_kernel", type=int, default=0)
    parser.add_argument("--delta_kernel", type=int, default=0)
    parser.add_argument("--stretch_kernel", type=int, default=0)

    parser.add_argument("--save_outputs", type=int, default=0)
    parser.add_argument("--save_state", type=int, default=0)
    parser.add_argument("--save_weights", type=int, default=0)
    parser.add_argument("--save_attractors", type=int, default=0,
                        help = "If 1, saves the number of attractors at each step."
                               " If 2, saves also the degeneracy of each."
                       )
    
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--loss_over_alpha_power", type=int, default=1, help="Divide the loss by alpha ^ *")
    parser.add_argument("--f0", type=int, default=1)

    parser.add_argument("--tau_over_h", type=float, default=0.0)
    parser.add_argument("--tau_over_h_kernel", type=float)
    parser.add_argument("--tau_alpha_crit", type=float)

    parser.add_argument("--max_wall", type=float, required=True)
    parser.add_argument("--max_wall_kernel", type=float)
    parser.add_argument("--wall_max_early_stopping", type=float)
    parser.add_argument("--chunk", type=int)
    parser.add_argument("--max_dgrad", type=float, default=1e-4)
    parser.add_argument("--max_dout", type=float, default=1e-1)

    parser.add_argument("--loss", type=str, default="softhinge")
    parser.add_argument("--loss_beta", type=float, default=20.0)
    parser.add_argument("--l2_decay", type=float, default=0.)
    parser.add_argument("--loss_margin", type=float, default=1.0)
    parser.add_argument("--stop_margin", type=float, default=1.0)
    parser.add_argument("--stop_frac", type=float, default=1.0)
    parser.add_argument("--bs", type=int)

    parser.add_argument("--ckpt_step", type=int, default=100)
    parser.add_argument("--ckpt_tau", type=float, default=1e4)

    parser.add_argument("--pickle", type=str, required=True)
    args = parser.parse_args()

    if args.pte is None:
        args.pte = args.ptr

    if args.chunk is None:
        args.chunk = max(args.ptr, args.pte, args.ptk)

    if args.max_wall_kernel is None:
        args.max_wall_kernel = args.max_wall

    if args.tau_over_h_kernel is None:
        args.tau_over_h_kernel = args.tau_over_h

    if args.seed_init == -1:
        args.seed_init = args.seed_trainset

    torch.save(args, args.pickle)
    saved = False
    try:
        for res in execute(args):
            res['git'] = git
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
