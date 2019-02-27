# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import numpy as np
import scipy
from scipy.io import wavfile as siowav
from scipy import integrate, interpolate
from scipy import optimize
import matplotlib
# matplotlib.use('Agg')
from matplotlib.mlab import find
from matplotlib import pyplot as plt
from ode_model import vdp_coupled, vdp_jacobian
import pdb
for path in [
        'utils'
]:
    sys.path.append(
        os.path.join(
            os.path.dirname(os.path.dirname(
                os.path.realpath(__file__))),
            path))
from sigproc import framesig


# Data
wav_root = '../../data/FEMH_Data/processed/resample_8k'
wav_list = './FEMH_data_8k.lst'

# Initial settings
y0 = [0, 0.1, 0, 0.1]
params0 = (0.25, 0.32, 0.30)


def envelope(x):
    '''
    Compute the analytic signal and the envelope.

    Parameters
    ----------
    x: np.array[float], shape (N,)
        The input signal.

    Returns
    -------
    xa: np.array[complex]
        The analytic signal.
    envelope: np.array[float]
        The envelope.
    '''
    # Analytic signal
    xa = scipy.signal.hilbert(x)
    envelope = np.abs(xa)
    return xa, envelope


def cross_correlation(x, y):
    '''
    Compute the cross-correlation of the two signal x and y.

    Parameters
    ----------
    x: np.array[float], shape (N,)
    y: np.array[float], shape (M,)

    Returns
    -------
    Rxy: np.array[complex]
    '''
    X = scipy.fft.fft(x)
    Y = scipy.fft.fft(y)
    X_ = np.conj(X)
    R_ = X_ * Y
    Rxy = scipy.fft.ifft(R_)
    Rxy = scipy.signal.correlate(x, y)
    return Rxy


def estimate_pitch(params, t, pitch_true, fs):
    '''
    Estimate the pitch via the ODE system.

    Parameters
    ----------
    params: np.array[float]
        Parameters to ODE system.
    t: np.array[float]
        Time sequence.
    pitch_true: np.array[float]
        True pitch.
    fs: int
        Sample rate.

    Returns
    -------
    pitch: List[float]
        Estimated pitch.
    '''
    # pdb.set_trace()
    # print(params)
    r = integrate.odeint(vdp_coupled, y0, t,
                         args=tuple(params), Dfun=vdp_jacobian)

    flow = r[:, 0] + r[:, 2]

    # fig = plt.figure(figsize=(6, 6))
    # ax1 = fig.add_subplot(211)
    # ax1.plot(flow[:10000], c='b', ls='-', lw=1.5)
    # ax1.set_xlabel('t', fontsize=12)
    # ax1.set_ylabel(r'$x_1 + x_2$', fontsize=12)
    # ax1 = fig.add_subplot(212)
    # ax1.plot(r[:10000, 0], c='b', ls='-', lw=1.5)
    # ax1.plot(r[:10000, 2], c='r', ls='--', lw=1.5)
    # ax1.set_xlabel('t', fontsize=12)
    # ax1.set_ylabel(r'$x$', fontsize=12)
    # ax1.legend([r'$x_1$', r'$x_2$'], loc='best')
    # ax1.axis('auto')
    # plt.tight_layout()
    # plt.savefig('glottal_area_flow_normal.pdf')

    flow = flow - np.mean(flow)

    pitch = []
    for i in range(len(pitch_true)):
        flow_seg = flow[i * 1000: (i + 1) * 1000]
        # Find frequency via zero crossing
        indices, = np.nonzero(np.ravel(
            (flow_seg[1:] >= 0) & (flow_seg[:-1] < 0)))
        crossings = [j - flow_seg[j] / (flow_seg[j + 1] - flow_seg[j])
                     for j in indices]  # linear interpolate
        cycle_len = np.mean(np.diff(crossings))  # avg len per cycle
        p = fs / float(cycle_len)  # cycle/s
        # p = p * 10  # TODO: decide time unit
        pitch.append(p)
        # print(cycle_len, p)

    return pitch


# Least-squares residual function
def residual(params, y_true, tmax, tlen, fs):
    t = np.linspace(0, tmax, tlen)
    yh = estimate_pitch(params, t, y_true, fs)
    # print(np.sum(np.abs(y_true - np.array(yh))))
    return y_true - np.array(yh)


def fit():
    alphas = []
    betas = []
    deltas = []

    wavfiles = [l.rstrip('\n').split()[0] for l in open(wav_list)][150:]

    FNULL = open(os.devnull, 'w')

    for f in wavfiles:
        # Read wav
        sample_rate, samples = siowav.read(os.path.join(wav_root, f))
        assert sample_rate == 8000, "{}: incompatible sample rate"\
            " need 8000 but got {}".format(f, sample_rate)

        # fig = plt.figure(figsize=(6, 6))
        # ax1 = fig.add_subplot(111)
        # ax1.plot(samples, c='b', ls='-', lw=2)
        # ax1.set_xlabel('t', fontsize=12)
        # ax1.set_ylabel('amp', fontsize=12)
        # ax1.axis('auto')
        # plt.tight_layout()
        # plt.savefig('signal_{}.pdf'.format(f.replace('/', '_').rstrip('.wav')))

        # Extract pitch
        subprocess.run(["./histopitch -in {} -out tmp.pit -srate {}".
                        format(os.path.join(wav_root, f),
                               sample_rate)
                        ],
                       shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
        pitch = np.loadtxt("tmp.pit")

        # Estimate ODE params
        tmax = 2000 * len(samples) / float(sample_rate)
        tlen = len(pitch) * 1000
        params_best = optimize.leastsq(residual, params0,
                                       args=(pitch, tmax, tlen, sample_rate))

        alphas.append(params_best[0][0])
        betas.append(params_best[0][1])
        deltas.append(params_best[0][2])

        print(f)
        print("parameter values are ", params_best)

        #  t = np.linspace(0, tmax, tlen)
        #  pitch_est = estimate_pitch(params_best, t, pitch, sample_rate)
        #  fig = plt.figure(figsize=(12, 6))
        #  ax1 = fig.add_subplot(121)
        #  ax1.plot(pitch, c='b', ls='-', lw=2)
        #  ax1.set_xlabel('t', fontsize=12)
        #  ax1.set_ylabel('pitch', fontsize=12)
        #  ax1.set_ylim(bottom=0)
        #  ax1.axis('auto')
        #  ax1 = fig.add_subplot(122)
        #  ax1.plot(pitch_est, c='b', ls='-', lw=2)
        #  ax1.set_xlabel('t', fontsize=12)
        #  ax1.set_ylabel('pitch', fontsize=12)
        #  ax1.set_ylim(bottom=0)
        #  ax1.axis('auto')
        #  plt.tight_layout()
        #  plt.savefig('pitch_{}.pdf'.format(f.replace('/', '_').rstrip('.wav')))

    # fig = plt.figure(figsize=(6, 6))
    # ax1 = fig.add_subplot(111)
    # ax1.scatter(np.abs(deltas), alphas, s=2, c='b', marker='.')
    # ax1.scatter(np.abs(deltas), betas, s=2, c='r', marker='o')
    # ax1.set_xlabel(r'$| \Delta |$', fontsize=12)
    # ax1.set_ylabel(r'$\alpha$', fontsize=12)
    # # ax1.set_xlim(0, 2)
    # # ax1.set_ylim(0, 1.5)
    # ax1.axis('auto')
    # plt.tight_layout()
    # plt.savefig('bifurcation_plot.pdf')


fit()


#  r = ode(vdp_coupled, vdp_jacobian)
#  r.set_integrator('lsoda',
                 #  with_jacobian=True,
                 #  ixpr=1)
#  r.set_f_params(0.8, 0.32, 1)
#  r.set_jac_params(0.8, 0.32, 1)

#  r.set_initial_value(y0, t0)

#  while r.successful() and r.t < tmax:
    #  r.integrate(r.t + dt)
    #  sol.append([r.t, *list(r.y)])

#  sol = np.array(sol)  # dim: t, xr, dxr, xl, dxl

#  plt.figure()
#  plt.plot(sol[:1000, 0], sol[:1000, 1], 'b.-')
#  plt.plot(sol[:1000, 0], sol[:1000, 3], 'r.-')
#  plt.xlabel('t')
#  plt.ylabel('x')
#  plt.show()

#  plt.figure()
#  plt.plot(sol[:, 2], sol[:, 4], 'b.-')
#  plt.xlabel('du')
#  plt.ylabel('dv')
#  plt.show()
