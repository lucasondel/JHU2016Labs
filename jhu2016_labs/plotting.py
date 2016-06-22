
"""Plotting functions."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def initFigure(x_min, x_max, y_min, y_max):
    fig = plt.figure()
    ax = plt.axes(xlim=(x_min, x_max), ylim=(y_min, y_max))
    ax.grid()
    return fig, ax


def plotGaussian(gaussian, fig=None, ax=None, x_min=-10, x_max=10,
                 y_min=0., y_max=1., label=None, color='k'):
    if fig is None or ax is None:
        fig, ax = initFigure(x_min, x_max, y_min, y_max)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$p(x)$')
    x = np.linspace(x_min, x_max, 1000)
    pdf = gaussian.pdf(x)
    ax.plot(x, pdf, lw=2, color=color, label=label)

    return fig, ax


def plotGaussianInteractive(gaussian, fig=None, ax=None, x_min=-10,
                            x_max=10, y_min=0., y_max=1., data=None,
                            use_precision=False):
    if fig is None or ax is None:
        fig, ax = initFigure(x_min, x_max, y_min, y_max)
    plt.subplots_adjust(left=0.25, bottom=0.25)
    x = np.linspace(x_min, x_max, 1000)
    y = gaussian.pdf(x)
    line, = ax.plot(x, y, lw=2)


    if data is not None:
        llh = gaussian.logLikelihood(data)
        text = 'log-likelihood: %.2f' % llh
        llh_text = ax.text(0.02, 0.95, text, transform=ax.transAxes)

    axcolor = 'lightgoldenrodyellow'
    axmean = plt.axes([.25, .1, .65, .03], axisbg=axcolor)
    axvar = plt.axes([.25, .15, .65, .03], axisbg=axcolor)

    smean = Slider(axmean, '$\mu$', -5., 5., valinit=0.)
    if use_precision:
        label = '$\lambda$'
        v_min = 0.
        v_max = 10.
    else:
        label = '$\sigma^2$'
        v_min = 0.
        v_max = 10.
    svar = Slider(axvar, label, v_min, v_max, valinit=1.)

    def update(val):
        gaussian.mean = smean.val
        if use_precision:
            gaussian.var = 1/svar.val
        else:
            gaussian.var = svar.val
        y = gaussian.pdf(x)
        line.set_data((x, y))
        if data is not None:
            llh = gaussian.logLikelihood(data)
            text = 'log-likelihood: %.2f' % llh
            llh_text.set_text(text)
        fig.canvas.draw_idle()

    smean.on_changed(update)
    svar.on_changed(update)


def plotGMM(gmm, fig=None, ax=None, show_components=False, x_min=-10,
            x_max=10, y_min=0., y_max=1., label=None, color='k', lw=2):
    if fig is None or ax is None:
        fig, ax = initFigure(x_min, x_max, y_min, y_max)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$p(x)$')
    x = np.linspace(x_min, x_max, 1000)
    pdfs = gmm.pdf(x, sum_pdf=False)
    pdf = pdfs.sum(axis=1)
    ax.plot(x, pdf, lw=lw, color=color, label=label)

    if show_components:
        for i in range(pdfs.shape[1]):
            ax.plot(x, pdfs[:, i], '--')
    return fig, ax

def plotNormalGamma(ng, x_min=-10, x_max=10, y_min=0.001, y_max=2):
    fig = plt.figure()
    ax = plt.axes()
    x = np.linspace(x_min, x_max, 1000)
    y = np.linspace(y_min, y_max, 1000)
    X, Y = np.meshgrid(x, y)

    old_settings = np.seterr(divide='ignore')
    Z = ng.pdf(X, Y)
    np.seterr(**old_settings)
    ax.set_xticks([x_min, (x_min+x_max)/2, x_max])
    ax.set_yticks([y_min, (y_min+y_max)/2, y_max])
    ax.set_xlabel('$\mu$')
    ax.set_ylabel('$\lambda$')
    ax.imshow(Z, origin='lower', cmap='Greys',
              extent=[x_min, x_max, 0., y_max])

