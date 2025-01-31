__all__ = [
    "GwcsBoundingBoxWarning",
    "GwcsFrameExistsError",
    "NoConvergence",
]


class NoConvergence(Exception):
    """
    An error class used to report non-convergence and/or divergence
    of numerical methods. It is used to report errors in the
    iterative solution used by
    the :py:meth:`~astropy.wcs.WCS.all_world2pix`.

    Attributes
    ----------

    best_solution : `numpy.ndarray`
        Best solution achieved by the numerical method.

    accuracy : `numpy.ndarray`
        Estimate of the accuracy of the ``best_solution``.

    niter : `int`
        Number of iterations performed by the numerical method
        to compute ``best_solution``.

    divergent : None, `numpy.ndarray`
        Indices of the points in ``best_solution`` array
        for which the solution appears to be divergent. If the
        solution does not diverge, ``divergent`` will be set to `None`.

    slow_conv : None, `numpy.ndarray`
        Indices of the solutions in ``best_solution`` array
        for which the solution failed to converge within the
        specified maximum number of iterations. If there are no
        non-converging solutions (i.e., if the required accuracy
        has been achieved for all input data points)
        then ``slow_conv`` will be set to `None`.

    """

    def __init__(
        self,
        *args,
        best_solution=None,
        accuracy=None,
        niter=None,
        divergent=None,
        slow_conv=None,
    ):
        super().__init__(*args)

        self.best_solution = best_solution
        self.accuracy = accuracy
        self.niter = niter
        self.divergent = divergent
        self.slow_conv = slow_conv


class GwcsFrameExistsError(ValueError):
    """
    An error used to report when a frame already exists in a pipeline.
    """


class GwcsBoundingBoxWarning(UserWarning):
    """
    A warning class to report issues with bounding boxes in GWCS.
    """
