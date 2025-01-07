import sys
import warnings

import numpy as np
import numpy.linalg as npla
from astropy.modeling.models import (
    Const1D,
    Identity,
    Mapping,
    Polynomial2D,
)
from scipy import linalg

from gwcs.wcstools import grid_from_bounding_box

__all__ = [
    "fit_2D_poly",
    "fix_transform_inputs",
    "make_sampling_grid",
    "reform_poly_coefficients",
    "store_2D_coefficients",
]


def _poly_fit_lu(xin, yin, xout, yout, degree, coord_pow=None):
    # This function fits 2D polynomials to data by writing the normal system
    # of equations and solving it using LU-decomposition. In theory this
    # should be less stable than the SVD method used by numpy's lstsq or
    # astropy's LinearLSQFitter because the condition of the normal matrix
    # is squared compared to the direct matrix. However, in practice,
    # in our (Mihai Cara) tests of fitting WCS distortions, solving the
    # normal system proved to be significantly more accurate, efficient,
    # and stable than SVD.
    #
    # coord_pow - a dictionary used to store powers of coordinate arrays
    #    of the form x**p * y**q used to build the pseudo-Vandermonde matrix.
    #    This improves efficiency especially when fitting multiple degrees
    #    on the same coordinate grid in _fit_2D_poly by reusing computed
    #    powers.
    powers = [
        (i, j) for i in range(degree + 1) for j in range(degree + 1 - i) if i + j > 0
    ]
    if coord_pow is None:
        coord_pow = {}

    nterms = len(powers)

    flt_type = np.longdouble

    # allocate array for the coefficients of the system of equations (a*x=b):
    a = np.empty((nterms, nterms), dtype=flt_type)
    bx = np.empty(nterms, dtype=flt_type)
    by = np.empty(nterms, dtype=flt_type)

    xout = xout.ravel()
    yout = yout.ravel()

    x = np.asarray(xin.ravel(), dtype=flt_type)
    y = np.asarray(yin.ravel(), dtype=flt_type)

    # pseudo_vander - a reduced Vandermonde matrix for 2D polynomials
    # that has only terms x^i * y^j with powers i, j that satisfy:
    # 0 < i + j <= degree.
    pseudo_vander = np.empty((x.size, nterms), dtype=float)

    def pow2(p, q):
        # computes product of powers of coordinate arrays (x**p) * (y**q)
        # in an efficient way avoiding unnecessary array copying
        # and/or raising to power
        if (p, q) in coord_pow:
            return coord_pow[(p, q)]
        if p == 0:
            arr = y**q if q > 1 else y
        elif q == 0:
            arr = x**p if p > 1 else x
        else:
            xp = x if p == 1 else x**p
            yq = y if q == 1 else y**q
            arr = xp * yq
        coord_pow[(p, q)] = arr
        return arr

    for i in range(nterms):
        pi, qi = powers[i]
        coord_pq = pow2(pi, qi)
        pseudo_vander[:, i] = coord_pq
        bx[i] = np.sum(xout * coord_pq, dtype=flt_type)
        by[i] = np.sum(yout * coord_pq, dtype=flt_type)

        for j in range(i, nterms):
            pj, qj = powers[j]
            coord_pq = pow2(pi + pj, qi + qj)
            a[i, j] = np.sum(coord_pq, dtype=flt_type)
            a[j, i] = a[i, j]

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("error", category=linalg.LinAlgWarning)
        try:
            lu_piv = linalg.lu_factor(a)
            poly_coeff_x = linalg.lu_solve(lu_piv, bx).astype(float)
            poly_coeff_y = linalg.lu_solve(lu_piv, by).astype(float)
        except (ValueError, linalg.LinAlgWarning, np.linalg.LinAlgError) as e:
            msg = f"Failed to fit SIP. Reported error:\n{e.args[0]}"
            raise np.linalg.LinAlgError(msg) from e

    if not np.all(np.isfinite([poly_coeff_x, poly_coeff_y])):
        msg = "Failed to fit SIP. Computed coefficients are not finite."
        raise np.linalg.LinAlgError(msg)

    cond = np.linalg.cond(a.astype(float))

    fitx = np.dot(pseudo_vander, poly_coeff_x)
    fity = np.dot(pseudo_vander, poly_coeff_y)

    dist = np.sqrt((xout - fitx) ** 2 + (yout - fity) ** 2)
    max_resid = dist.max()

    return poly_coeff_x, poly_coeff_y, max_resid, powers, cond


def fit_2D_poly(
    degree,
    max_error,
    plate_scale,
    xin,
    yin,
    xout,
    yout,
    xind,
    yind,
    xoutd,
    youtd,
    verbose=False,
):
    """
    Fit a pair of ordinary 2D polynomials to the supplied transform.

    """
    # The case of one pass with the specified polynomial degree
    if degree is None:
        deglist = list(range(1, 10))
    elif hasattr(degree, "__iter__"):
        deglist = sorted(map(int, degree))
        if deglist[0] < 1 or deglist[-1] > 9:
            msg = "Allowed values for SIP degree are [1...9]"
            raise ValueError(msg)
    else:
        degree = int(degree)
        if degree < 1 or degree > 9:
            msg = "Allowed values for SIP degree are [1...9]"
            raise ValueError(msg)
        deglist = [degree]

    single_degree = len(deglist) == 1

    fit_error = np.inf
    if verbose and not single_degree:
        sys.stdout.write(f"Maximum specified SIP approximation error: {max_error}")
    max_error *= plate_scale

    fit_warning_msg = "Failed to achieve requested SIP approximation accuracy."

    # Fit lowest degree SIP first.
    coord_pow = {}  # hold coordinate arrays powers for optimization purpose
    for deg in deglist:
        try:
            cfx_i, cfy_i, fit_error_i, powers_i, cond = _poly_fit_lu(
                xin, yin, xout, yout, degree=deg, coord_pow=coord_pow
            )
            if verbose and not single_degree:
                sys.stdout.write(
                    f"   - SIP degree: {deg}. "
                    f"Maximum residual: {fit_error_i / plate_scale:.5g}"
                )

        except np.linalg.LinAlgError:
            if single_degree:
                # Nothing to do if failure is for the lowest degree
                raise
            # Keep results from the previous iteration. Discard current fit
            break

        if not np.isfinite(cond):
            # Ill-conditioned system
            if single_degree:
                warnings.warn("The fit may be poorly conditioned.", stacklevel=2)
                cfx = cfx_i
                cfy = cfy_i
                fit_error = fit_error_i
                powers = powers_i
            break

        if fit_error_i >= fit_error:
            # Accuracy does not improve. Likely ill-conditioned system
            break

        cfx = cfx_i
        cfy = cfy_i
        powers = powers_i

        fit_error = fit_error_i

        if fit_error <= max_error:
            # Requested accuracy has been achieved
            fit_warning_msg = None
            break

        # Continue to the next degree

    fit_poly_x = Polynomial2D(degree=deg, c0_0=0.0)
    fit_poly_y = Polynomial2D(degree=deg, c0_0=0.0)
    for cx, cy, (p, q) in zip(cfx, cfy, powers, strict=False):
        setattr(fit_poly_x, f"c{p:1d}_{q:1d}", cx)
        setattr(fit_poly_y, f"c{p:1d}_{q:1d}", cy)

    if fit_warning_msg:
        warnings.warn(fit_warning_msg, linalg.LinAlgWarning, stacklevel=2)

    if fit_error <= max_error or single_degree:
        # Check to see if double sampling meets error requirement.
        max_resid = _compute_distance_residual(
            xoutd, youtd, fit_poly_x(xind, yind), fit_poly_y(xind, yind)
        )
        if verbose:
            sys.stdout.write(
                "* Maximum residual, double sampled grid: "
                f"{max_resid / plate_scale:.5g}"
            )

        if max_resid > min(5.0 * fit_error, max_error):
            warnings.warn(
                "Double sampling check FAILED: Sampling may be too coarse for "
                "the distortion model being fitted.",
                stacklevel=2,
            )

        # Residuals on the double-dense grid may be better estimates
        # of the accuracy of the fit. So we report the largest of
        # the residuals (on single- and double-sampled grid) as the fit error:
        fit_error = max(max_resid, fit_error)

    if verbose:
        if single_degree:
            sys.stdout.write(f"Maximum residual: {fit_error / plate_scale:.5g}")
        else:
            sys.stdout.write(
                f"* Final SIP degree: {deg}. "
                f"Maximum residual: {fit_error / plate_scale:.5g}"
            )

    return fit_poly_x, fit_poly_y, fit_error / plate_scale


def make_sampling_grid(npoints, bounding_box, crpix):
    step = np.subtract.reduce(bounding_box, axis=1) / (1.0 - npoints)
    crpix = np.asanyarray(crpix)[:, None, None]
    x, y = grid_from_bounding_box(bounding_box, step=step, center=False) - crpix
    return x.flatten(), y.flatten()


def _compute_distance_residual(undist_x, undist_y, fit_poly_x, fit_poly_y):
    """
    Compute the distance residuals and return the rms and maximum values.
    """
    dist = np.sqrt((undist_x - fit_poly_x) ** 2 + (undist_y - fit_poly_y) ** 2)
    return dist.max()


def reform_poly_coefficients(fit_poly_x, fit_poly_y):
    """
    The fit polynomials must be recombined to align with the SIP decomposition

    The result is the f(u,v) and g(u,v) polynomials, and the CD matrix.
    """
    # Extract values for CD matrix and recombining
    c11 = fit_poly_x.c1_0.value
    c12 = fit_poly_x.c0_1.value
    c21 = fit_poly_y.c1_0.value
    c22 = fit_poly_y.c0_1.value
    sip_poly_x = fit_poly_x.copy()
    sip_poly_y = fit_poly_y.copy()
    # Force low order coefficients to be 0 as defined in SIP
    sip_poly_x.c0_0 = 0
    sip_poly_y.c0_0 = 0
    sip_poly_x.c1_0 = 0
    sip_poly_x.c0_1 = 0
    sip_poly_y.c1_0 = 0
    sip_poly_y.c0_1 = 0

    cdmat = ((c11, c12), (c21, c22))
    invcdmat = npla.inv(np.array(cdmat))
    degree = fit_poly_x.degree
    # Now loop through all remaining coefficients
    for i in range(degree + 1):
        for j in range(degree + 1):
            if (i + j > 1) and (i + j < degree + 1):
                old_x = getattr(fit_poly_x, f"c{i}_{j}").value
                old_y = getattr(fit_poly_y, f"c{i}_{j}").value
                newcoeff = np.dot(invcdmat, np.array([[old_x], [old_y]]))
                setattr(sip_poly_x, f"c{i}_{j}", newcoeff[0, 0])
                setattr(sip_poly_y, f"c{i}_{j}", newcoeff[1, 0])

    return cdmat, sip_poly_x, sip_poly_y


def store_2D_coefficients(hdr, poly_model, coeff_prefix, keeplinear=False):
    """
    Write the polynomial model coefficients to the header.
    """
    mindeg = int(not keeplinear)
    degree = poly_model.degree
    for i in range(degree + 1):
        for j in range(degree + 1):
            if (i + j) > mindeg and (i + j < degree + 1):
                hdr[f"{coeff_prefix}_{i}_{j}"] = getattr(poly_model, f"c{i}_{j}").value


def fix_transform_inputs(transform, inputs):
    # This is a workaround to the bug in https://github.com/astropy/astropy/issues/11360
    # Once that bug is fixed, the code below can be replaced with fix_inputs
    if not inputs:
        return transform

    c = None
    mapping = []
    for k in range(transform.n_inputs):
        if k in inputs:
            mapping.append(0)
        else:
            # this assumes that n_inputs > 0 and that axis 0 always exist
            c = 0 if c is None else (c + 1)
            mapping.append(c)

    in_selector = Mapping(mapping, n_inputs=transform.n_inputs - len(inputs))

    input_fixer = Const1D(inputs[0]) if 0 in inputs else Identity(1)
    for k in range(1, transform.n_inputs):
        input_fixer &= Const1D(inputs[k]) if k in inputs else Identity(1)

    return in_selector | input_fixer | transform
