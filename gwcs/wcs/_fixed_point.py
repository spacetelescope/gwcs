import numpy as np
from scipy import optimize

from ._exception import NoConvergence

__all__ = ["vectorized_fixed_point"]


def vectorized_fixed_point(
    transform,
    pix0,
    world,
    tolerance,
    maxiter,
    adaptive,
    detect_divergence,
    quiet,
    with_bounding_box,
    fill_value,
):
    # ############################################################
    # #            INITIALIZE ITERATIVE PROCESS:                ##
    # ############################################################

    # make a copy of the initial approximation
    pix0 = np.atleast_2d(np.array(pix0))  # 0-order solution
    pix = np.array(pix0)

    world0 = np.atleast_2d(np.array(world))
    world = np.array(world0)

    # estimate pixel scale using approximate algorithm
    # from https://trs.jpl.nasa.gov/handle/2014/40409
    try:
        bounding_box = transform.bounding_box
    except NotImplementedError:
        bounding_box = None
    else:
        bounding_box = bounding_box.bounding_box(order="F")

    if bounding_box is None:
        crpix = np.ones(transform.n_inputs)
    else:
        crpix = np.mean(bounding_box, axis=-1)

    l1, phi1 = np.deg2rad(transform(*(crpix - 0.5)))
    l2, phi2 = np.deg2rad(transform(*(crpix + [-0.5, 0.5])))  # noqa: RUF005
    l3, phi3 = np.deg2rad(transform(*(crpix + 0.5)))
    l4, phi4 = np.deg2rad(transform(*(crpix + [0.5, -0.5])))  # noqa: RUF005
    area = np.abs(
        0.5
        * (
            (l4 - l2) * (np.sin(phi1) - np.sin(phi3))
            + (l1 - l3) * (np.sin(phi2) - np.sin(phi4))
        )
    )
    inv_pscale = 1 / np.rad2deg(np.sqrt(area))

    # form equation:
    def f(x):
        w = np.array(transform(*(x.T), with_bounding_box=False)).T
        dw = np.mod(np.subtract(w, world) - 180.0, 360.0) - 180.0
        return np.add(inv_pscale * dw, x)

    def froot(x):
        return (
            np.mod(
                np.subtract(transform(*x, with_bounding_box=False), worldi) - 180.0,
                360.0,
            )
            - 180.0
        )

    # compute correction:
    def correction(pix):
        p1 = f(pix)
        p2 = f(p1)
        d = p2 - 2.0 * p1 + pix
        idx = np.where(d != 0)
        corr = pix - p2
        corr[idx] = np.square(p1[idx] - pix[idx]) / d[idx]
        return corr

    # initial iteration:
    dpix = correction(pix)

    # Update initial solution:
    pix -= dpix

    # Norm (L2) squared of the correction:
    dn = np.sum(dpix * dpix, axis=1)
    dnprev = dn.copy()  # if adaptive else dn
    tol2 = tolerance**2

    # Prepare for iterative process
    k = 1
    ind = None
    inddiv = None

    # Turn off numpy runtime warnings for 'invalid' and 'over':
    old_invalid = np.geterr()["invalid"]
    old_over = np.geterr()["over"]
    np.seterr(invalid="ignore", over="ignore")

    # ############################################################
    # #                NON-ADAPTIVE ITERATIONS:                 ##
    # ############################################################
    if not adaptive:
        # Fixed-point iterations:
        while np.nanmax(dn) >= tol2 and k < maxiter:
            # Find correction to the previous solution:
            dpix = correction(pix)

            # Compute norm (L2) squared of the correction:
            dn = np.sum(dpix * dpix, axis=1)

            # Check for divergence (we do this in two stages
            # to optimize performance for the most common
            # scenario when successive approximations converge):

            if detect_divergence:
                divergent = dn >= dnprev
                if np.any(divergent):
                    # Find solutions that have not yet converged:
                    slowconv = dn >= tol2
                    (inddiv,) = np.where(divergent & slowconv)

                    if inddiv.shape[0] > 0:
                        # Update indices of elements that
                        # still need correction:
                        conv = dn < dnprev
                        iconv = np.where(conv)

                        # Apply correction:
                        dpixgood = dpix[iconv]
                        pix[iconv] -= dpixgood
                        dpix[iconv] = dpixgood

                        # For the next iteration choose
                        # non-divergent points that have not yet
                        # converged to the requested accuracy:
                        (ind,) = np.where(slowconv & conv)
                        world = world[ind]
                        dnprev[ind] = dn[ind]
                        k += 1

                        # Switch to adaptive iterations:
                        adaptive = True
                        break

                # Save current correction magnitudes for later:
                dnprev = dn

            # Apply correction:
            pix -= dpix
            k += 1

    # ############################################################
    # #                  ADAPTIVE ITERATIONS:                   ##
    # ############################################################
    if adaptive:
        if ind is None:
            (ind,) = np.where(np.isfinite(pix).all(axis=1))
            world = world[ind]

        # "Adaptive" fixed-point iterations:
        while ind.shape[0] > 0 and k < maxiter:
            # Find correction to the previous solution:
            dpixnew = correction(pix[ind])

            # Compute norm (L2) of the correction:
            dnnew = np.sum(np.square(dpixnew), axis=1)

            # Bookkeeping of corrections:
            dnprev[ind] = dn[ind].copy()
            dn[ind] = dnnew

            if detect_divergence:
                # Find indices of pixels that are converging:
                conv = np.logical_or(dnnew < dnprev[ind], dnnew < tol2)
                if not np.all(conv):
                    conv = np.ones_like(dnnew, dtype=bool)
                iconv = np.where(conv)
                iiconv = ind[iconv]

                # Apply correction:
                dpixgood = dpixnew[iconv]
                pix[iiconv] -= dpixgood
                dpix[iiconv] = dpixgood

                # Find indices of solutions that have not yet
                # converged to the requested accuracy
                # AND that do not diverge:
                (subind,) = np.where((dnnew >= tol2) & conv)

            else:
                # Apply correction:
                pix[ind] -= dpixnew
                dpix[ind] = dpixnew

                # Find indices of solutions that have not yet
                # converged to the requested accuracy:
                (subind,) = np.where(dnnew >= tol2)

            # Choose solutions that need more iterations:
            ind = ind[subind]
            world = world[subind]

            k += 1

    # ############################################################
    # #         FINAL DETECTION OF INVALID, DIVERGING,          ##
    # #         AND FAILED-TO-CONVERGE POINTS                   ##
    # ############################################################
    # Identify diverging and/or invalid points:
    invalid = (~np.all(np.isfinite(pix), axis=1)) & (
        np.all(np.isfinite(world0), axis=1)
    )

    # When detect_divergence is False, dnprev is outdated
    # (it is the norm of the very first correction).
    # Still better than nothing...
    (inddiv,) = np.where(((dn >= tol2) & (dn >= dnprev)) | invalid)
    if inddiv.shape[0] == 0:
        inddiv = None

    # If there are divergent points, attempt to find a solution using
    # scipy's 'hybr' method:
    if detect_divergence and inddiv is not None and inddiv.size:
        bad = []
        for idx in inddiv:
            worldi = world0[idx]
            result = optimize.root(
                froot,
                pix0[idx],
                method="hybr",
                tol=tolerance / (np.linalg.norm(pix0[idx]) + 1),
                options={"maxfev": 2 * maxiter},
            )

            if result["success"]:
                pix[idx, :] = result["x"]
                invalid[idx] = False
            else:
                bad.append(idx)

        inddiv = np.array(bad, dtype=int) if bad else None

    # Identify points that did not converge within 'maxiter'
    # iterations:
    if k >= maxiter:
        (ind,) = np.where((dn >= tol2) & (dn < dnprev) & (~invalid))
        if ind.shape[0] == 0:
            ind = None
    else:
        ind = None

    # Restore previous numpy error settings:
    np.seterr(invalid=old_invalid, over=old_over)

    # ############################################################
    # #  RAISE EXCEPTION IF DIVERGING OR TOO SLOWLY CONVERGING  ##
    # #  DATA POINTS HAVE BEEN DETECTED:                        ##
    # ############################################################
    if (ind is not None or inddiv is not None) and not quiet:
        if inddiv is None:
            msg = (
                "'WCS.numerical_inverse' failed to "
                f"converge to the requested accuracy after {k:d} "
                "iterations."
            )
            raise NoConvergence(
                msg,
                best_solution=pix,
                accuracy=np.abs(dpix),
                niter=k,
                slow_conv=ind,
                divergent=None,
            )
        msg = (
            "'WCS.numerical_inverse' failed to "
            "converge to the requested accuracy.\n"
            f"After {k:d} iterations, the solution is diverging "
            "at least for one input point."
        )
        raise NoConvergence(
            msg,
            best_solution=pix,
            accuracy=np.abs(dpix),
            niter=k,
            slow_conv=ind,
            divergent=inddiv,
        )

    if with_bounding_box and bounding_box is not None:
        # find points outside the bounding box and replace their values
        # with fill_value
        valid = np.logical_not(invalid)
        in_bb = np.ones_like(invalid, dtype=np.bool_)

        for c, (x1, x2) in zip(pix[valid].T, bounding_box, strict=False):
            in_bb[valid] &= (c >= x1) & (c <= x2)
        pix[np.logical_not(in_bb)] = fill_value

    return pix
