# Licensed under a 3-clause BSD style license - see LICENSE.rst
import functools
import itertools
import warnings
import numpy as np
import numpy.linalg as npla
from scipy import optimize
from astropy.modeling.core import Model # , fix_inputs
from astropy.modeling import utils as mutils
from astropy.modeling.models import (Shift, Polynomial2D, Sky2Pix_TAN,
                                     RotateCelestial2Native, Mapping)
from astropy.modeling.fitting import LinearLSQFitter
import astropy.io.fits as fits

from .api import GWCSAPIMixin
from . import coordinate_frames as cf
from .utils import CoordinateFrameError
from . import utils
from .wcstools import grid_from_bounding_box


try:
    from astropy.modeling.core import fix_inputs
    HAS_FIX_INPUTS = True
except ImportError:
    HAS_FIX_INPUTS = False


__all__ = ['WCS', 'NoConvergence']

_ITER_INV_KWARGS = ['tolerance', 'maxiter', 'adaptive', 'detect_divergence', 'quiet']


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
    def __init__(self, *args, best_solution=None, accuracy=None, niter=None,
                 divergent=None, slow_conv=None):
        super().__init__(*args)

        self.best_solution = best_solution
        self.accuracy = accuracy
        self.niter = niter
        self.divergent = divergent
        self.slow_conv = slow_conv


class WCS(GWCSAPIMixin):
    """
    Basic WCS class.

    Parameters
    ----------
    forward_transform : `~astropy.modeling.Model` or a list
        The transform between ``input_frame`` and ``output_frame``.
        A list of (frame, transform) tuples where ``frame`` is the starting frame and
        ``transform`` is the transform from this frame to the next one or ``output_frame``.
        The last tuple is (transform, None), where None indicates the end of the pipeline.
    input_frame : str, `~gwcs.coordinate_frames.CoordinateFrame`
        A coordinates object or a string name.
    output_frame : str, `~gwcs.coordinate_frames.CoordinateFrame`
        A coordinates object or a string name.
    name : str
        a name for this WCS

    """

    def __init__(self, forward_transform=None, input_frame='detector', output_frame=None,
                 name=""):
        #self.low_level_wcs = self
        self._approx_inverse = None
        self._available_frames = []
        self._pipeline = []
        self._name = name
        self._array_shape = None
        self._initialize_wcs(forward_transform, input_frame, output_frame)
        self._pixel_shape = None
        self._pipeline = [Step(*step) for step in self._pipeline]

    def _initialize_wcs(self, forward_transform, input_frame, output_frame):
        if forward_transform is not None:
            if isinstance(forward_transform, Model):
                if output_frame is None:
                    raise CoordinateFrameError("An output_frame must be specified"
                                               "if forward_transform is a model.")

                _input_frame, inp_frame_obj = self._get_frame_name(input_frame)
                _output_frame, outp_frame_obj = self._get_frame_name(output_frame)
                super(WCS, self).__setattr__(_input_frame, inp_frame_obj)
                super(WCS, self).__setattr__(_output_frame, outp_frame_obj)

                self._pipeline = [(input_frame, forward_transform.copy()),
                                  (output_frame, None)]
            elif isinstance(forward_transform, list):
                for item in forward_transform:
                    name, frame_obj = self._get_frame_name(item[0])
                    super(WCS, self).__setattr__(name, frame_obj)
                    #self._pipeline.append((name, item[1]))
                    self._pipeline = forward_transform
            else:
                raise TypeError("Expected forward_transform to be a model or a "
                                "(frame, transform) list, got {0}".format(
                                    type(forward_transform)))
        else:
            # Initialize a WCS without a forward_transform - allows building a WCS programmatically.
            if output_frame is None:
                raise CoordinateFrameError("An output_frame must be specified"
                                           "if forward_transform is None.")
            _input_frame, inp_frame_obj = self._get_frame_name(input_frame)
            _output_frame, outp_frame_obj = self._get_frame_name(output_frame)
            super(WCS, self).__setattr__(_input_frame, inp_frame_obj)
            super(WCS, self).__setattr__(_output_frame, outp_frame_obj)
            self._pipeline = [(_input_frame, None),
                              (_output_frame, None)]

    def get_transform(self, from_frame, to_frame):
        """
        Return a transform between two coordinate frames.

        Parameters
        ----------
        from_frame : str or `~gwcs.coordinate_frame.CoordinateFrame`
            Initial coordinate frame name of object.
        to_frame : str, or instance of `~gwcs.cordinate_frames.CoordinateFrame`
            End coordinate frame name or object.

        Returns
        -------
        transform : `~astropy.modeling.Model`
            Transform between two frames.
        """
        if not self._pipeline:
            return None
        try:
            from_ind = self._get_frame_index(from_frame)
        except ValueError:
            raise CoordinateFrameError("Frame {0} is not in the available "
                                       "frames".format(from_frame))
        try:
            to_ind = self._get_frame_index(to_frame)
        except ValueError:
            raise CoordinateFrameError("Frame {0} is not in the available frames".format(to_frame))
        if to_ind < from_ind:
            #transforms = np.array(self._pipeline[to_ind: from_ind], dtype="object")[:, 1].tolist()
            transforms = [step.transform for step in self._pipeline[to_ind: from_ind]]
            transforms = [tr.inverse for tr in transforms[::-1]]
        elif to_ind == from_ind:
            return None
        else:
            #transforms = np.array(self._pipeline[from_ind: to_ind], dtype="object")[:, 1].copy()
            transforms = [step.transform for step in self._pipeline[from_ind: to_ind]]
        return functools.reduce(lambda x, y: x | y, transforms)

    def set_transform(self, from_frame, to_frame, transform):
        """
        Set/replace the transform between two coordinate frames.

        Parameters
        ----------
        from_frame : str or `~gwcs.coordinate_frame.CoordinateFrame`
            Initial coordinate frame.
        to_frame : str, or instance of `~gwcs.cordinate_frames.CoordinateFrame`
            End coordinate frame.
        transform : `~astropy.modeling.Model`
            Transform between ``from_frame`` and ``to_frame``.
        """
        from_name, from_obj = self._get_frame_name(from_frame)
        to_name, to_obj = self._get_frame_name(to_frame)
        if not self._pipeline:
            if from_name != self._input_frame:
                raise CoordinateFrameError(
                    "Expected 'from_frame' to be {0}".format(self._input_frame))
            if to_frame != self._output_frame:
                raise CoordinateFrameError(
                    "Expected 'to_frame' to be {0}".format(self._output_frame))
        try:
            from_ind = self._get_frame_index(from_name)
        except ValueError:
            raise CoordinateFrameError("Frame {0} is not in the available frames".format(from_name))
        try:
            to_ind = self._get_frame_index(to_name)
        except ValueError:
            raise CoordinateFrameError("Frame {0} is not in the available frames".format(to_name))

        if from_ind + 1 != to_ind:
            raise ValueError("Frames {0} and {1} are not  in sequence".format(from_name, to_name))
        self._pipeline[from_ind].transform = transform

    @property
    def forward_transform(self):
        """
        Return the total forward transform - from input to output coordinate frame.

        """

        if self._pipeline:
            #return functools.reduce(lambda x, y: x | y, [step[1] for step in self._pipeline[: -1]])
            return functools.reduce(lambda x, y: x | y, [step.transform for step in self._pipeline[:-1]])
        else:
            return None

    @property
    def backward_transform(self):
        """
        Return the total backward transform if available - from output to input coordinate system.

        Raises
        ------
        NotImplementedError :
            An analytical inverse does not exist.

        """
        try:
            backward = self.forward_transform.inverse
        except NotImplementedError as err:
            raise NotImplementedError("Could not construct backward transform. \n{0}".format(err))
        return backward

    def _get_frame_index(self, frame):
        """
        Return the index in the pipeline where this frame is locate.
        """
        if isinstance(frame, cf.CoordinateFrame):
            frame = frame.name
        #frame_names = [getattr(item[0], "name", item[0]) for item in self._pipeline]
        frame_names = [step.frame if isinstance(step.frame, str) else step.frame.name for step in self._pipeline]
        return frame_names.index(frame)

    def _get_frame_name(self, frame):
        """
        Return the name of the frame and a ``CoordinateFrame`` object.

        Parameters
        ----------
        frame : str, `~gwcs.coordinate_frames.CoordinateFrame`
            Coordinate frame.

        Returns
        -------
        name : str
            The name of the frame
        frame_obj : `~gwcs.coordinate_frames.CoordinateFrame`
            Frame instance or None (if `frame` is str)
        """
        if isinstance(frame, str):
            name = frame
            frame_obj = None
        else:
            name = frame.name
            frame_obj = frame
        return name, frame_obj

    def __call__(self, *args, **kwargs):
        """
        Executes the forward transform.

        args : float or array-like
            Inputs in the input coordinate system, separate inputs
            for each dimension.
        with_units : bool
            If ``True`` returns a `~astropy.coordinates.SkyCoord` or
            `~astropy.units.Quantity` object, by using the units of
            the output cooridnate frame.
            Optional, default=False.
        with_bounding_box : bool, optional
             If True(default) values in the result which correspond to
             any of the inputs being outside the bounding_box are set
             to ``fill_value``.
        fill_value : float, optional
            Output value for inputs outside the bounding_box
            (default is np.nan).
        """
        transform = self.forward_transform
        if transform is None:
            raise NotImplementedError("WCS.forward_transform is not implemented.")

        with_units = kwargs.pop("with_units", False)
        if 'with_bounding_box' not in kwargs:
            kwargs['with_bounding_box'] = True
        if 'fill_value' not in kwargs:
            kwargs['fill_value'] = np.nan

        if self.bounding_box is not None:
            # Currently compound models do not attempt to combine individual model
            # bounding boxes. Get the forward transform and assign the bounding_box to it
            # before evaluating it. The order Model.bounding_box is reversed.
            axes_ind = self._get_axes_indices()
            if transform.n_inputs > 1:
                transform.bounding_box = [self.bounding_box[ind] for ind in axes_ind][::-1]
            else:
                transform.bounding_box = self.bounding_box
        result = transform(*args, **kwargs)

        if with_units:
            if self.output_frame.naxes == 1:
                result = self.output_frame.coordinates(result)
            else:
                result = self.output_frame.coordinates(*result)

        return result

    def in_image(self, *args, **kwargs):
        """
        This method tests if one or more of the input world coordinates are
        contained within forward transformation's image and that it maps to
        the domain of definition of the forward transformation.
        In practical terms, this function tests
        that input world coordinate(s) can be converted to input frame and that
        it is within the forward transformation's ``bounding_box`` when
        defined.

        Parameters
        ----------
        args : float, array like, `~astropy.coordinates.SkyCoord` or
            `~astropy.units.Unit` coordinates to be inverted

        kwargs : dict
            keyword arguments to be passed either to ``backward_transform``
            (when defined) or to the iterative invert method.

        Returns
        -------
        result : bool, numpy.ndarray
           A single boolean value or an array of boolean values with `True`
           indicating that the WCS footprint contains the coordinate
           and `False` if input is outside the footprint.

        """
        kwargs['with_bounding_box'] = True
        kwargs['fill_value'] = np.nan

        coords = self.invert(*args, **kwargs)

        result = np.isfinite(coords)
        if self.input_frame.naxes > 1:
            result = np.all(result, axis=0)

        if self.bounding_box is None or not np.any(result):
            return result

        if self.input_frame.naxes == 1:
            x1, x2 = self.bounding_box

            if len(np.shape(args[0])) > 0:
                result[result] = (coords[result] >= x1) & (coords[result] <= x2)
            elif result:
                result = (coords >= x1) and (coords <= x2)

        else:
            if len(np.shape(args[0])) > 0:
                for c, (x1, x2) in zip(coords, self.bounding_box):
                    result[result] = (c[result] >= x1) & (c[result] <= x2)

            elif result:
                result = all([(c >= x1) and (c <= x2) for c, (x1, x2) in zip(coords, self.bounding_box)])

        return result

    def invert(self, *args, **kwargs):
        """
        Invert coordinates from output frame to input frame using analytical or
        user-supplied inverse. When neither analytical nor user-supplied
        inverses are defined, a numerical solution will be attempted using
        :py:meth:`numerical_inverse`.

        .. note::
            Currently numerical inverse is implemented only for 2D imaging WCS.

        Parameters
        ----------
        args : float, array like, `~astropy.coordinates.SkyCoord` or `~astropy.units.Unit`
            Coordinates to be inverted. The number of arguments must be equal
            to the number of world coordinates given by ``world_n_dim``.

        with_bounding_box : bool, optional
             If `True` (default) values in the result which correspond to any
             of the inputs being outside the bounding_box are set to
             ``fill_value``.

        fill_value : float, optional
            Output value for inputs outside the bounding_box (default is ``np.nan``).

        with_units : bool, optional
            If ``True`` returns a `~astropy.coordinates.SkyCoord` or
            `~astropy.units.Quantity` object, by using the units of
            the output cooridnate frame. Default is `False`.

        Other Parameters
        ----------------
        kwargs : dict
            Keyword arguments to be passed to :py:meth:`numerical_inverse`
            (when defined) or to the iterative invert method.

        Returns
        -------
        result : tuple
            Returns a tuple of scalar or array values for each axis.

        """
        with_units = kwargs.pop('with_units', False)

        if not utils.isnumerical(args[0]):
            args = self.output_frame.coordinate_to_quantity(*args)
            if self.output_frame.naxes == 1:
                args = [args]
            try:
                if not self.backward_transform.uses_quantity:
                    args = utils.get_values(self.output_frame.unit, *args)
            except (NotImplementedError, KeyError):
                args = utils.get_values(self.output_frame.unit, *args)

        if 'with_bounding_box' not in kwargs:
            kwargs['with_bounding_box'] = True

        if 'fill_value' not in kwargs:
            kwargs['fill_value'] = np.nan

        try:
            # remove iterative inverse-specific keyword arguments:
            akwargs = {k: v for k, v in kwargs.items() if k not in _ITER_INV_KWARGS}
            result = self.backward_transform(*args, **akwargs)
        except (NotImplementedError, KeyError):
            result = self.numerical_inverse(*args, **kwargs, with_units=with_units)

        if with_units and self.input_frame:
            if self.input_frame.naxes == 1:
                return self.input_frame.coordinates(result)
            else:
                return self.input_frame.coordinates(*result)
        else:
            return result

    def numerical_inverse(self, *args, **kwargs):
        """
        Invert coordinates from output frame to input frame using numerical
        inverse.

        .. note::
            Currently numerical inverse is implemented only for 2D imaging WCS.

        .. note::
            This method uses a combination of vectorized fixed-point
            iterations algorithm and `scipy.optimize.root`. The later is used
            for input coordinates for which vectorized algorithm diverges.

        Parameters
        ----------
        args : float, array like, `~astropy.coordinates.SkyCoord` or `~astropy.units.Unit`
            Coordinates to be inverted. The number of arguments must be equal
            to the number of world coordinates given by ``world_n_dim``.

        with_bounding_box : bool, optional
             If `True` (default) values in the result which correspond to any
             of the inputs being outside the bounding_box are set to
             ``fill_value``.

        fill_value : float, optional
            Output value for inputs outside the bounding_box (default is ``np.nan``).

        with_units : bool, optional
            If ``True`` returns a `~astropy.coordinates.SkyCoord` or
            `~astropy.units.Quantity` object, by using the units of
            the output cooridnate frame. Default is `False`.

        tolerance : float, optional
            *Absolute tolerance* of solution. Iteration terminates when the
            iterative solver estimates that the "true solution" is
            within this many pixels current estimate, more
            specifically, when the correction to the solution found
            during the previous iteration is smaller
            (in the sense of the L2 norm) than ``tolerance``.
            Default ``tolerance`` is 1.0e-5.

        maxiter : int, optional
            Maximum number of iterations allowed to reach a solution.
            Default is 50.

        quiet : bool, optional
            Do not throw :py:class:`NoConvergence` exceptions when
            the method does not converge to a solution with the
            required accuracy within a specified number of maximum
            iterations set by ``maxiter`` parameter. Instead,
            simply return the found solution. Default is `True`.

        Other Parameters
        ----------------
        adaptive : bool, optional
            Specifies whether to adaptively select only points that
            did not converge to a solution within the required
            accuracy for the next iteration. Default (`True`) is recommended.

            .. note::
               The :py:meth:`numerical_inverse` uses a vectorized
               implementation of the method of consecutive
               approximations (see ``Notes`` section below) in which it
               iterates over *all* input points *regardless* until
               the required accuracy has been reached for *all* input
               points. In some cases it may be possible that
               *almost all* points have reached the required accuracy
               but there are only a few of input data points for
               which additional iterations may be needed (this
               depends mostly on the characteristics of the geometric
               distortions for a given instrument). In this situation
               it may be advantageous to set ``adaptive`` = `True` in
               which case :py:meth:`numerical_inverse` will continue
               iterating *only* over the points that have not yet
               converged to the required accuracy.

            .. note::
               When ``detect_divergence`` is `True`,
               :py:meth:`numerical_inverse` will automatically switch
               to the adaptive algorithm once divergence has been
               detected.

        detect_divergence : bool, optional
            Specifies whether to perform a more detailed analysis
            of the convergence to a solution. Normally
            :py:meth:`numerical_inverse` may not achieve the required
            accuracy if either the ``tolerance`` or ``maxiter`` arguments
            are too low. However, it may happen that for some
            geometric distortions the conditions of convergence for
            the the method of consecutive approximations used by
            :py:meth:`numerical_inverse` may not be satisfied, in which
            case consecutive approximations to the solution will
            diverge regardless of the ``tolerance`` or ``maxiter``
            settings.

            When ``detect_divergence`` is `False`, these divergent
            points will be detected as not having achieved the
            required accuracy (without further details). In addition,
            if ``adaptive`` is `False` then the algorithm will not
            know that the solution (for specific points) is diverging
            and will continue iterating and trying to "improve"
            diverging solutions. This may result in ``NaN`` or
            ``Inf`` values in the return results (in addition to a
            performance penalties). Even when ``detect_divergence``
            is `False`, :py:meth:`numerical_inverse`, at the end of the
            iterative process, will identify invalid results
            (``NaN`` or ``Inf``) as "diverging" solutions and will
            raise :py:class:`NoConvergence` unless the ``quiet``
            parameter is set to `True`.

            When ``detect_divergence`` is `True` (default),
            :py:meth:`numerical_inverse` will detect points for which
            current correction to the coordinates is larger than
            the correction applied during the previous iteration
            **if** the requested accuracy **has not yet been
            achieved**. In this case, if ``adaptive`` is `True`,
            these points will be excluded from further iterations and
            if ``adaptive`` is `False`, :py:meth:`numerical_inverse` will
            automatically switch to the adaptive algorithm. Thus, the
            reported divergent solution will be the latest converging
            solution computed immediately *before* divergence
            has been detected.

            .. note::
               When accuracy has been achieved, small increases in
               current corrections may be possible due to rounding
               errors (when ``adaptive`` is `False`) and such
               increases will be ignored.

            .. note::
               Based on our testing using JWST NIRCAM images, setting
               ``detect_divergence`` to `True` will incur about 5-10%
               performance penalty with the larger penalty
               corresponding to ``adaptive`` set to `True`.
               Because the benefits of enabling this
               feature outweigh the small performance penalty,
               especially when ``adaptive`` = `False`, it is
               recommended to set ``detect_divergence`` to `True`,
               unless extensive testing of the distortion models for
               images from specific instruments show a good stability
               of the numerical method for a wide range of
               coordinates (even outside the image itself).

            .. note::
               Indices of the diverging inverse solutions will be
               reported in the ``divergent`` attribute of the
               raised :py:class:`NoConvergence` exception object.

        Returns
        -------
        result : tuple
            Returns a tuple of scalar or array values for each axis.

        Raises
        ------
        NoConvergence
            The iterative method did not converge to a
            solution to the required accuracy within a specified
            number of maximum iterations set by the ``maxiter``
            parameter. To turn off this exception, set ``quiet`` to
            `True`. Indices of the points for which the requested
            accuracy was not achieved (if any) will be listed in the
            ``slow_conv`` attribute of the
            raised :py:class:`NoConvergence` exception object.

            See :py:class:`NoConvergence` documentation for
            more details.

        NotImplementedError
            Numerical inverse has not been implemented for this WCS.

        ValueError
            Invalid argument values.

        Examples
        --------
        >>> from astropy.utils.data import get_pkg_data_filename
        >>> from gwcs import NoConvergence
        >>> import asdf
        >>> import numpy as np

        >>> filename = get_pkg_data_filename('data/nircamwcs.asdf', package='gwcs.tests')
        >>> w = asdf.open(filename).tree['wcs']

        >>> ra, dec = w([1,2,3], [1,1,1])
        >>> print(ra)  # doctest: +FLOAT_CMP
        [5.927628   5.92757069 5.92751337]
        >>> print(dec)  # doctest: +FLOAT_CMP
        [-72.01341247 -72.01341273 -72.013413  ]

        >>> x, y = w.numerical_inverse(ra, dec)
        >>> print(x)  # doctest: +FLOAT_CMP
        [1.00000005 2.00000005 3.00000006]
        >>> print(y)  # doctest: +FLOAT_CMP
        [1.00000004 0.99999979 1.00000015]

        >>> x, y = w.numerical_inverse(ra, dec, maxiter=3, tolerance=1.0e-10, quiet=False)
        Traceback (most recent call last):
        ...
        gwcs.wcs.NoConvergence: 'WCS.numerical_inverse' failed to converge to the
        requested accuracy after 3 iterations.

        >>> w.numerical_inverse(
        ...     *w([1, 300000, 3], [2, 1000000, 5], with_bounding_box=False),
        ...     adaptive=False,
        ...     detect_divergence=True,
        ...     quiet=False,
        ...     with_bounding_box=False
        ... )
        Traceback (most recent call last):
        ...
        gwcs.wcs.NoConvergence: 'WCS.numerical_inverse' failed to converge to the
        requested accuracy. After 4 iterations, the solution is diverging at
        least for one input point.

        >>> # Now try to use some diverging data:
        >>> divradec = w([1, 300000, 3], [2, 1000000, 5], with_bounding_box=False)
        >>> print(divradec)  # doctest: +FLOAT_CMP
        (array([  5.92762673, 148.21600848,   5.92750827]),
         array([-72.01339464,  -7.80968079, -72.01334172]))
        >>> try:  # doctest: +SKIP
        ...     x, y = w.numerical_inverse(*divradec, maxiter=20,
        ...                                tolerance=1.0e-4, adaptive=True,
        ...                                detect_divergence=True,
        ...                                quiet=False)
        ... except NoConvergence as e:
        ...     print(f"Indices of diverging points: {e.divergent}")
        ...     print(f"Indices of poorly converging points: {e.slow_conv}")
        ...     print(f"Best solution:\\n{e.best_solution}")
        ...     print(f"Achieved accuracy:\\n{e.accuracy}")
        Indices of diverging points: None
        Indices of poorly converging points: [1]
        Best solution:
        [[1.00000040e+00 1.99999841e+00]
         [6.33507833e+17 3.40118820e+17]
         [3.00000038e+00 4.99999841e+00]]
        Achieved accuracy:
        [[2.75925982e-05 1.18471543e-05]
         [3.65405005e+04 1.31364188e+04]
         [2.76552923e-05 1.14789013e-05]]

        """
        tolerance = kwargs.get('tolerance', 1e-5)
        maxiter = kwargs.get('maxiter', 50)
        adaptive = kwargs.get('adaptive', True)
        detect_divergence = kwargs.get('detect_divergence', True)
        quiet = kwargs.get('quiet', True)
        with_bounding_box = kwargs.get('with_bounding_box', True)
        fill_value = kwargs.get('fill_value', np.nan)
        with_units = kwargs.pop('with_units', False)

        if not utils.isnumerical(args[0]):
            args = self.output_frame.coordinate_to_quantity(*args)
            if self.output_frame.naxes == 1:
                args = [args]
            args = utils.get_values(self.output_frame.unit, *args)

        args_shape = np.shape(args)
        nargs = args_shape[0]
        arg_dim = len(args_shape) - 1

        if nargs != self.world_n_dim:
            raise ValueError("Number of input coordinates is different from "
                             "the number of defined world coordinates in the "
                             f"WCS ({self.world_n_dim:d})")

        if self.world_n_dim != self.pixel_n_dim:
            raise NotImplementedError(
                "Support for iterative inverse for transformations with "
                "different number of inputs and outputs was not implemented."
            )

        # initial guess:
        if nargs == 2 and self._approx_inverse is None:
            self._calc_approx_inv(max_inv_pix_error=5, inv_degree=None)

        if self._approx_inverse is None:
            if self.bounding_box is None:
                x0 = np.ones(self.pixel_n_dim)
            else:
                x0 = np.mean(self.bounding_box, axis=-1)

        if arg_dim == 0:
            argsi = args

            if nargs == 2 and self._approx_inverse is not None:
                x0 = self._approx_inverse(*argsi)
                if not np.all(np.isfinite(x0)):
                    return [np.array(np.nan) for _ in range(nargs)]

            result = tuple(self._vectorized_fixed_point(
                x0, argsi,
                tolerance=tolerance,
                maxiter=maxiter,
                adaptive=adaptive,
                detect_divergence=detect_divergence,
                quiet=quiet,
                with_bounding_box=with_bounding_box,
                fill_value=fill_value
            ).T.ravel().tolist())

        else:
            arg_shape = args_shape[1:]
            nelem = np.prod(arg_shape)

            args = np.reshape(args, (nargs, nelem))

            if self._approx_inverse is None:
                x0 = np.full((nelem, nargs), x0)
            else:
                x0 = np.array(self._approx_inverse(*args)).T

            result = self._vectorized_fixed_point(
                x0, args.T,
                tolerance=tolerance,
                maxiter=maxiter,
                adaptive=adaptive,
                detect_divergence=detect_divergence,
                quiet=quiet,
                with_bounding_box=with_bounding_box,
                fill_value=fill_value
            ).T

            result = tuple(np.reshape(result, args_shape))

        if with_units and self.input_frame:
            if self.input_frame.naxes == 1:
                return self.input_frame.coordinates(result)
            else:
                return self.input_frame.coordinates(*result)
        else:
            return result

    def _vectorized_fixed_point(self, pix0, world, tolerance, maxiter,
                                adaptive, detect_divergence, quiet,
                                with_bounding_box, fill_value):
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
        if self.bounding_box is None:
            crpix = np.ones(self.pixel_n_dim)
        else:
            crpix = np.mean(self.bounding_box, axis=-1)

        l1, phi1 = np.deg2rad(self.__call__(*(crpix - 0.5)))
        l2, phi2 = np.deg2rad(self.__call__(*(crpix + [-0.5, 0.5])))
        l3, phi3 = np.deg2rad(self.__call__(*(crpix + 0.5)))
        l4, phi4 = np.deg2rad(self.__call__(*(crpix + [0.5, -0.5])))
        area = np.abs(0.5 * ((l4 - l2) * np.sin(phi1) +
                             (l1 - l3) * np.sin(phi2) +
                             (l2 - l4) * np.sin(phi3) +
                             (l3 - l2) * np.sin(phi4)))
        inv_pscale = 1 / np.rad2deg(np.sqrt(area))

        # form equation:
        def f(x):
            w = np.array(self.__call__(*(x.T), with_bounding_box=False)).T
            dw = np.mod(np.subtract(w, world) - 180.0, 360.0) - 180.0
            return np.add(inv_pscale * dw, x)

        def froot(x):
            return np.mod(np.subtract(self.__call__(*x, with_bounding_box=False), worldi) - 180.0, 360.0) - 180.0

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
        old_invalid = np.geterr()['invalid']
        old_over = np.geterr()['over']
        np.seterr(invalid='ignore', over='ignore')

        # ############################################################
        # #                NON-ADAPTIVE ITERATIONS:                 ##
        # ############################################################
        if not adaptive:
            # Fixed-point iterations:
            while (np.nanmax(dn) >= tol2 and k < maxiter):
                # Find correction to the previous solution:
                dpix = correction(pix)

                # Compute norm (L2) squared of the correction:
                dn = np.sum(dpix * dpix, axis=1)

                # Check for divergence (we do this in two stages
                # to optimize performance for the most common
                # scenario when successive approximations converge):

                if detect_divergence:
                    divergent = (dn >= dnprev)
                    if np.any(divergent):
                        # Find solutions that have not yet converged:
                        slowconv = (dn >= tol2)
                        inddiv, = np.where(divergent & slowconv)

                        if inddiv.shape[0] > 0:
                            # Update indices of elements that
                            # still need correction:
                            conv = (dn < dnprev)
                            iconv = np.where(conv)

                            # Apply correction:
                            dpixgood = dpix[iconv]
                            pix[iconv] -= dpixgood
                            dpix[iconv] = dpixgood

                            # For the next iteration choose
                            # non-divergent points that have not yet
                            # converged to the requested accuracy:
                            ind, = np.where(slowconv & conv)
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
                ind, = np.where(np.isfinite(pix).all(axis=1))
                world = world[ind]

            # "Adaptive" fixed-point iterations:
            while (ind.shape[0] > 0 and k < maxiter):
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
                        conv = np.ones_like(dnnew, dtype=np.bool)
                    iconv = np.where(conv)
                    iiconv = ind[iconv]

                    # Apply correction:
                    dpixgood = dpixnew[iconv]
                    pix[iiconv] -= dpixgood
                    dpix[iiconv] = dpixgood

                    # Find indices of solutions that have not yet
                    # converged to the requested accuracy
                    # AND that do not diverge:
                    subind, = np.where((dnnew >= tol2) & conv)

                else:
                    # Apply correction:
                    pix[ind] -= dpixnew
                    dpix[ind] = dpixnew

                    # Find indices of solutions that have not yet
                    # converged to the requested accuracy:
                    subind, = np.where(dnnew >= tol2)

                # Choose solutions that need more iterations:
                ind = ind[subind]
                world = world[subind]

                k += 1

        # ############################################################
        # #         FINAL DETECTION OF INVALID, DIVERGING,          ##
        # #         AND FAILED-TO-CONVERGE POINTS                   ##
        # ############################################################
        # Identify diverging and/or invalid points:
        invalid = ((~np.all(np.isfinite(pix), axis=1)) &
                   (np.all(np.isfinite(world0), axis=1)))

        # When detect_divergence is False, dnprev is outdated
        # (it is the norm of the very first correction).
        # Still better than nothing...
        inddiv, = np.where(((dn >= tol2) & (dn >= dnprev)) | invalid)
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
                    method='hybr',
                    tol=tolerance / (np.linalg.norm(pix0[idx]) + 1),
                    options={'maxfev': 2 * maxiter}
                )

                if result['success']:
                    pix[idx, :] = result['x']
                    invalid[idx] = False
                else:
                    bad.append(idx)

            if bad:
                inddiv = np.array(bad, dtype=np.int)
            else:
                inddiv = None

        # Identify points that did not converge within 'maxiter'
        # iterations:
        if k >= maxiter:
            ind, = np.where((dn >= tol2) & (dn < dnprev) & (~invalid))
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
                raise NoConvergence(
                    "'WCS.numerical_inverse' failed to "
                    "converge to the requested accuracy after {:d} "
                    "iterations.".format(k), best_solution=pix,
                    accuracy=np.abs(dpix), niter=k,
                    slow_conv=ind, divergent=None)
            else:
                raise NoConvergence(
                    "'WCS.numerical_inverse' failed to "
                    "converge to the requested accuracy.\n"
                    "After {:d} iterations, the solution is diverging "
                    "at least for one input point."
                    .format(k), best_solution=pix,
                    accuracy=np.abs(dpix), niter=k,
                    slow_conv=ind, divergent=inddiv)

        if with_bounding_box and self.bounding_box is not None:
            # find points outside the bounding box and replace their values
            # with fill_value
            valid = np.logical_not(invalid)
            in_bb = np.ones_like(invalid, dtype=np.bool_)

            for c, (x1, x2) in zip(pix[valid].T, self.bounding_box):
                in_bb[valid] &= (c >= x1) & (c <= x2)
            pix[np.logical_not(in_bb)] = fill_value

        return pix

    def transform(self, from_frame, to_frame, *args, **kwargs):
        """
        Transform positions between two frames.

        Parameters
        ----------
        from_frame : str or `~gwcs.coordinate_frames.CoordinateFrame`
            Initial coordinate frame.
        to_frame : str, or instance of `~gwcs.cordinate_frames.CoordinateFrame`
            Coordinate frame into which to transform.
        args : float or array-like
            Inputs in ``from_frame``, separate inputs for each dimension.
        output_with_units : bool
            If ``True`` - returns a `~astropy.coordinates.SkyCoord` or
            `~astropy.units.Quantity` object.
        with_bounding_box : bool, optional
             If True(default) values in the result which correspond to any of the inputs being
             outside the bounding_box are set to ``fill_value``.
        fill_value : float, optional
            Output value for inputs outside the bounding_box (default is np.nan).
        """
        transform = self.get_transform(from_frame, to_frame)
        if not utils.isnumerical(args[0]):
            inp_frame = getattr(self, from_frame)
            args = inp_frame.coordinate_to_quantity(*args)
            if not transform.uses_quantity:
                args = utils.get_values(inp_frame.unit, *args)

        with_units = kwargs.pop("with_units", False)
        if 'with_bounding_box' not in kwargs:
            kwargs['with_bounding_box'] = True
        if 'fill_value' not in kwargs:
            kwargs['fill_value'] = np.nan

        result = transform(*args, **kwargs)

        if with_units:
            to_frame_name, to_frame_obj = self._get_frame_name(to_frame)
            if to_frame_obj is not None:
                if to_frame_obj.naxes == 1:
                    result = to_frame_obj.coordinates(result)
                else:
                    result = to_frame_obj.coordinates(*result)
            else:
                raise TypeError("Coordinate objects could not be created because"
                                "frame {0} is not defined.".format(to_frame_name))

        return result

    @property
    def available_frames(self):
        """
        List all frames in this WCS object.

        Returns
        -------
        available_frames : dict
            {frame_name: frame_object or None}
        """
        if self._pipeline:
            #return [getattr(frame[0], "name", frame[0]) for frame in self._pipeline]
            return [step.frame if isinstance(step.frame, str) else step.frame.name for step in self._pipeline ]
        else:
            return None

    def insert_transform(self, frame, transform, after=False):
        """
        Insert a transform before (default) or after a coordinate frame.

        Append (or prepend) a transform to the transform connected to frame.

        Parameters
        ----------
        frame : str or `~gwcs.coordinate_frame.CoordinateFrame`
            Coordinate frame which sets the point of insertion.
        transform : `~astropy.modeling.Model`
            New transform to be inserted in the pipeline
        after : bool
            If True, the new transform is inserted in the pipeline
            immediately after ``frame``.
        """
        name, _ = self._get_frame_name(frame)
        frame_ind = self._get_frame_index(name)
        if not after:
            current_transform = self._pipeline[frame_ind - 1].transform
            self._pipeline[frame_ind - 1].transform = current_transform | transform
        else:
            current_transform = self._pipeline[frame_ind].transform
            self._pipeline[frame_ind].transform = transform | current_transform

    def insert_frame(self, input_frame, transform, output_frame):
        """
        Insert a new frame into an existing pipeline. This frame must be
        anchored to a frame already in the pipeline by a transform. This
        existing frame is identified solely by its name, although an entire
        `~gwcs.coordinate_frame.CoordinateFrame` can be passed (e.g., the
        `input_frame` or `output_frame` attribute). This frame is never
        modified.

        Parameters
        ----------
        input_frame : str or `~gwcs.coordinate_frame.CoordinateFrame`
            Coordinate frame at start of new transform
        transform : `~astropy.modeling.Model`
            New transform to be inserted in the pipeline
        output_frame: str or `~gwcs.coordinate_frame.CoordinateFrame`
            Coordinate frame at end of new transform
        """
        input_name, input_frame_obj = self._get_frame_name(input_frame)
        output_name, output_frame_obj = self._get_frame_name(output_frame)
        try:
            input_index = self._get_frame_index(input_frame)
        except ValueError:
            input_index = None
            if input_frame_obj is None:
                raise ValueError(f"New coordinate frame {input_name} must "
                                 "be defined")
        try:
            output_index = self._get_frame_index(output_frame)
        except ValueError:
            output_index = None
            if output_frame_obj is None:
                raise ValueError(f"New coordinate frame {output_name} must "
                                 "be defined")

        new_frames = [input_index, output_index].count(None)
        if new_frames == 0:
            raise ValueError("Could not insert frame as both frames "
                             f"{input_name} and {output_name} already exist")
        elif new_frames == 2:
            raise ValueError("Could not insert frame as neither frame "
                             f"{input_name} nor {output_name} exists")

        if input_index is None:
            self._pipeline = (self._pipeline[:output_index] +
                              [Step(input_frame_obj, transform)] +
                              self._pipeline[output_index:])
            super(WCS, self).__setattr__(input_name, input_frame_obj)
        else:
            split_step = self._pipeline[input_index]
            self._pipeline = (self._pipeline[:input_index] +
                              [Step(split_step.frame, transform),
                               Step(output_frame_obj, split_step.transform)] +
                              self._pipeline[input_index + 1:])
            super(WCS, self).__setattr__(output_name, output_frame_obj)

    @property
    def unit(self):
        """The unit of the coordinates in the output coordinate system."""
        if self._pipeline:
            try:
                #return getattr(self, self._pipeline[-1][0].name).unit
                return self._pipeline[-1].frame.unit
            except AttributeError:
                return None
        else:
            return None

    @property
    def output_frame(self):
        """Return the output coordinate frame."""
        if self._pipeline:
            frame = self._pipeline[-1].frame
            if not isinstance(frame, str):
                frame = frame.name
            return getattr(self, frame)
        else:
            return None

    @property
    def input_frame(self):
        """Return the input coordinate frame."""
        if self._pipeline:
            frame = self._pipeline[0].frame
            if not isinstance(frame, str):
                frame = frame.name
            return getattr(self, frame)
        else:
            return None

    @property
    def name(self):
        """Return the name for this WCS."""
        return self._name

    @name.setter
    def name(self, value):
        """Set the name for the WCS."""
        self._name = value

    @property
    def pipeline(self):
        """Return the pipeline structure."""
        return self._pipeline

    @property
    def bounding_box(self):
        """
        Return the range of acceptable values for each input axis.
        The order of the axes is `~gwcs.coordinate_frames.CoordinateFrame.axes_order`.
        """
        frames = self.available_frames
        transform_0 = self.get_transform(frames[0], frames[1])
        try:
            bb = transform_0.bounding_box
        except NotImplementedError:
            return None
        if transform_0.n_inputs == 1:
            return bb
        try:
            axes_order = self.input_frame.axes_order
        except AttributeError:
            axes_order = np.arange(transform_0.n_inputs)
        # Model.bounding_box is in python order, need to reverse it first.
        return tuple(bb[::-1][i] for i in axes_order)

    @bounding_box.setter
    def bounding_box(self, value):
        """
        Set the range of acceptable values for each input axis.

        The order of the axes is `~gwcs.coordinate_frames.CoordinateFrame.axes_order`.
        For two inputs and axes_order(0, 1) the bounding box is ((xlow, xhigh), (ylow, yhigh)).

        Parameters
        ----------
        value : tuple or None
            Tuple of tuples with ("low", high") values for the range.
        """
        frames = self.available_frames
        transform_0 = self.get_transform(frames[0], frames[1])
        if value is None:
            transform_0.bounding_box = value
        else:
            try:
                # Make sure the dimensions of the new bbox are correct.
                mutils._BoundingBox.validate(transform_0, value)
            except Exception:
                raise
            # get the sorted order of axes' indices
            axes_ind = self._get_axes_indices()
            if transform_0.n_inputs == 1:
                transform_0.bounding_box = value
            else:
                # The axes in bounding_box in modeling follow python order
                #transform_0.bounding_box = np.array(value)[axes_ind][::-1]
                transform_0.bounding_box = [value[ind] for ind in axes_ind][::-1]
        self.set_transform(frames[0], frames[1], transform_0)

    def _get_axes_indices(self):
        try:
            axes_ind = np.argsort(self.input_frame.axes_order)
        except AttributeError:
            # the case of a frame being a string
            axes_ind = np.arange(self.forward_transform.n_inputs)
        return axes_ind

    def __str__(self):
        from astropy.table import Table
        #col1 = [item[0] for item in self._pipeline]
        col1 = [step.frame for step in self._pipeline]
        col2 = []
        for item in self._pipeline[: -1]:
            #model = item[1]
            model = item.transform
            if model.name is not None:
                col2.append(model.name)
            else:
                col2.append(model.__class__.__name__)
        col2.append(None)
        t = Table([col1, col2], names=['From', 'Transform'])
        return str(t)

    def __repr__(self):
        fmt = "<WCS(output_frame={0}, input_frame={1}, forward_transform={2})>".format(
            self.output_frame, self.input_frame, self.forward_transform)
        return fmt

    def footprint(self, bounding_box=None, center=False, axis_type="all"):
        """
        Return the footprint in world coordinates.

        Parameters
        ----------
        bounding_box : tuple of floats: (start, stop)
            `prop: bounding_box`
        center : bool
            If `True` use the center of the pixel, otherwise use the corner.
        axis_type : str
            A supported ``output_frame.axes_type`` or "all" (default).
            One of ['spatial', 'spectral', 'temporal'] or a custom type.

        Returns
        -------
        coord : ndarray
            Array of coordinates in the output_frame mapping
            corners to the output frame. For spatial coordinates the order
            is clockwise, starting from the bottom left corner.

        """
        def _order_clockwise(v):
            return np.asarray([[v[0][0], v[1][0]], [v[0][0], v[1][1]],
                               [v[0][1], v[1][1]], [v[0][1], v[1][0]]]).T

        if bounding_box is None:
            if self.bounding_box is None:
                raise TypeError("Need a valid bounding_box to compute the footprint.")
            bb = self.bounding_box
        else:
            bb = bounding_box

        all_spatial = all([t.lower() == "spatial" for t in self.output_frame.axes_type])

        if all_spatial:
            vertices = _order_clockwise(bb)
        else:
            vertices = np.array(list(itertools.product(*bb))).T

        if center:
            vertices = utils._toindex(vertices)

        result = np.asarray(self.__call__(*vertices, **{'with_bounding_box': False}))

        axis_type = axis_type.lower()
        if axis_type == 'spatial' and all_spatial:
            return result.T

        if axis_type != "all":
            axtyp_ind = np.array([t.lower() for t in self.output_frame.axes_type]) == axis_type
            if not axtyp_ind.any():
                raise ValueError('This WCS does not have axis of type "{}".'.format(axis_type))
            result = np.asarray([(r.min(), r.max()) for r in result[axtyp_ind]])

            if axis_type == "spatial":
                result = _order_clockwise(result)
            else:
                result.sort()
                result = np.squeeze(result)

        return result.T

    def fix_inputs(self, fixed):
        """
        Return a new unique WCS by fixing inputs to constant values.

        Parameters
        ----------
        fixed : dict
            Keyword arguments with fixed values corresponding to `self.selector`.

        Returns
        -------
        new_wcs : `WCS`
            A new unique WCS corresponding to the values in `fixed`.

        Examples
        --------
        >>> w = WCS(pipeline, selector={"spectral_order": [1, 2]}) # doctest: +SKIP
        >>> new_wcs = w.set_inputs(spectral_order=2) # doctest: +SKIP
        >>> new_wcs.inputs # doctest: +SKIP
            ("x", "y")

        """
        if not HAS_FIX_INPUTS:
            raise ImportError('"fix_inputs" needs astropy version >= 4.0.')

        new_pipeline = []
        step0 = self.pipeline[0]
        new_transform = fix_inputs(step0[1], fixed)
        new_pipeline.append((step0[0], new_transform))
        new_pipeline.extend(self.pipeline[1:])
        return self.__class__(new_pipeline)

    def to_fits_sip(self, bounding_box=None, max_pix_error=0.25, degree=None,
                    max_inv_pix_error=0.25, inv_degree=None,
                    npoints=32, crpix=None, verbose=False):
        """
        Construct a SIP-based approximation to the WCS in the form of a FITS header

        This assumes a tangent projection.

        The default mode in using this attempts to achieve roughly 0.25 pixel
        accuracy over the whole image.

        Parameters
        ----------
        bounding_box : tuple, optional
            A pair of tuples, each consisting of two numbers
            Represents the range of pixel values in both dimensions
            ((xmin, xmax), (ymin, ymax))

        max_pix_error : float, optional
            Maximum allowed error over the domain of the pixel array. This
            error is the equivalent pixel error that corresponds to the maximum
            error in the output coordinate resulting from the fit based on
            a nominal plate scale.

        degree : int, iterable, None, optional
            Degree of the SIP polynomial. Default value `None` indicates that
            all allowed degree values (``[1...9]``) will be considered and
            the lowest degree that meets accuracy requerements set by
            ``max_pix_error`` will be returned. Alternatively, ``degree`` can be
            an iterable containing allowed values for the SIP polynomial degree.
            This option is similar to default `None` but it allows caller to
            restrict the range of allowed SIP degrees used for fitting.
            Finally, ``degree`` can be an integer indicating the exact SIP degree
            to be fit to the WCS transformation. In this case
            ``max_pixel_error`` is ignored.

        max_inv_error : float, optional
            Maximum allowed inverse error over the domain of the pixel array
            in pixel units. If None, no inverse is generated.

        inv_degree : int, iterable, None, optional
            Degree of the SIP polynomial. Default value `None` indicates that
            all allowed degree values (``[1...9]``) will be considered and
            the lowest degree that meets accuracy requerements set by
            ``max_pix_error`` will be returned. Alternatively, ``degree`` can be
            an iterable containing allowed values for the SIP polynomial degree.
            This option is similar to default `None` but it allows caller to
            restrict the range of allowed SIP degrees used for fitting.
            Finally, ``degree`` can be an integer indicating the exact SIP degree
            to be fit to the WCS transformation. In this case
            ``max_inv_pixel_error`` is ignored.

        npoints : int, optional
            The number of points in each dimension to sample the bounding box
            for use in the SIP fit. Minimum number of points is 3.

        crpix : list of float, None, optional
            Coordinates (1-based) of the reference point for the new FITS WCS.
            When not provided, i.e., when set to `None` (default) the reference
            pixel will be chosen near the center of the bounding box.

        verbose : bool, optional
            Print progress of fits.

        Returns
        -------
        FITS header with all SIP WCS keywords

        Raises
        ------
        ValueError
            If the WCS is not 2D, an exception will be raised. If the specified accuracy
            (both forward and inverse, both rms and maximum) is not achieved an exception
            will be raised.

        Notes
        -----

        Use of this requires a judicious choice of required accuracies. Attempts to use
        higher degrees (~7 or higher) will typically fail due floating point problems
        that arise with high powers.

        """
        if not isinstance(self.output_frame, cf.CelestialFrame):
            raise ValueError(
                "The to_fits_sip method only works with celestial frame transforms")

        if npoints < 8:
            raise ValueError("Number of sampling points is too small. 'npoints' must be >= 8.")

        transform = self.forward_transform

        # Determine reference points.
        if bounding_box is None and self.bounding_box is None:
            raise ValueError("A bounding_box is needed to proceed.")
        if bounding_box is None:
            bounding_box = self.bounding_box

        (xmin, xmax), (ymin, ymax) = bounding_box
        if crpix is None:
            crpix1 = round((xmax + xmin) / 2, 1)
            crpix2 = round((ymax + ymin) / 2, 1)
        else:
            crpix1 = crpix[0] - 1
            crpix2 = crpix[1] - 1

        # check that the bounding box has some reasonable size:
        if (xmax - xmin) < 1 or (ymax - ymin) < 1:
            raise ValueError("Bounding box is too small for fitting a SIP polynomial")

        crval1, crval2 = transform(crpix1, crpix2)
        hdr = fits.Header()
        hdr['naxis'] = 2
        hdr['naxis1'] = int(xmax) + 1
        hdr['naxis2'] = int(ymax) + 1
        hdr['ctype1'] = 'RA---TAN-SIP'
        hdr['ctype2'] = 'DEC--TAN-SIP'
        hdr['CRPIX1'] = crpix1 + 1
        hdr['CRPIX2'] = crpix2 + 1
        hdr['CRVAL1'] = crval1
        hdr['CRVAL2'] = crval2
        hdr['cd1_1'] = 1  # Placeholders for FITS card order, all will change.
        hdr['cd1_2'] = 0
        hdr['cd2_1'] = 0
        hdr['cd2_2'] = 1
        # Now rotate to native system and deproject. Recall that transform
        # expects pixels in the original coordinate system, but the SIP
        # transform is relative to crpix coordinates, thus the initial shift.
        ntransform = ((Shift(crpix1) & Shift(crpix2)) | transform
                      | RotateCelestial2Native(crval1, crval2, 180)
                      | Sky2Pix_TAN())

        # standard sampling:
        u, v = _make_sampling_grid(npoints, bounding_box, crpix=[crpix1, crpix2])
        undist_x, undist_y = ntransform(u, v)

        # Double sampling to check if sampling is sufficient.
        ud, vd = _make_sampling_grid(2 * npoints, bounding_box, crpix=[crpix1, crpix2])
        undist_xd, undist_yd = ntransform(ud, vd)

        # Determine approximate pixel scale in order to compute error threshold
        # from the specified pixel error. Computed at the center of the array.
        x0, y0 = ntransform(0, 0)
        xx, xy = ntransform(1, 0)
        yx, yy = ntransform(0, 1)
        pixarea = np.abs((xx - x0) * (yy - y0) - (xy - y0) * (yx - x0))
        plate_scale = np.sqrt(pixarea)
        max_error = max_pix_error * plate_scale

        # The fitting section.
        fit_poly_x, fit_poly_y, max_resid = _fit_2D_poly(
            ntransform, npoints,
            degree, max_error,
            u, v, undist_x, undist_y,
            ud, vd, undist_xd, undist_yd,
            verbose=verbose
        )

        # The following is necessary to put the fit into the SIP formalism.
        cdmat, sip_poly_x, sip_poly_y = _reform_poly_coefficients(fit_poly_x, fit_poly_y)
        # cdmat = np.array([[fit_poly_x.c1_0.value, fit_poly_x.c0_1.value],
        #                   [fit_poly_y.c1_0.value, fit_poly_y.c0_1.value]])
        det = cdmat[0][0] * cdmat[1][1] - cdmat[0][1] * cdmat[1][0]
        U = ( cdmat[1][1] * undist_x - cdmat[0][1] * undist_y) / det
        V = (-cdmat[1][0] * undist_x + cdmat[0][0] * undist_y) / det
        detd = cdmat[0][0] * cdmat[1][1] - cdmat[0][1] * cdmat[1][0]
        Ud = ( cdmat[1][1] * undist_xd - cdmat[0][1] * undist_yd) / detd
        Vd = (-cdmat[1][0] * undist_xd + cdmat[0][0] * undist_yd) / detd

        if max_inv_pix_error:
            fit_inv_poly_u, fit_inv_poly_v, max_inv_resid = _fit_2D_poly(ntransform,
                                                            npoints, inv_degree,
                                                            max_inv_pix_error,
                                                            U, V, u-U, v-V,
                                                            Ud, Vd, ud-Ud, vd-Vd,
                                                            verbose=verbose)
        pdegree = fit_poly_x.degree
        if pdegree > 1:
            hdr['a_order'] = pdegree
            hdr['b_order'] = pdegree
            _store_2D_coefficients(hdr, sip_poly_x, 'A')
            _store_2D_coefficients(hdr, sip_poly_y, 'B')
            hdr['sipmxerr'] = (max_resid * plate_scale, 'Max diff from GWCS (equiv pix).')
            if max_inv_pix_error:
                hdr['sipiverr'] = (max_inv_resid, 'Max diff for inverse (pixels)')
                _store_2D_coefficients(hdr, fit_inv_poly_u, 'AP', keeplinear=True)
                _store_2D_coefficients(hdr, fit_inv_poly_v, 'BP', keeplinear=True)
            if max_inv_pix_error:
                ipdegree = fit_inv_poly_u.degree
                hdr['ap_order'] = ipdegree
                hdr['bp_order'] = ipdegree
        else:
            hdr['ctype1'] = 'RA---TAN'
            hdr['ctype2'] = 'DEC--TAN'

        hdr['cd1_1'] = cdmat[0][0]
        hdr['cd1_2'] = cdmat[0][1]
        hdr['cd2_1'] = cdmat[1][0]
        hdr['cd2_2'] = cdmat[1][1]
        return hdr

    def to_fits_tab(self, bounding_box=None, bin_ext_name='WCS-TABLE',
                    coord_col_name='coordinates', sampling=1):
        """
        Construct a FITS WCS ``-TAB``-based approximation to the WCS
        in the form of a FITS header and a binary table extension. For the
        description of the FITS WCS ``-TAB`` convention, see
        "Representations of spectral coordinates in FITS" in
        `Greisen, E. W. et al. A&A 446 (2) 747-771 (2006)
        <https://doi.org/10.1051/0004-6361:20053818>`_ .

        Parameters
        ----------
        bounding_box : tuple, optional
            Specifies the range of acceptable values for each input axis.
            The order of the axes is
            `~gwcs.coordinate_frames.CoordinateFrame.axes_order`.
            For two image axes ``bounding_box`` is of the form
            ``((xmin, xmax), (ymin, ymax))``.

        bin_ext_name : str, optional
            Extension name for the `~astropy.io.fits.BinTableHDU` extension.

        coord_col_name : str, optional
            Field name of the coordinate array in the structured array
            stored in `~astropy.io.fits.BinTableHDU` data. This corresponds to
            ``TTYPEi`` field in the FITS header of the binary table extension.

        sampling : float, tuple, optional
            The target "density" of grid nodes per pixel to be used when
            creating the coordinate array for the ``-TAB`` FITS WCS convention.
            It is equal to ``1/step`` where ``step`` is the distance between
            grid nodes in pixels. ``sampling`` can be specified as a single
            number to be used for all axes or as a `tuple` of numbers
            that specify the sampling for each image axis.

        Returns
        -------
        hdr : `~astropy.io.fits.Header`
            Header with WCS-TAB information associated (to be used) with image
            data.

        bin_table : `~astropy.io.fits.BinTableHDU`
            Binary table extension containing the coordinate array.

        Raises
        ------
        ValueError
            When ``bounding_box`` is not defined either through the input
            ``bounding_box`` parameter or this object's ``bounding_box``
            property.

        ValueError
            When ``sampling`` is a `tuple` of length larger than 1 that
            does not match the number of image axes.

        RuntimeError
            If the number of image axes (`~gwcs.WCS.pixel_n_dim`) is larger
            than the number of world axes (`~gwcs.WCS.world_n_dim`).

        """
        if bounding_box is None:
            if self.bounding_box is None:
                raise ValueError(
                    "Need a valid bounding_box to compute the footprint."
                )
            bounding_box = self.bounding_box

        else:
            # validate user-supplied bounding box:
            frames = self.available_frames
            transform_0 = self.get_transform(frames[0], frames[1])
            mutils._BoundingBox.validate(transform_0, bounding_box)

        if self.pixel_n_dim > self.world_n_dim:
            raise RuntimeError(
                "The case when the number of input axes is larger than the "
                "number of output axes is not supported."
            )

        try:
            sampling = np.broadcast_to(sampling, (self.pixel_n_dim, ))
        except ValueError:
            raise ValueError("Number of sampling values either must be 1 "
                             "or it must match the number of pixel axes.")

        # 1D grid coordinates:
        gcrds = []
        cdelt = []
        for (xmin, xmax), s in zip(bounding_box, sampling):
            npix = max(2, 1 + int(np.ceil(abs((xmax - xmin) / s))))
            gcrds.append(np.linspace(xmin, xmax, npix))
            cdelt.append((npix - 1) / (xmax - xmin) if xmin != xmax else 1)

        # n-dim coordinate arrays:
        coord = np.stack(
            self(*np.meshgrid(*gcrds[::-1], indexing='ij')[::-1]),
            axis=-1
        )

        # create header with WCS info:
        hdr = fits.Header()

        for k in range(self.world_n_dim):
            k1 = k + 1
            ct = cf.get_ctype_from_ucd(self.world_axis_physical_types[k])
            if len(ct) > 4:
                raise ValueError("Axis type name too long.")

            hdr['CTYPE{:d}'.format(k1)] = ct + (4 - len(ct)) * '-' + '-TAB'
            hdr['CUNIT{:d}'.format(k1)] = self.world_axis_units[k]
            hdr['PS{:d}_0'.format(k1)] = bin_ext_name
            hdr['PS{:d}_1'.format(k1)] = coord_col_name
            hdr['PV{:d}_3'.format(k1)] = k1
            hdr['CRVAL{:d}'.format(k1)] = 1

            if k < self.pixel_n_dim:
                hdr['CRPIX{:d}'.format(k1)] = gcrds[k][0] + 1
                hdr['PC{0:d}_{0:d}'.format(k1)] = 1.0
                hdr['CDELT{:d}'.format(k1)] = cdelt[k]
            else:
                hdr['CRPIX{:d}'.format(k1)] = 1
                coord = coord[None, :]

        # structured array (data) for binary table HDU:
        arr = np.array(
            [(coord, )],
            dtype=[
                (coord_col_name, np.float64, coord.shape),
            ]
        )

        # create binary table HDU:
        bin_tab = fits.BinTableHDU(arr)
        bin_tab.header['EXTNAME'] = bin_ext_name

        return hdr, bin_tab

    def _calc_approx_inv(self, max_inv_pix_error=5, inv_degree=None, npoints=16):
        """
        Compute polynomial fit for the inverse transformation to be used as
        initial aproximation/guess for the iterative solution.
        """
        self._approx_inverse = None

        try:
            # try to use analytic inverse if available:
            self._approx_inverse = functools.partial(self.backward_transform,
                                                     with_bounding_box=False)
            return
        except (NotImplementedError, KeyError):
            pass

        if not isinstance(self.output_frame, cf.CelestialFrame):
            # The _calc_approx_inv method only works with celestial frame transforms
            return

        # Determine reference points.
        if self.bounding_box is None:
            # A bounding_box is needed to proceed.
            return

        crpix = np.mean(self.bounding_box, axis=1)

        crval1, crval2 = self.forward_transform(*crpix)

        # Rotate to native system and deproject. Set center of the projection
        # transformation to the middle of the bounding box ("image") in order
        # to minimize projection effects across the entire image,
        # thus the initial shift.
        ntransform = ((Shift(crpix[0]) & Shift(crpix[1])) | self.forward_transform
                      | RotateCelestial2Native(crval1, crval2, 180)
                      | Sky2Pix_TAN())

        # standard sampling:
        u, v = _make_sampling_grid(npoints, self.bounding_box, crpix=crpix)
        undist_x, undist_y = ntransform(u, v)

        # Double sampling to check if sampling is sufficient.
        ud, vd = _make_sampling_grid(2 * npoints, self.bounding_box, crpix=crpix)
        undist_xd, undist_yd = ntransform(ud, vd)

        fit_inv_poly_u, fit_inv_poly_v, max_inv_resid = _fit_2D_poly(
            ntransform,
            npoints, None,
            max_inv_pix_error,
            undist_x, undist_y, u, v,
            undist_xd, undist_yd, ud, vd,
            verbose=True
        )

        self._approx_inverse = (RotateCelestial2Native(crval1, crval2, 180) |
                                Sky2Pix_TAN() | Mapping((0, 1, 0, 1)) |
                                (fit_inv_poly_u & fit_inv_poly_v) |
                                (Shift(crpix[0]) & Shift(crpix[1])))


def _fit_2D_poly(ntransform, npoints, degree, max_error,
                 xin, yin, xout, yout,
                 xind, yind, xoutd, youtd,
                 verbose=False):
    """
    Fit a pair of ordinary 2D polynomials to the supplied transform.

    """
    llsqfitter = LinearLSQFitter()

    # The case of one pass with the specified polynomial degree
    if degree is None:
        deglist = range(1, 10)
    elif hasattr(degree, '__iter__'):
        deglist = sorted(map(int, degree))
        if set(deglist).difference(range(1, 10)):
            raise ValueError("Allowed values for SIP degree are [1...9]")
    else:
        degree = int(degree)
        if degree < 1 or degree > 9:
            raise ValueError("Allowed values for SIP degree are [1...9]")
        deglist = [degree]

    prev_max_error = float(np.inf)
    if verbose:
        print(f'maximum_specified_error: {max_error}')
    for deg in deglist:
        poly_x = Polynomial2D(degree=deg)
        poly_y = Polynomial2D(degree=deg)
        fit_poly_x = llsqfitter(poly_x, xin, yin, xout)
        fit_poly_y = llsqfitter(poly_y, xin, yin, yout)
        max_resid = _compute_distance_residual(xout, yout,
                                               fit_poly_x(xin, yin),
                                               fit_poly_y(xin, yin))
        if max_resid > prev_max_error:
            raise RuntimeError('Failed to achieve required error tolerance')
        if verbose:
            print(f'Degree = {deg}, max_resid = {max_resid}')
        if max_resid < max_error:
            # Check to see if double sampling meets error requirement.
            max_resid = _compute_distance_residual(xoutd, youtd,
                                                   fit_poly_x(xind, yind),
                                                   fit_poly_y(xind, yind))
            if verbose:
                print(f'Double sampling check: maximum residual={max_resid}')
            if max_resid < max_error:
                if verbose:
                    print('terminating condition met')
                break
    return fit_poly_x, fit_poly_y, max_resid


def _make_sampling_grid(npoints, bounding_box, crpix):
    step = np.subtract.reduce(bounding_box, axis=1) / (1.0 - npoints)
    crpix = np.asanyarray(crpix)[:, None, None]
    return grid_from_bounding_box(bounding_box, step=step, center=False) - crpix


def _compute_distance_residual(undist_x, undist_y, fit_poly_x, fit_poly_y):
    """
    Compute the distance residuals and return the rms and maximum values.
    """
    dist = np.sqrt((undist_x - fit_poly_x)**2 + (undist_y - fit_poly_y)**2)
    max_resid = dist.max()
    return max_resid


def _reform_poly_coefficients(fit_poly_x, fit_poly_y):
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
    for i in range(0, degree + 1):
        for j in range(0, degree + 1):
            if (i + j > 1) and (i + j < degree + 1):
                old_x = getattr(fit_poly_x, f'c{i}_{j}').value
                old_y = getattr(fit_poly_y, f'c{i}_{j}').value
                newcoeff = np.dot(invcdmat, np.array([[old_x], [old_y]]))
                setattr(sip_poly_x, f'c{i}_{j}', newcoeff[0, 0])
                setattr(sip_poly_y, f'c{i}_{j}', newcoeff[1, 0])

    return cdmat, sip_poly_x, sip_poly_y


def _store_2D_coefficients(hdr, poly_model, coeff_prefix, keeplinear=False):
    """
    Write the polynomial model coefficients to the header.
    """
    mindeg = int(not keeplinear)
    degree = poly_model.degree
    for i in range(0, degree + 1):
        for j in range(0, degree + 1):
            if (i + j) > mindeg and (i + j < degree + 1):
                hdr[f'{coeff_prefix}_{i}_{j}'] = getattr(poly_model, f'c{i}_{j}').value


class Step:
    """
    Represents a ``step`` in the WCS pipeline.

    Parameters
    ----------
    frame : `~gwcs.coordinate_frames.CoordinateFrame`
        A gwcs coordinate frame object.
    transform : `~astropy.modeling.core.Model` or None
        A transform from this step's frame to next step's frame.
        The transform of the last step should be ``None``.
    """
    def __init__(self, frame, transform=None):
        self.frame = frame
        self.transform = transform

    @property
    def frame(self):
        return self._frame

    @frame.setter
    def frame(self, val):
        if not isinstance(val, (cf.CoordinateFrame, str)):
            raise TypeError('"frame" should be an instance of CoordinateFrame or a string.')

        self._frame = val

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, val):
        if val is not None and not isinstance(val, (Model)):
            raise TypeError('"transform" should be an instance of astropy.modeling.Model.')
        self._transform = val

    @property
    def frame_name(self):
        if isinstance(self.frame, str):
            return self.frame
        return self.frame.name

    def __getitem__(self, ind):
        warnings.warn("Indexing a WCS.pipeline step is deprecated. "
                      "Use the `frame` and `transform` attributes instead.", DeprecationWarning)
        if ind not in (0, 1):
            raise IndexError("Allowed inices are 0 (frame) and 1 (transform).")
        if ind == 0:
            return self.frame
        return self.transform

    def __str__(self):
        return f"{self.frame_name}\t {getattr(self.transform, 'name', 'None') or self.transform.__class__.__name__}"

    def __repr__(self):
        return f"Step(frame={self.frame_name}, \
                      transform={getattr(self.transform, 'name', 'None') or self.transform.__class__.__name__})"
