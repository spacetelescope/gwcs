# Licensed under a 3-clause BSD style license - see LICENSE.rst
import functools
import itertools
import warnings
import numpy as np
import numpy.linalg as npla
from scipy import optimize, linalg
from astropy import units as u
from astropy.modeling.core import Model
from astropy.modeling.models import (
    Identity, Mapping, Const1D, Shift, Polynomial2D,
    Sky2Pix_TAN, RotateCelestial2Native
)
from astropy.modeling import projections, fix_inputs
import astropy.io.fits as fits
from astropy.wcs.utils import celestial_frame_to_wcs, proj_plane_pixel_scales

from .api import GWCSAPIMixin
from . import coordinate_frames as cf
from .utils import CoordinateFrameError
from . import utils
from .wcstools import grid_from_bounding_box

try:
    from astropy.modeling.bounding_box import ModelBoundingBox as Bbox
    from astropy.modeling.bounding_box import CompoundBoundingBox
    new_bbox = True
except ImportError:
    from astropy.modeling.utils import _BoundingBox as Bbox
    new_bbox = False


__all__ = ['WCS', 'Step', 'NoConvergence']

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


class _WorldAxisInfo():
    def __init__(self, axis, frame, world_axis_order, cunit, ctype, input_axes):
        """
        A class for holding information about a world axis from an output frame.

        Parameters
        ----------
        axis : int
            Output axis number [in the forward transformation].

        frame : cf.CoordinateFrame
            Coordinate frame to which this axis belongs.

        world_axis_order : int
            Index of this axis in `gwcs.WCS.output_frame.axes_order`

        cunit : str
            Axis unit using FITS convension (``CUNIT``).

        ctype : str
            Axis FITS type (``CTYPE``).

        input_axes : tuple of int
            Tuple of input axis indices contributing to this world axis.

        """
        self.axis = axis
        self.frame = frame
        self.world_axis_order = world_axis_order
        self.cunit = cunit
        self.ctype = ctype
        self.input_axes = input_axes


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

        pipe = []
        for step in self._pipeline:
            if isinstance(step, Step):
                pipe.append(Step(step.frame, step.transform))
            else:
                pipe.append(Step(*step))
        self._pipeline = pipe

    def _initialize_wcs(self, forward_transform, input_frame, output_frame):
        if forward_transform is not None:
            if isinstance(forward_transform, Model):
                if output_frame is None:
                    raise CoordinateFrameError("An output_frame must be specified "
                                               "if forward_transform is a model.")

                _input_frame, inp_frame_obj = self._get_frame_name(input_frame)
                _output_frame, outp_frame_obj = self._get_frame_name(output_frame)
                super(WCS, self).__setattr__(_input_frame, inp_frame_obj)
                super(WCS, self).__setattr__(_output_frame, outp_frame_obj)

                self._pipeline = [(input_frame, forward_transform.copy()),
                                  (output_frame, None)]
            elif isinstance(forward_transform, list):
                for item in forward_transform:
                    if isinstance(item, Step):
                        name, frame_obj = self._get_frame_name(item.frame)
                    else:
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
                raise CoordinateFrameError("An output_frame must be specified "
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
        from_frame : str or `~gwcs.coordinate_frames.CoordinateFrame`
            Initial coordinate frame name of object.
        to_frame : str, or instance of `~gwcs.coordinate_frames.CoordinateFrame`
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
        from_frame : str or `~gwcs.coordinate_frames.CoordinateFrame`
            Initial coordinate frame.
        to_frame : str, or instance of `~gwcs.coordinate_frames.CoordinateFrame`
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
            `~astropy.coordinates.SpectralCoord` object, by using the units of
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
            if new_bbox:
                transform.bounding_box = self.bounding_box
            else:
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
            if new_bbox:
                x1, x2 = self.bounding_box.bounding_box()
            else:
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
            `~astropy.coordinates.SpectralCoord` object, by using the units of
            the output cooridnate frame. Default is `False`.

        Other Parameters
        ----------------
        kwargs : dict
            Keyword arguments to be passed to :py:meth:`numerical_inverse`
            (when defined) or to the iterative invert method.

        Returns
        -------
        result : tuple or value
            Returns a tuple of scalar or array values for each axis. Unless
            ``input_frame.naxes == 1`` when it shall return the value.

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
            `~astropy.coordinates.SpectralCoord` object, by using the units of
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
        >>> assert np.allclose(ra, [5.927628, 5.92757069, 5.92751337]);
        >>> assert np.allclose(dec, [-72.01341247, -72.01341273, -72.013413])

        >>> x, y = w.numerical_inverse(ra, dec)
        >>> assert np.allclose(x, [1.00000005, 2.00000005, 3.00000006]);
        >>> assert np.allclose(y, [1.00000004, 0.99999979, 1.00000015]);

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
        >>> divra, divdec = w([1, 300000, 3], [2, 1000000, 5], with_bounding_box=False)
        >>> assert np.allclose(divra, [5.92762673, 148.21600848, 5.92750827])
        >>> assert np.allclose(divdec, [-72.01339464, -7.80968079, -72.01334172])
        >>> try:  # doctest: +SKIP
        ...     x, y = w.numerical_inverse(divra, divdec, maxiter=20,
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
        area = np.abs(0.5 * ((l4 - l2) * (np.sin(phi1) - np.sin(phi3)) +
                             (l1 - l3) * (np.sin(phi2) - np.sin(phi4))))
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
                inddiv = np.array(bad, dtype=int)
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
        to_frame : str, or instance of `~gwcs.coordinate_frames.CoordinateFrame`
            Coordinate frame into which to transform.
        args : float or array-like
            Inputs in ``from_frame``, separate inputs for each dimension.
        output_with_units : bool
            If ``True`` - returns a `~astropy.coordinates.SkyCoord` or
            `~astropy.coordinates.SpectralCoord` object.
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
        frame : str or `~gwcs.coordinate_frames.CoordinateFrame`
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
        `~gwcs.coordinate_frames.CoordinateFrame` can be passed (e.g., the
        `input_frame` or `output_frame` attribute). This frame is never
        modified.

        Parameters
        ----------
        input_frame : str or `~gwcs.coordinate_frames.CoordinateFrame`
            Coordinate frame at start of new transform
        transform : `~astropy.modeling.Model`
            New transform to be inserted in the pipeline
        output_frame: str or `~gwcs.coordinate_frames.CoordinateFrame`
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

        if new_bbox:
            return bb
        else:
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
                if new_bbox:
                    if isinstance(value, CompoundBoundingBox):
                        bbox = CompoundBoundingBox.validate(transform_0, value, order='F')
                    else:
                        bbox = Bbox.validate(transform_0, value, order='F')
                else:
                    Bbox.validate(transform_0, value)
            except Exception:
                raise

            if new_bbox:
                transform_0.bounding_box = bbox
            else:
                # get the sorted order of axes' indices
                axes_ind = self._get_axes_indices()
                if transform_0.n_inputs == 1:
                    transform_0.bounding_box = value
                else:
                    # The axes in bounding_box in modeling follow python order
                    #transform_0.bounding_box = np.array(value)[axes_ind][::-1]
                    transform_0.bounding_box = [value[ind] for ind in axes_ind][::-1]

        self.set_transform(frames[0], frames[1], transform_0)

    def attach_compound_bounding_box(self, cbbox, selector_args):
        if new_bbox:
            frames = self.available_frames
            transform_0 = self.get_transform(frames[0], frames[1])

            self.bounding_box = CompoundBoundingBox.validate(transform_0, cbbox, selector_args=selector_args,
                                                             order='F')
        else:
            raise NotImplementedError('Compound bounding box is not supported for your version of astropy')

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
            ``prop: bounding_box``
        center : bool
            If `True` use the center of the pixel, otherwise use the corner.
        axis_type : str
            A supported ``output_frame.axes_type`` or ``"all"`` (default).
            One of [``'spatial'``, ``'spectral'``, ``'temporal'``] or a custom type.

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
            Keyword arguments with fixed values corresponding to ``self.selector``.

        Returns
        -------
        new_wcs : `WCS`
            A new unique WCS corresponding to the values in ``fixed``.

        Examples
        --------
        >>> w = WCS(pipeline, selector={"spectral_order": [1, 2]}) # doctest: +SKIP
        >>> new_wcs = w.set_inputs(spectral_order=2) # doctest: +SKIP
        >>> new_wcs.inputs # doctest: +SKIP
            ("x", "y")

        """
        new_pipeline = []
        step0 = self.pipeline[0]
        new_transform = fix_inputs(step0[1], fixed)
        new_pipeline.append((step0[0], new_transform))
        new_pipeline.extend(self.pipeline[1:])
        return self.__class__(new_pipeline)

    def to_fits_sip(self, bounding_box=None, max_pix_error=0.25, degree=None,
                    max_inv_pix_error=0.25, inv_degree=None,
                    npoints=32, crpix=None, projection='TAN',
                    verbose=False):
        """
        Construct a SIP-based approximation to the WCS for the axes
        corresponding to the `~gwcs.coordinate_frames.CelestialFrame`
        in the form of a FITS header.

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
            a nominal plate scale. Ignored when ``degree`` is an integer or
            a list with a single degree.

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

        max_inv_pix_error : float, optional
            Maximum allowed inverse error over the domain of the pixel array
            in pixel units. If None, no inverse is generated. Ignored when
            ``degree`` is an integer or a list with a single degree.

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
            pixel will be chosen near the center of the bounding box for axes
            corresponding to the celestial frame.

        projection : str, `~astropy.modeling.projections.Pix2SkyProjection`, optional
            Projection to be used for the created FITS WCS. It can be specified
            as a string of three characters specifying a FITS projection code
            from Table 13 in
            `Representations of World Coordinates in FITS \
            <https://doi.org/10.1051/0004-6361:20021326>`_
            (Paper I), Greisen, E. W., and Calabretta, M. R., A & A, 395,
            1061-1075, 2002. Alternatively, it can be an instance of one of the
            `astropy's Pix2Sky_* <https://docs.astropy.org/en/stable/modeling/\
            reference_api.html#module-astropy.modeling.projections>`_
            projection models inherited from
            :py:class:`~astropy.modeling.projections.Pix2SkyProjection`.

        verbose : bool, optional
            Print progress of fits.

        Returns
        -------
        FITS header with all SIP WCS keywords

        Raises
        ------
        ValueError
            If the WCS is not at least 2D, an exception will be raised. If the
            specified accuracy (both forward and inverse, both rms and maximum)
            is not achieved an exception will be raised.

        Notes
        -----

        Use of this requires a judicious choice of required accuracies.
        Attempts to use higher degrees (~7 or higher) will typically fail due
        to floating point problems that arise with high powers.

        """
        _, _, celestial_group = self._separable_groups(detect_celestial=True)
        if celestial_group is None:
            raise ValueError("The to_fits_sip requires an output celestial frame.")

        hdr = self._to_fits_sip(
            celestial_group=celestial_group,
            keep_axis_position=False,
            bounding_box=bounding_box,
            max_pix_error=max_pix_error,
            degree=degree,
            max_inv_pix_error=max_inv_pix_error,
            inv_degree=inv_degree,
            npoints=npoints,
            crpix=crpix,
            projection=projection,
            matrix_type='CD',
            verbose=verbose
        )

        return hdr

    def _to_fits_sip(self, celestial_group, keep_axis_position,
                     bounding_box, max_pix_error, degree,
                     max_inv_pix_error, inv_degree,
                     npoints, crpix, projection, matrix_type,
                     verbose):
        r"""
        Construct a SIP-based approximation to the WCS for the axes
        corresponding to the `~gwcs.coordinate_frames.CelestialFrame`
        in the form of a FITS header.

        The default mode in using this attempts to achieve roughly 0.25 pixel
        accuracy over the whole image.

        Below we describe only parameters additional to the ones explained for
        `to_fits_sip`.

        Other Parameters
        ----------------
        frame : gwcs.coordinate_frames.CelestialFrame
            A celestial frame.

        celestial_group : list of ``_WorldAxisInfo``
            A group of two celestial axes to be represented using standard
            image FITS WCS and maybe ``-SIP`` polynomials.

        keep_axis_position : bool
            This parameter controls whether to keep/preserve output axes
            indices in this WCS object when creating FITS WCS and create a FITS
            header with ``CTYPE`` axes indices preserved from the ``frame``
            object or whether to reset the indices of output celestial axes
            to 1 and 2 with ``CTYPE1``, ``CTYPE2``. Default is `False`.

            .. warning::
                Returned header will have both ``NAXIS`` and ``WCSAXES`` set
                to 2. If ``max(axes_mapping) > 2`` this will lead to an invalid
                WCS. It is caller's responsibility to adjust NAXIS to a valid
                value.

            .. note::
                The ``lon``/``lat`` order is still preserved regardless of this
                setting.

        matrix_type : {'CD', 'PC-CDELT1', 'PC-SUM1', 'PC-DET1', 'PC-SCALE'}
            Specifies formalism (``PC`` or ``CD``) to be used for the linear
            transformation matrix and normalization for the ``PC`` matrix
            *when non-linear polynomial terms are not required to achieve
            requested accuracy*.

            .. note:: ``CD`` matrix is always used when requested SIP
                approximation accuracy requires non-linear terms (when
                ``CTYPE`` ends in ``-SIP``). This parameter is ignored when
                non-linear polynomial terms are used.

            - ``'CD'``: use ``CD`` matrix;

            - ``'PC-CDELT1'``: set ``PC=CD`` and ``CDELTi=1``. This is the
              behavior of `~astropy.wcs.WCS.to_header` method;

            - ``'PC-SUM1'``: normalize ``PC`` matrix such that sum
              of its squared elements is 1: :math:`\Sigma PC_{ij}^2=1`;

            - ``'PC-DET1'``: normalize ``PC`` matrix such that :math:`|\det(PC)|=1`;

            - ``'PC-SCALE'``: normalize ``PC`` matrix such that ``CDELTi``
              are estimates of the linear pixel scales.

        Returns
        -------
        FITS header with all SIP WCS keywords

        Raises
        ------
        ValueError
            If the WCS is not at least 2D, an exception will be raised. If the
            specified accuracy (both forward and inverse, both rms and maximum)
            is not achieved an exception will be raised.

        """
        if isinstance(matrix_type, str):
            matrix_type = matrix_type.upper()

        if matrix_type not in ['CD', 'PC-CDELT1', 'PC-SUM1', 'PC-DET1', 'PC-SCALE']:
            raise ValueError(f"Unsupported 'matrix_type' value: {repr(matrix_type)}.")

        if npoints < 8:
            raise ValueError("Number of sampling points is too small. 'npoints' must be >= 8.")

        if isinstance(projection, str):
            projection = projection.upper()
            try:
                sky2pix_proj = getattr(projections, f'Sky2Pix_{projection}')(name=projection)
            except AttributeError:
                raise ValueError("Unsupported FITS WCS sky projection: {projection}")

        elif isinstance(projection, projections.Sky2PixProjection):
            sky2pix_proj = projection
            projection = projection.name
            if not projection or not isinstance(projection, str) or len(projection) != 3:
                raise ValueError("Unsupported FITS WCS sky projection: {sky2pix_proj}")
            try:
                getattr(projections, f'Sky2Pix_{projection}')()
            except AttributeError:
                raise ValueError("Unsupported FITS WCS sky projection: {projection}")

        else:
            raise TypeError(
                "'projection' must be either a FITS WCS string projection code "
                "or an instance of astropy.modeling.projections.Pix2SkyProjection.")

        frame = celestial_group[0].frame

        lon_axis = frame.axes_order[0]
        lat_axis = frame.axes_order[1]

        # identify input axes:
        input_axes = []
        for wax in celestial_group:
            input_axes.extend(wax.input_axes)
        input_axes = sorted(set(input_axes))

        if len(input_axes) != 2:
            raise ValueError("Only CelestialFrame that correspond to two "
                             "input axes are supported.")

        # Axis number for FITS axes.
        # iax? - image axes; nlon, nlat - celestial axes:
        if keep_axis_position:
            nlon = lon_axis + 1
            nlat = lat_axis + 1
            iax1, iax2 = (i + 1 for i in input_axes)
        else:
            nlon, nlat = (1, 2) if lon_axis < lat_axis else (2, 1)
            iax1 = 1
            iax2 = 2

        # Determine reference points.
        if bounding_box is None and self.bounding_box is None:
            raise ValueError("A bounding_box is needed to proceed.")
        if bounding_box is None:
            bounding_box = self.bounding_box

        bb_center = np.mean(bounding_box, axis=1)

        fixi_dict = {
            k: bb_center[k] for k in set(range(self.pixel_n_dim)).difference(input_axes)
        }

        # transform = fix_inputs(self.forward_transform, fixi_dict)
        # This is a workaround to the bug in https://github.com/astropy/astropy/issues/11360
        # Once that bug is fixed, the code below can be replaced with fix_inputs
        # statement commented out immediately above.
        transform = _fix_transform_inputs(self.forward_transform, fixi_dict)

        transform = transform | Mapping((lon_axis, lat_axis),
                                        n_inputs=self.forward_transform.n_outputs)

        (xmin, xmax) = bounding_box[input_axes[0]]
        (ymin, ymax) = bounding_box[input_axes[1]]

        # 0-based crpix:
        if crpix is None:
            crpix1 = round(bb_center[input_axes[0]], 1)
            crpix2 = round(bb_center[input_axes[1]], 1)
        else:
            crpix1 = crpix[0] - 1
            crpix2 = crpix[1] - 1

        # check that the bounding box has some reasonable size:
        if (xmax - xmin) < 1 or (ymax - ymin) < 1:
            raise ValueError("Bounding box is too small for fitting a SIP polynomial")

        lon, lat = transform(crpix1, crpix2)

        # Now rotate to native system and deproject. Recall that transform
        # expects pixels in the original coordinate system, but the SIP
        # transform is relative to crpix coordinates, thus the initial shift.
        ntransform = ((Shift(crpix1) & Shift(crpix2)) | transform
                      | RotateCelestial2Native(lon, lat, 180)
                      | sky2pix_proj)

        # standard sampling:
        u, v = _make_sampling_grid(
            npoints,
            tuple(bounding_box[k] for k in input_axes),
            crpix=[crpix1, crpix2]
        )
        undist_x, undist_y = ntransform(u, v)

        # Double sampling to check if sampling is sufficient.
        ud, vd = _make_sampling_grid(
            2 * npoints,
            tuple(bounding_box[k] for k in input_axes),
            crpix=[crpix1, crpix2]
        )
        undist_xd, undist_yd = ntransform(ud, vd)

        # Determine approximate pixel scale in order to compute error threshold
        # from the specified pixel error. Computed at the center of the array.
        x0, y0 = ntransform(0, 0)
        xx, xy = ntransform(1, 0)
        yx, yy = ntransform(0, 1)
        pixarea = np.abs((xx - x0) * (yy - y0) - (xy - y0) * (yx - x0))
        plate_scale = np.sqrt(pixarea)

        # The fitting section.
        if verbose:
            print("\nFitting forward SIP ...")
        fit_poly_x, fit_poly_y, max_resid = _fit_2D_poly(
            degree, max_pix_error, plate_scale,
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
            if verbose:
                print("\nFitting inverse SIP ...")
            fit_inv_poly_u, fit_inv_poly_v, max_inv_resid = _fit_2D_poly(
                inv_degree,
                max_inv_pix_error, 1,
                U, V, u-U, v-V,
                Ud, Vd, ud-Ud, vd-Vd,
                verbose=verbose
        )

        # create header with WCS info:
        w = celestial_frame_to_wcs(frame.reference_frame, projection=projection)
        w.wcs.crval = [lon, lat]
        w.wcs.crpix = [crpix1 + 1, crpix2 + 1]
        w.wcs.pc = cdmat if nlon < nlat else cdmat[::-1]
        w.wcs.set()
        hdr = w.to_header(True)

        # data array info:
        hdr.insert(0, ('NAXIS', 2, 'number of array dimensions'))
        hdr.insert(1, (f'NAXIS{iax1:d}', int(xmax) + 1))
        hdr.insert(2, (f'NAXIS{iax2:d}', int(ymax) + 1))
        assert len(hdr['NAXIS*']) == 3

        # list of celestial axes related keywords:
        cel_kwd = ['CRVAL', 'CTYPE', 'CUNIT']

        # Add SIP info:
        if fit_poly_x.degree > 1:
            mat_kind = 'CD'
            # CDELT is not used with CD matrix (PC->CD later):
            del hdr['CDELT?']

            hdr['CTYPE1'] = hdr['CTYPE1'].strip() + '-SIP'
            hdr['CTYPE2'] = hdr['CTYPE2'].strip() + '-SIP'
            hdr['A_ORDER'] = fit_poly_x.degree
            hdr['B_ORDER'] = fit_poly_x.degree
            _store_2D_coefficients(hdr, sip_poly_x, 'A')
            _store_2D_coefficients(hdr, sip_poly_y, 'B')
            hdr['sipmxerr'] = (max_resid, 'Max diff from GWCS (equiv pix).')

            if max_inv_pix_error:
                hdr['AP_ORDER'] = fit_inv_poly_u.degree
                hdr['BP_ORDER'] = fit_inv_poly_u.degree
                _store_2D_coefficients(hdr, fit_inv_poly_u, 'AP', keeplinear=True)
                _store_2D_coefficients(hdr, fit_inv_poly_v, 'BP', keeplinear=True)
                hdr['sipiverr'] = (max_inv_resid, 'Max diff for inverse (pixels)')

        else:
            if matrix_type.startswith('PC'):
                mat_kind = 'PC'
                cel_kwd.append('CDELT')

                if matrix_type == 'PC-CDELT1':
                    cdelt = [1.0, 1.0]

                elif matrix_type == 'PC-SUM1':
                    norm = np.sqrt(np.sum(w.wcs.pc**2))
                    cdelt = [norm, norm]

                elif matrix_type == 'PC-DET1':
                    det_pc = np.linalg.det(w.wcs.pc)
                    norm = np.sqrt(np.abs(det_pc))
                    cdelt = [norm, np.sign(det_pc) * norm]

                elif matrix_type == 'PC-SCALE':
                    cdelt = proj_plane_pixel_scales(w)

                for i in range(1, 3):
                    s = cdelt[i - 1]
                    hdr[f'CDELT{i}'] = s
                    for j in range(1, 3):
                        pc_kwd = f'PC{i}_{j}'
                        if pc_kwd in hdr:
                            hdr[pc_kwd] = w.wcs.pc[i - 1, j - 1] / s

            else:
                mat_kind = 'CD'
                del hdr['CDELT?']

            hdr['sipmxerr'] = (max_resid, 'Max diff from GWCS (equiv pix).')

        # Construct CD matrix while remapping input axes.
        # We do not update comments to typical comments for CD matrix elements
        # (such as 'partial of second axis coordinate w.r.t. y'), because
        # when input frame has number of axes > 2, then imaging
        # axes arbitrary.
        old_nlon, old_nlat = (1, 2) if nlon < nlat else (2, 1)

        # Remap input axes (CRPIX) and output axes-related parameters
        # (CRVAL, CUNIT, CTYPE, CD/PC). This has to be done in two steps to avoid
        # name conflicts (i.e., swapping CRPIX1<->CRPIX2).

        # remap input axes:
        axis_rename = {}
        if iax1 != 1:
            axis_rename['CRPIX1'] = f'CRPIX{iax1}'
        if iax2 != 2:
            axis_rename['CRPIX2'] = f'CRPIX{iax2}'

        # CP/PC matrix:
        axis_rename[f'PC{old_nlon}_1'] = f'{mat_kind}{nlon}_{iax1}'
        axis_rename[f'PC{old_nlon}_2'] = f'{mat_kind}{nlon}_{iax2}'
        axis_rename[f'PC{old_nlat}_1'] = f'{mat_kind}{nlat}_{iax1}'
        axis_rename[f'PC{old_nlat}_2'] = f'{mat_kind}{nlat}_{iax2}'

        # remap celestial axes keywords:
        for kwd in cel_kwd:
            for iold, inew in [(1, nlon), (2, nlat)]:
                if iold != inew:
                    axis_rename[f'{kwd:s}{iold:d}'] = f'{kwd:s}{inew:d}'

        # construct new header cards with remapped axes:
        new_cards = []
        for c in hdr.cards:
            if c[0] in axis_rename:
                c = fits.Card(keyword=axis_rename[c.keyword], value=c.value, comment=c.comment)
            new_cards.append(c)

        hdr = fits.Header(new_cards)
        hdr['WCSAXES'] = 2
        hdr.insert('WCSAXES', ('WCSNAME', f'{self.output_frame.name}'), after=True)

        # for PC matrix formalism, set diagonal elements to 0 if necessary
        # (by default, in PC formalism, diagonal matrix elements by default
        # are 0):
        if mat_kind == 'PC':
            if nlon not in [iax1, iax2]:
                hdr.insert(
                    f'{mat_kind}{nlon}_{iax2}',
                    (f'{mat_kind}{nlon}_{nlon}', 0.0,
                     'Coordinate transformation matrix element')
                )
            if nlat not in [iax1, iax2]:
                hdr.insert(
                    f'{mat_kind}{nlat}_{iax2}',
                    (f'{mat_kind}{nlat}_{nlat}', 0.0,
                     'Coordinate transformation matrix element')
                )

        return hdr

    def _separable_groups(self, detect_celestial):
        """
        This method finds sets (groups) of separable axes - axes that are
        dependent on other axes within a set/group but do not depend on
        axes from other groups. In other words, axes from different
        groups are separable.

        Parameters
        ----------
        detect_celestial : bool
            If `True`, will return, as the third return value, the group of
            celestial axes separately from all other (groups of) axes. If
            no celestial frame is detected, then return value for the
            celestial axes group will be set to `None`.

        Returns
        -------
        axes_groups : list of lists of ``_WorldAxisInfo``
            Each inner list represents a group of non-separable (among
            themselves) axes and each axis in a group is independent of axes
            in *other* groups. Each axis in a group is represented through
            the `_WorldAxisInfo` class used to store relevant information about
            an axis. When ``detect_celestial`` is set to `True`, celestial axes
            group is not included in this list.

        world_axes : list of ``_WorldAxisInfo``
            A flattened version of ``axes_groups``. Even though it is not
            difficult to flatten ``axes_groups``, this list is a by-product
            of other checks and returned here for efficiency. When
            ``detect_celestial`` is set to `True`, celestial axes
            group is not included in this list.

        celestial_group : list of ``_WorldAxisInfo``
            A group of two celestial axes. This group is returned *only when*
            ``detect_celestial`` is set to `True`.

        """
        def find_frame(axis_number):
            for frame in frames:
                if axis_number in frame.axes_order:
                    return frame
            else:
                raise RuntimeError("Encountered an output axes that does not "
                                   "belong to any output coordinate frames.")

        # use correlation matrix to find separable axes:
        corr_mat = self.axis_correlation_matrix
        axes_sets = [set(np.flatnonzero(r)) for r in corr_mat.T]

        k = 0
        while len(axes_sets) - 1 > k:
            for m in range(len(axes_sets) - 1, k, -1):
                if axes_sets[k].isdisjoint(axes_sets[m]):
                    continue
                axes_sets[k] = axes_sets[k].union(axes_sets[m])
                del axes_sets[m]
            k += 1

        # create a mapping of output axes to input/image axes groups:
        mapping = {k: tuple(np.flatnonzero(r)) for k, r in enumerate(corr_mat)}

        axes_groups = []
        world_axes = []  # flattened version of axes_groups
        input_axes = []  # all input axes

        if isinstance(self.output_frame, cf.CompositeFrame):
            frames = self.output_frame.frames
        else:
            frames = [self.output_frame]

        celestial_group = None

        # identify which separable group of axes belong
        for s in axes_sets:
            axis_info_group = []  # group of separable output axes info

            # Find the frame to which the first axis in the group belongs.
            # Most likely this frame will be the frame of all other axes in
            # this group; if not, we will update it later.
            s = sorted(s)
            frame = find_frame(s[0])

            celestial = (detect_celestial and len(s) == 2 and
                         len(frame.axes_order) == 2 and
                         isinstance(frame, cf.CelestialFrame))

            for axno in s:
                if axno not in frame.axes_order:
                    frame = find_frame(axno)
                    celestial = False  # Celestial axes must belong to the same frame

                # index of the axis in this frame's
                fidx = frame.axes_order.index(axno)
                if hasattr(frame.unit[fidx], 'get_format_name'):
                    cunit = frame.unit[fidx].get_format_name(u.format.Fits).upper()
                else:
                    cunit = ''

                axis_info = _WorldAxisInfo(
                    axis=axno,
                    frame=frame,
                    world_axis_order=self.output_frame.axes_order.index(axno),
                    cunit=cunit,
                    ctype=cf.get_ctype_from_ucd(self.world_axis_physical_types[axno]),
                    input_axes=mapping[axno]
                )
                axis_info_group.append(axis_info)
                input_axes.extend(mapping[axno])

            world_axes.extend(axis_info_group)
            if celestial:
                celestial_group = axis_info_group
            else:
                axes_groups.append(axis_info_group)

        # sanity check:
        input_axes = set(sum((ax.input_axes for ax in world_axes),
                             world_axes[0].input_axes.__class__()))
        n_inputs = len(input_axes)

        if (n_inputs != self.pixel_n_dim or
            max(input_axes) + 1 != n_inputs or
            min(input_axes) < 0):
            raise ValueError("Input axes indices are inconsistent with the "
                             "forward transformation.")

        if detect_celestial:
            return axes_groups, world_axes, celestial_group
        else:
            return axes_groups, world_axes

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
            Extension name for the `~astropy.io.fits.BinTableHDU` HDU for those
            axes groups that will be converted using FITW WCS' ``-TAB``
            algorith. Extension version will be determined automatically
            based on the number of separable group of axes.

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

        bin_table_hdu : `~astropy.io.fits.BinTableHDU`
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
            If the number of image axes (``~gwcs.WCS.pixel_n_dim``) is larger
            than the number of world axes (``~gwcs.WCS.world_n_dim``).

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
            Bbox.validate(transform_0, bounding_box)

        if self.forward_transform.n_inputs == 1:
            bounding_box = [bounding_box]

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

        _, world_axes = self._separable_groups(detect_celestial=False)

        hdr, bin_table_hdu = self._to_fits_tab(
            hdr=None,
            world_axes_group=world_axes,
            use_cd=False,
            bounding_box=bounding_box,
            bin_ext=bin_ext_name,
            coord_col_name=coord_col_name,
            sampling=sampling
        )

        return hdr, bin_table_hdu

    def to_fits(self, bounding_box=None, max_pix_error=0.25, degree=None,
                max_inv_pix_error=0.25, inv_degree=None, npoints=32,
                crpix=None, projection='TAN', bin_ext_name='WCS-TABLE',
                coord_col_name='coordinates', sampling=1, verbose=False):
        """
        Construct a FITS WCS ``-TAB``-based approximation to the WCS
        in the form of a FITS header and a binary table extension. For the
        description of the FITS WCS ``-TAB`` convention, see
        "Representations of spectral coordinates in FITS" in
        `Greisen, E. W. et al. A&A 446 (2) 747-771 (2006)
        <https://doi.org/10.1051/0004-6361:20053818>`_ . If WCS contains
        celestial frame, PC/CD formalism will be used for the celestial axes.

        .. note::
            SIP distortion fitting requires that the WCS object has only two
            celestial axes. When WCS does not contain celestial axes,
            SIP fitting parameters (``max_pix_error``, ``degree``,
            ``max_inv_pix_error``, ``inv_degree``, and ``projection``)
            are ignored. When a WCS, in addition to celestial
            frame, contains other types of axes, SIP distortion fitting is
            disabled (ony linear terms are fitted for celestial frame).

        Parameters
        ----------
        bounding_box : tuple, optional
            Specifies the range of acceptable values for each input axis.
            The order of the axes is
            `~gwcs.coordinate_frames.CoordinateFrame.axes_order`.
            For two image axes ``bounding_box`` is of the form
            ``((xmin, xmax), (ymin, ymax))``.

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

            .. note::
                When WCS object has When ``degree`` is `None` and the WCS object has

        max_inv_pix_error : float, optional
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
            pixel will be chosen near the center of the bounding box for axes
            corresponding to the celestial frame.

        projection : str, `~astropy.modeling.projections.Pix2SkyProjection`, optional
            Projection to be used for the created FITS WCS. It can be specified
            as a string of three characters specifying a FITS projection code
            from Table 13 in
            `Representations of World Coordinates in FITS \
            <https://doi.org/10.1051/0004-6361:20021326>`_
            (Paper I), Greisen, E. W., and Calabretta, M. R., A & A, 395,
            1061-1075, 2002. Alternatively, it can be an instance of one of the
            `astropy's Pix2Sky_* <https://docs.astropy.org/en/stable/modeling/\
            reference_api.html#module-astropy.modeling.projections>`_
            projection models inherited from
            :py:class:`~astropy.modeling.projections.Pix2SkyProjection`.

        bin_ext_name : str, optional
            Extension name for the `~astropy.io.fits.BinTableHDU` HDU for those
            axes groups that will be converted using FITW WCS' ``-TAB``
            algorith. Extension version will be determined automatically
            based on the number of separable group of axes.

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

        verbose : bool, optional
            Print progress of fits.

        Returns
        -------
        hdr : `~astropy.io.fits.Header`
            Header with WCS-TAB information associated (to be used) with image
            data.

        hdulist : a list of `~astropy.io.fits.BinTableHDU`
            A Python list of binary table extensions containing the coordinate
            array for TAB extensions; one extension per separable axes group.

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
            If the number of image axes (``~gwcs.WCS.pixel_n_dim``) is larger
            than the number of world axes (``~gwcs.WCS.world_n_dim``).

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
            Bbox.validate(transform_0, bounding_box)

        if self.forward_transform.n_inputs == 1:
            bounding_box = [bounding_box]

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

        world_axes_groups, _, celestial_group = self._separable_groups(
            detect_celestial=True
        )

        # Find celestial axes group and treat it separately from other axes:
        if celestial_group:
            # if world_axes_groups is empty, then we have only celestial axes
            # and so we can allow arbitrary degree for SIP. When there are
            # other axes types present, issue a warning and set 'degree' to 1
            # because use of SIP when world_n_dim > 2 currently is not supported by
            # astropy.wcs.WCS - see https://github.com/astropy/astropy/pull/11452
            if world_axes_groups and (degree is None or np.max(degree) != 2):
                if degree is not None:
                    warnings.warn(
                        "SIP distortion is not supported when the number\n"
                        "of axes in WCS is larger than 2. Setting 'degree'\n"
                        "to 1 and 'max_inv_pix_error' to None."
                    )
                degree = 1
                max_inv_pix_error = None

            hdr = self._to_fits_sip(
                celestial_group=celestial_group,
                keep_axis_position=True,
                bounding_box=bounding_box,
                max_pix_error=max_pix_error,
                degree=degree,
                max_inv_pix_error=max_inv_pix_error,
                inv_degree=inv_degree,
                npoints=npoints,
                crpix=crpix,
                projection=projection,
                matrix_type='PC-CDELT1',
                verbose=verbose
            )
            use_cd = 'A_ORDER' in hdr

        else:
            use_cd = False
            hdr = fits.Header()
            hdr['NAXIS'] = 0
            hdr['WCSAXES'] = 0

        # now handle non-celestial axes using -TAB convention for each
        # separable axes group:
        hdulist = []
        for extver0, world_axes_group in enumerate(world_axes_groups):
            # For each subset of separable axes call _to_fits_tab to
            # convert that group to a single Bin TableHDU with a
            # coordinate array for this group of axes:
            hdr, bin_table_hdu = self._to_fits_tab(
                hdr=hdr,
                world_axes_group=world_axes_group,
                use_cd=use_cd,
                bounding_box=bounding_box,
                bin_ext=(bin_ext_name, extver0 + 1),
                coord_col_name=coord_col_name,
                sampling=sampling
            )
            hdulist.append(bin_table_hdu)

        hdr.add_comment('FITS WCS created by approximating a gWCS')

        return hdr, hdulist


    def _to_fits_tab(self, hdr, world_axes_group, use_cd, bounding_box,
                     bin_ext, coord_col_name, sampling):
        """
        Construct a FITS WCS ``-TAB``-based approximation to the WCS
        in the form of a FITS header and a binary table extension. For the
        description of the FITS WCS ``-TAB`` convention, see
        "Representations of spectral coordinates in FITS" in
        `Greisen, E. W. et al. A&A 446 (2) 747-771 (2006)
        <https://doi.org/10.1051/0004-6361:20053818>`_ .

        Below we describe only parameters additional to the ones explained for
        `to_fits_tab`.

        .. warn::
            For this helper function, parameters ``bounding_box`` and
            ``sampling`` (when provided as a tuple) are expected to have
            the same length as the number of input axes in the *full* WCS
            object. That is, the number of elements in ``bounding_box`` and
            ``sampling`` is not be affected by ``ignore_axes``.

        Other Parameters
        ----------------
        hdr : astropy.io.fits.Header, None
            The first time this function is called, ``hdr`` should be set to
            `None` or be an empty :py:class:`~astropy.io.fits.Header` object.
            On subsequent calls, updated header from the previous iteration
            should be provided.

        world_axes_group : tuple of dict
            A list of world axes to represent through FITS' -TAB convention.
            This is a list of dictionaries with each dicti

        axes_mapping : dict
            A dictionary that maps output axis index to a tuple of input
            axis indices. In a typical scenario of two input image axes
            and two output celestial axes for a FITS-like WCS,
            this dictionary would look like ``{0: (0, 1), 1: (0, 1)}``
            with the two non-separable input axes.

        fix_axes : dict
            A dictionary containing as keys image axes' indices to be
            fixed and as values - the values to which inputs should be kept
            fixed. For example, this dictionary may be used to indicate the
            celestial axes that should not be included into -TAB approximation
            because they will be approximated using -SIP.

        use_cd : bool
            When `True` - CD-matrix formalism will be used instead of the
            PC-matrix formalism.

        bin_ext : str, tuple of str and int
            Extension name  and optionally version for the
            `~astropy.io.fits.BinTableHDU` HDU. When only a string extension
            name is provided, extension version will be set to 1.
            When ``bin_ext`` is a tuple, first element should be extension
            name and the second element is a positive integer extension version
            number.

        Returns
        -------
        hdr : `~astropy.io.fits.Header`
            Header with WCS-TAB information associated (to be used) with image
            data.

        bin_table_hdu : `~astropy.io.fits.BinTableHDU`
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

        ValueError
            When extension version is smaller than 1.

        TypeError

        RuntimeError
            If the number of image axes (``~gwcs.WCS.pixel_n_dim``) is larger
            than the number of world axes (``~gwcs.WCS.world_n_dim``).

        """
        if isinstance(bin_ext, str):
            bin_ext = (bin_ext, 1)

        if new_bbox:
            if isinstance(bounding_box, Bbox):
                bounding_box = bounding_box.bounding_box(order='F')
            if isinstance(bounding_box, list):
                for index, bbox in enumerate(bounding_box):
                    if isinstance(bbox, Bbox):
                        bounding_box[index] = bbox.bounding_box(order='F')

        # identify input axes:
        input_axes = []
        world_axes_idx = []
        for ax in world_axes_group:
            world_axes_idx.append(ax.axis)
            input_axes.extend(ax.input_axes)
        input_axes = sorted(set(input_axes))
        n_inputs = len(input_axes)
        n_outputs = len(world_axes_group)
        world_axes_idx.sort()

        # Create initial header and deal with non-degenerate axes
        if hdr is None:
            hdr = fits.Header()
            hdr['NAXIS'] = n_inputs, 'number of array dimensions'
            hdr['WCSAXES'] = n_outputs
            hdr.insert('WCSAXES', ('WCSNAME', f'{self.output_frame.name}'), after=True)

        else:
            hdr['NAXIS'] += n_inputs
            hdr['WCSAXES'] += n_outputs

        # see what axes have been already populated in the header:
        used_hdr_axes = []
        for v in hdr['naxis*'].keys():
            try:
                used_hdr_axes.append(int(v.split('NAXIS')[1]) - 1)
            except ValueError:
                continue

        degenerate_axis_start = max(
            self.pixel_n_dim + 1,
            max(used_hdr_axes) + 1 if used_hdr_axes else 1
        )

        # Deal with non-degenerate axes and add NAXISi to the header:
        offset = hdr.index('NAXIS')

        for iax in input_axes:
            iiax = int(np.searchsorted(used_hdr_axes, iax))
            hdr.insert(iiax + offset + 1, (f'NAXIS{iax + 1:d}', int(max(bounding_box[iiax])) + 1))

        # 1D grid coordinates:
        gcrds = []
        cdelt = []
        bb = [bounding_box[k] for k in input_axes]
        for (xmin, xmax), s in zip(bb, sampling):
            npix = max(2, 1 + int(np.ceil(abs((xmax - xmin) / s))))
            gcrds.append(np.linspace(xmin, xmax, npix))
            cdelt.append((npix - 1) / (xmax - xmin) if xmin != xmax else 1)

        # In the forward transformation, select only inputs and outputs
        # that we need given world_axes_group parameter:
        bb_center = np.mean(bounding_box, axis=1)

        fixi_dict = {
            k: bb_center[k] for k in set(range(self.pixel_n_dim)).difference(input_axes)
        }

        transform = _fix_transform_inputs(self.forward_transform, fixi_dict)
        transform = transform | Mapping(world_axes_idx,
                                        n_inputs=self.forward_transform.n_outputs)

        xyz = np.meshgrid(*gcrds[::-1], indexing='ij')[::-1]

        shape = xyz[0].shape
        xyz = [v.ravel() for v in xyz]

        coord = np.stack(
            transform(*xyz),
            axis=-1
        )

        coord = coord.reshape(shape + (len(world_axes_group), ))

        # create header with WCS info:
        if hdr is None:
            hdr = fits.Header()

        for m, axis_info in enumerate(world_axes_group):
            k = axis_info.axis
            widx = world_axes_idx.index(k)
            k1 = k + 1
            ct = cf.get_ctype_from_ucd(self.world_axis_physical_types[k])
            if len(ct) > 4:
                raise ValueError("Axis type name too long.")

            hdr[f'CTYPE{k1:d}'] = ct + (4 - len(ct)) * '-' + '-TAB'
            hdr[f'CUNIT{k1:d}'] = self.world_axis_units[k]
            hdr[f'PS{k1:d}_0'] = bin_ext[0]
            hdr[f'PV{k1:d}_1'] = bin_ext[1]
            hdr[f'PS{k1:d}_1'] = coord_col_name
            hdr[f'PV{k1:d}_3'] = widx + 1
            hdr[f'CRVAL{k1:d}'] = 1

            if widx < n_inputs:
                m1 = input_axes[widx] + 1
                hdr[f'CRPIX{m1:d}'] = gcrds[widx][0] + 1
                if use_cd:
                    hdr[f'CD{k1:d}_{m1:d}'] = cdelt[widx]
                else:
                    if k1 != m1:
                        hdr[f'PC{k1:d}_{k1:d}'] = 0.0
                    hdr[f'PC{k1:d}_{m1:d}'] = 1.0
                    hdr[f'CDELT{k1:d}'] = cdelt[widx]
            else:
                m1 = degenerate_axis_start
                degenerate_axis_start += 1

                hdr[f'CRPIX{m1:d}'] = 1
                if use_cd:
                    hdr[f'CD{k1:d}_{m1:d}'] = 1.0
                else:
                    if k1 != m1:
                        hdr[f'PC{k1:d}_{k1:d}'] = 0.0
                    hdr[f'PC{k1:d}_{m1:d}'] = 1.0
                    hdr[f'CDELT{k1:d}'] = 1

                # Uncomment 3 lines below to enable use of degenerate axes:
                # hdr['NAXIS'] = hdr['NAXIS'] + 1
                # naxisi_max = max(int(k[5:]) for k in  hdr['naxis*'] if k[5:].strip())
                # hdr.insert(f'NAXIS{naxisi_max:d}', (f'NAXIS{m1:d}', 1), after=True)
                # NOTE: in this case make sure NAXIS=WCSAXES

                coord = coord[None, :]

        # structured array (data) for binary table HDU:
        arr = np.array(
            [(coord, )],
            dtype=[
                (coord_col_name, np.float64, coord.shape),
            ]
        )

        # create binary table HDU:
        bin_table_hdu = fits.BinTableHDU(arr, name=bin_ext[0], ver=bin_ext[1])

        return hdr, bin_table_hdu

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
            None,
            max_inv_pix_error, 1,
            undist_x, undist_y, u, v,
            undist_xd, undist_yd, ud, vd,
            verbose=True
        )

        self._approx_inverse = (RotateCelestial2Native(crval1, crval2, 180) |
                                Sky2Pix_TAN() | Mapping((0, 1, 0, 1)) |
                                (fit_inv_poly_u & fit_inv_poly_v) |
                                (Shift(crpix[0]) & Shift(crpix[1])))


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
        (i, j)
        for i in range(degree + 1) for j in range(degree + 1 - i) if i + j > 0
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
        warnings.simplefilter('error', category=linalg.LinAlgWarning)
        try:
            lu_piv = linalg.lu_factor(a)
            poly_coeff_x = linalg.lu_solve(lu_piv, bx).astype(float)
            poly_coeff_y = linalg.lu_solve(lu_piv, by).astype(float)
        except (ValueError, linalg.LinAlgWarning, np.linalg.LinAlgError) as e:
            raise np.linalg.LinAlgError(
                f"Failed to fit SIP. Reported error:\n{e.args[0]}"
            )

    if not np.all(np.isfinite([poly_coeff_x, poly_coeff_y])):
        raise np.linalg.LinAlgError(
            "Failed to fit SIP. Computed coefficients are not finite."
        )

    cond = np.linalg.cond(a.astype(float))

    fitx = np.dot(pseudo_vander, poly_coeff_x)
    fity = np.dot(pseudo_vander, poly_coeff_y)

    dist = np.sqrt((xout - fitx)**2 + (yout - fity)**2)
    max_resid = dist.max()

    return poly_coeff_x, poly_coeff_y, max_resid, powers, cond


def _fit_2D_poly(degree, max_error, plate_scale,
                 xin, yin, xout, yout,
                 xind, yind, xoutd, youtd,
                 verbose=False):
    """
    Fit a pair of ordinary 2D polynomials to the supplied transform.

    """
    # The case of one pass with the specified polynomial degree
    if degree is None:
        deglist = list(range(1, 10))
    elif hasattr(degree, '__iter__'):
        deglist = sorted(map(int, degree))
        if deglist[0] < 1 or deglist[-1] > 9:
            raise ValueError("Allowed values for SIP degree are [1...9]")
    else:
        degree = int(degree)
        if degree < 1 or degree > 9:
            raise ValueError("Allowed values for SIP degree are [1...9]")
        deglist = [degree]

    single_degree = len(deglist) == 1

    fit_error = np.inf
    if verbose and not single_degree:
        print(f'Maximum specified SIP approximation error: {max_error}')
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
                print(
                    f"   - SIP degree: {deg}. "
                    f"Maximum residual: {fit_error_i / plate_scale:.5g}"
                )

        except np.linalg.LinAlgError as e:
            if single_degree:
                # Nothing to do if failure is for the lowest degree
                raise e
            else:
                # Keep results from the previous iteration. Discard current fit
                break

        if not np.isfinite(cond):
            # Ill-conditioned system
            if single_degree:
                warnings.warn("The fit may be poorly conditioned.")
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
    for cx, cy, (p, q) in zip(cfx, cfy, powers):
        setattr(fit_poly_x, f'c{p:1d}_{q:1d}', cx)
        setattr(fit_poly_y, f'c{p:1d}_{q:1d}', cy)

    if fit_warning_msg:
        warnings.warn(fit_warning_msg, linalg.LinAlgWarning)

    if fit_error <= max_error or single_degree:
        # Check to see if double sampling meets error requirement.
        max_resid = _compute_distance_residual(
            xoutd,
            youtd,
            fit_poly_x(xind, yind),
            fit_poly_y(xind, yind)
        )
        if verbose:
            print(
                "* Maximum residual, double sampled grid: "
                f"{max_resid / plate_scale:.5g}"
            )

        if max_resid > min(5.0 * fit_error, max_error):
            warnings.warn(
                "Double sampling check FAILED: Sampling may be too coarse for "
                "the distortion model being fitted."
            )

        # Residuals on the double-dense grid may be better estimates
        # of the accuracy of the fit. So we report the largest of
        # the residuals (on single- and double-sampled grid) as the fit error:
        fit_error = max(max_resid, fit_error)

    if verbose:
        if single_degree:
            print(
                f"Maximum residual: {fit_error / plate_scale:.5g}"
            )
        else:
            print(
                f"* Final SIP degree: {deg}. "
                f"Maximum residual: {fit_error / plate_scale:.5g}"
            )

    return fit_poly_x, fit_poly_y, fit_error / plate_scale


def _make_sampling_grid(npoints, bounding_box, crpix):
    step = np.subtract.reduce(bounding_box, axis=1) / (1.0 - npoints)
    crpix = np.asanyarray(crpix)[:, None, None]
    x, y = grid_from_bounding_box(bounding_box, step=step, center=False) - crpix
    return x.flatten(), y.flatten()


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


def _fix_transform_inputs(transform, inputs):
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

    in_selector = Mapping(
        mapping,
        n_inputs = transform.n_inputs - len(inputs)
    )

    input_fixer = Const1D(inputs[0]) if 0 in inputs else Identity(1)
    for k in range(1, transform.n_inputs):
        input_fixer &= Const1D(inputs[k]) if k in inputs else Identity(1)

    transform = in_selector | input_fixer | transform

    return transform


class Step:
    """
    Represents a ``step`` in the WCS pipeline.

    Parameters
    ----------
    frame : `~gwcs.coordinate_frames.CoordinateFrame`
        A gwcs coordinate frame object.
    transform : `~astropy.modeling.Model` or None
        A transform from this step's frame to next step's frame.
        The transform of the last step should be `None`.
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
