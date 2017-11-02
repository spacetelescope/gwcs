# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, unicode_literals, print_function

import functools
import warnings

import numpy as np
import six
from astropy.modeling.core import Model

from . import coordinate_frames
from .utils import CoordinateFrameError
from .utils import _toindex
from . import utils


__all__ = ['WCS']


class WCS(object):

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
        self._available_frames = []
        self._pipeline = []
        self._name = name
        self._initialize_wcs(forward_transform, input_frame, output_frame)

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

                self._pipeline = [(_input_frame, forward_transform.copy()),
                                  (_output_frame, None)]
            elif isinstance(forward_transform, list):
                for item in forward_transform:
                    name, frame_obj = self._get_frame_name(item[0])
                    super(WCS, self).__setattr__(name, frame_obj)
                    self._pipeline.append((name, item[1]))

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
            raise CoordinateFrameError("Frame {0} is not in the available frames".format(from_frame))
        try:
            to_ind = self._get_frame_index(to_frame)
        except ValueError:
            raise CoordinateFrameError("Frame {0} is not in the available frames".format(to_frame))
        if to_ind < from_ind:
            transforms = np.array(self._pipeline[to_ind: from_ind])[:, 1].tolist()
            transforms = [tr.inverse for tr in transforms[::-1]]
        elif to_ind == from_ind:
            return None
        else:
            transforms = np.array(self._pipeline[from_ind: to_ind])[:, 1].copy()
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
        self._pipeline[from_ind] = (self._pipeline[from_ind][0], transform)

    @property
    def forward_transform(self):
        """
        Return the total forward transform - from input to output coordinate frame.

        """

        if self._pipeline:
            return functools.reduce(lambda x, y: x | y, [step[1] for step in self._pipeline[: -1]])
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
        if isinstance(frame, coordinate_frames.CoordinateFrame):
            frame = frame.name
        return np.asarray(self._pipeline)[:, 0].tolist().index(frame)

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
        if isinstance(frame, six.string_types):
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
            Inputs in the input coordinate system, separate inputs for each dimension.
        output : str, optional
            One of [``numericals``, ``numericals_plus``]
            If ``numericals_plus`` - returns a `~astropy.coordinates.SkyCoord` or
            `~astropy.units.Quantity` object.
        with_bounding_box : bool, optional
             If True(default) values in the result which correspond to any of the inputs being
             outside the bounding_box are set to ``fill_value``.
        fill_value : float, optional
            Output value for inputs outside the bounding_box (default is np.nan).
        """
        if self.forward_transform is None:
            raise NotImplementedError("WCS.forward_transform is not implemented.")

        output = kwargs.pop("output", "numericals")
        if 'with_bounding_box' not in kwargs:
            kwargs['with_bounding_box'] = True
        if 'fill_value' not in kwargs:
            kwargs['fill_value'] = np.nan

        transform = self.forward_transform
        if self.bounding_box is not None:
            # Currently compound models do not attempt to combine individual model
            # bounding boxes. Get the forward transform and assign the ounding_box to it
            # before evaluating it. The order Model.bounding_box is reversed.
            axes_ind = self._get_axes_indices()
            if transform.n_inputs > 1:
                transform.bounding_box = np.array(self.bounding_box)[axes_ind][::-1]
            else:
                transform.bounding_box = self.bounding_box
        result = transform(*args, **kwargs)

        if output not in ["numericals", "numericals_plus"]:
            raise ValueError("'output' should be 'numericals' or "
                             "'numericals_plus', not '{0}'.".format(output))
        if output == 'numericals_plus':
            if self.output_frame.naxes == 1:
                result = self.output_frame.coordinates(result)
            else:
                result = self.output_frame.coordinates(*result)
        elif output is not None and output != "numericals":
            raise ValueError("Type of output unrecognized {0}".format(output))
        return result


    def invert(self, *args, **kwargs):
        """
        Invert coordnates.

        The analytical inverse of the forward transform is used, if available.
        If not an iterative method is used.

        Parameters
        ----------
        args : float, array like, `~astropy.coordinates.SkyCoord` or `~astropy.units.Unit`
            coordinates to be inverted
        kwargs : dict
            keyword arguments to be passed to the iterative invert method.
        with_bounding_box : bool, optional
             If True(default) values in the result which correspond to any of the inputs being
             outside the bounding_box are set to ``fill_value``.
        fill_value : float, optional
            Output value for inputs outside the bounding_box (default is np.nan).
        """
        if not utils.isnumerical(args[0]):
            args = utils._get_values(self.unit, *args)

        output = kwargs.pop('output', None)
        if 'with_bounding_box' not in kwargs:
            kwargs['with_bounding_box'] = True
        if 'fill_value' not in kwargs:
            kwargs['fill_value'] = np.nan

        try:
            result = self.backward_transform(*args, **kwargs)
        except (NotImplementedError, KeyError):
            result = self._invert(*args, **kwargs)

        if output == 'numericals_plus':
            if self.input_frame.naxes == 1:
                return self.input_frame.coordinates(result)
            else:
                return self.input_frame.coordinates(*result)
        else:
            return result

    def _invert(self, *args, **kwargs):
        """
        Implement iterative inverse here.
        """
        raise NotImplementedError

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
        output : str
            One of [``numericals``, ``numericals_plus``]
            If ``numericals_plus`` - returns a `~astropy.coordinates.SkyCoord` or
            `~astropy.units.Quantity` object.
        with_bounding_box : bool, optional
             If True(default) values in the result which correspond to any of the inputs being
             outside the bounding_box are set to ``fill_value``.
        fill_value : float, optional
            Output value for inputs outside the bounding_box (default is np.nan).
        """
        transform = self.get_transform(from_frame, to_frame)
        if not utils.isnumerical(args[0]):
            args = utils._get_values(self.unit, *args)

        output = kwargs.pop("output", None)
        if 'with_bounding_box' not in kwargs:
            kwargs['with_bounding_box'] = True
        if 'fill_value' not in kwargs:
            kwargs['fill_value'] = np.nan

        result = transform(*args, **kwargs)

        if output == "numericals_plus":
            to_frame_name, to_frame_obj = self._get_frame_name(to_frame)
            if to_frame_obj is not None:
                if to_frame_obj.naxes == 1:
                    result = to_frame_obj.coordinates(result)
                else:
                    result = to_frame_obj.coordinates(*result)
            else:
                raise TypeError("Coordinate objects could not be created because"
                                "frame {0} is not defined.".format(to_frame_name))
        elif output is not None and output != "numericals":
            raise ValueError("Type of output unrecognized {0}".format(output))
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
            return [frame[0] for frame in self._pipeline]
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
            fr, current_transform = self._pipeline[frame_ind - 1]
            self._pipeline[frame_ind - 1] = (fr, current_transform | transform)
        else:
            fr, current_transform = self._pipeline[frame_ind]
            self._pipeline[frame_ind] = (fr, transform | current_transform)

    @property
    def unit(self):
        """The unit of the coordinates in the output coordinate system."""
        if self._pipeline:
            try:
                return getattr(self, self._pipeline[-1][0]).unit
            except AttributeError:
                return None
        else:
            return None

    @property
    def output_frame(self):
        """Return the output coordinate frame."""
        if self._pipeline:
            return getattr(self, self._pipeline[-1][0])
        else:
            return None

    @property
    def input_frame(self):
        """Return the input coordinate frame."""
        if self._pipeline:
            return getattr(self, self._pipeline[0][0])
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
            # Model.bounding_box is in numpy order, need to reverse it first.
            bb = transform_0.bounding_box#[::-1]
        except NotImplementedError:
            return None
        if transform_0.n_inputs == 1:
            return bb
        try:
            axes_order = self.input_frame.axes_order
        except AttributeError:
            axes_order = np.arange(transform_0.n_inputs)
        bb = np.array(bb[::-1])[np.array(axes_order)]
        return tuple(tuple(item) for item in bb)

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
            axes_ind = self._get_axes_indices()
            try:
                transform_0.bounding_box = np.array(value)[axes_ind][::-1]
            except IndexError:
                raise utils.DimensionalityError("The bounding_box does not match the number of inputs.")
        self.set_transform(frames[0], frames[1], transform_0)

    def _get_axes_indices(self):
        try:
            axes_ind = np.argsort(self.input_frame.axes_order)
        except AttributeError:
            # the case of a frame being a string
            axes_ind = np.arange(self.forward_transform.n_inputs)
        return axes_ind

    @property
    def domain(self):
        """
        Return the range of acceptable values for each input axis.
        """
        warnings.warn('"domain" was deprecated in v0.8 and will be'
                      'removed in the next version. Use "bounding_box" instead.')
        frames = self.available_frames
        transform_meta = self.get_transform(frames[0], frames[1]).meta
        if 'domain' in transform_meta:
            return transform_meta['domain']
        else:
            return None

    @domain.setter
    def domain(self, value):
        """
        Set the range of acceptable values for each input axis.
        """
        warnings.warn('"domain" was deprecated in v.0.8 and will be removed'
                      ' in the next version. Use "bounding_box" instead.')
        self._validate_domain(value)
        frames = self.available_frames
        transform = self.get_transform(frames[0], frames[1])
        transform.meta['domain'] = value
        self.set_transform(frames[0], frames[1], transform)

    def _validate_domain(self, domain):
        n_inputs = self.forward_transform.n_inputs
        if len(domain) != n_inputs:
            raise ValueError("The number of domains should match the number "
                             "of inputs {0}".format(n_inputs))
        if not isinstance(domain, (list, tuple)) or \
           not all([isinstance(d, dict) for d in domain]):
            raise TypeError('"domain" should be a list of dictionaries for each axis in the input_frame'
                            "[{'lower': low_x, "
                            "'upper': high_x, "
                            "'includes_lower': bool, "
                            "'includes_upper': bool}]")

    def __str__(self):
        from astropy.table import Table
        col1 = [item[0] for item in self._pipeline]
        col2 = []
        for item in self._pipeline[: -1]:
            model = item[1]
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

    def footprint(self, bounding_box=None, center=False):
        """
        Return the footprint of the observation in world coordinates.

        Parameters
        ----------
        bounding_box : tuple of floats: (start, stop)
            `prop: bounding_box`
        center : bool
            If `True` use the center of the pixel, otherwise use the corner.

        Returns
        -------
        coord : array of coordinates in the output_frame.
            The order is counter-clockwise starting with the bottom left corner.
        """
        if bounding_box is None:
            if self.bounding_box is None:
                raise TypeError("Need a valid bounding_box to compute the footprint.")
            bb = self.bounding_box
        else:
            bb = bounding_box
        vertices = np.asarray([[bb[0][0], bb[1][0]], [bb[0][0], bb[1][1]],
                               [bb[0][1], bb[1][1]], [bb[0][1], bb[1][0]]]).T
        if center:
            vertices = _toindex(vertices)
        result = self.__call__(*vertices, **{'with_bounding_box': False})
        return np.asarray(result)
