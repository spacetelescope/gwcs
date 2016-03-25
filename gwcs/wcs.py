# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, unicode_literals, print_function

import copy
import functools
import numpy as np
from astropy.extern import six
from astropy.modeling.core import Model

from . import coordinate_frames
from .utils import (ModelDimensionalityError, CoordinateFrameError)
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
            _inp_frame = getattr(self, self.input_frame)
            if  _inp_frame is not None and _inp_frame.naxes != self.forward_transform.n_inputs:
                message = "The number of inputs {0} of the forward transform \
                does not match the number of axes {1} of the input axes. \
                ".format(self.forward_transform.n_inputs, _inp_frame.naxes)
                raise ModelDimensionalityError(message)
            _outp_frame = getattr(self, self.output_frame)
            if _outp_frame is not None and _outp_frame.naxes != self.forward_transform.n_outputs:
                message = "The number of outputs {0} of the forward transform \
                does not match the number of axes {1} of the output axes. \
                ".format(self.forward_transform.n_outputs, _outp_frame.naxes)
                raise ModelDimensionalityError(message)
        else:
            if output_frame is None:
                raise CoordinateFrameError("An output_frame must be specified"
                                           "if forward_transform is None.")
            _input_frame, inp_frame_obj = self._get_frame_name(input_frame)
            _output_frame, outp_frame_obj = self._get_frame_name(output_frame)
            super(WCS, self).__setattr__(_input_frame, inp_frame_obj)
            super(WCS, self).__setattr__(_output_frame, outp_frame_obj)
            self._pipeline = [(_input_frame, None),
                              (_output_frame, None)]
        for frame in self.available_frames:
            if getattr(self, frame) is not None:
                getattr(self, frame)._set_wcsobj(self)


    def get_transform(self, from_frame, to_frame):
        """
        Return a transform between two coordinate frames.

        Parameters
        ----------
        from_frame : str or `~gwcs.coordinate_frame.CoordinateFrame`
            Initial coordinate frame.
        to_frame : str, or instance of `~gwcs.cordinate_frames.CoordinateFrame`
            Coordinate frame into which to transform.

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
            transforms = np.array(self._pipeline[from_ind: to_ind])[:, 1].tolist()
        return functools.reduce(lambda x, y: x | y, transforms)

    def set_transform(self, from_frame, to_frame, transform):
        """
        Set/replace the transform between two coordinate frames.

        Parameters
        ----------
        from_frame : str or `~gwcs.coordinate_frame.CoordinateFrame`
            Initial coordinate frame.
        to_frame : str, or instance of `~gwcs.cordinate_frames.CoordinateFrame`
            Coordinate frame into which to transform.
        transform : `~astropy.modeling.Model`
            Transform between two frames.
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
        backward = self.forward_transform.inverse
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

    def __call__(self, *args):
        """
        Executes the forward transform.

        args : float or array-like
            Inputs in the input coordinate system, separate inputs for each dimension.

        """
        if self.forward_transform is not None:
            return self.forward_transform(*args)

    def invert(self, *args, **kwargs):
        """
        Invert coordnates.

        The analytical inverse of the forward transform is used, if available.
        If not an iterative method is used.

        Parameters
        ----------
        args : float or array like
            coordinates to be inverted
        kwargs : dict
            keyword arguments to be passed to the iterative invert method.
        """
        try:
            return self.forward_transform.inverse(*args)
        except (NotImplementedError, KeyError):
            return self._invert(*args, **kwargs)

    def _invert(self, *args, **kwargs):
        """
        Implement iterative inverse here.
        """
        raise NotImplementedError

    def transform(self, from_frame, to_frame, *args):
        """
        Transform potitions between two frames.


        Parameters
        ----------
        from_frame : str or `~gwcs.coordinate_frames.CoordinateFrame`
            Initial coordinate frame.
        to_frame : str, or instance of `~gwcs.cordinate_frames.CoordinateFrame`
            Coordinate frame into which to transform.
        args : float
            input coordinates to transform
        """
        transform = self.get_transform(from_frame, to_frame)
        return transform(*args)

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
            immediately after `frame`.
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
            return self._pipeline[-1][0]
        else:
            return None

    @property
    def input_frame(self):
        """Return the input coordinate frame."""
        if self._pipeline:
            return self._pipeline[0][0]
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
    def domain(self):
        frames = self.available_frames
        transform_meta = self.get_transform(frames[0], frames[1]).meta
        if 'domain' in transform_meta:
            return transform_meta['domain']
        else:
            return None

    @domain.setter
    def domain(self, value):
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
        for item in self._pipeline:
            model = item[1]
            if model is not None:
                if model.name is not None:
                    col2.append("comp"+model.name)
                else:
                    lenmodels = len(model.submodel_names)
                    col2.append("CompoundModel_" + str(lenmodels))
            else:
                col2.append("None")
        t = Table([col1, col2], names=['From',  'Transform'])
        return str(t)

    def __repr__(self):
        fmt = "<WCS(output_frame={0}, input_frame={1}, forward_transform={2})>".format(
            self.output_frame, self.input_frame, self.forward_transform)
        return fmt

    def footprint(self, domain=None, center=True):
        """
        Return the footprint of the observation in world coordinates.

        Parameters
        ----------
        domain : slice or tuple of floats: (start, stop, step) or (start, stop) or (stop,)
            size of image
        center : bool
            If `True` use the center of the pixel, otherwise use the corner.

        Returns
        -------
        coord : (4, 2) array of (*x*, *y*) coordinates.
            The order is counter-clockwise starting with the bottom left corner.
        """
        if domain is None:
            if self.domain is None:
                raise TypeError("Need a valid domain to compute the footprint.")
            domain = self.domain
        self._validate_domain(domain)

        bounds = utils._domain_to_bounds(domain)
        vertices = np.asarray([[bounds[0][0], bounds[1][0]], [bounds[0][0], bounds[1][1]],
                               [bounds[0][1], bounds[1][1]], [bounds[0][1], bounds[1][0]]])
        vertices = _toindex(vertices).T
        if not center:
            vertices += .5
        result = self.__call__(*vertices)
        return np.asarray(result)
