# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, unicode_literals, print_function

import copy
import functools
import numpy as np
from astropy.extern import six
from astropy.io import fits
from astropy.modeling import models
from astropy.modeling.core import Model
from astropy.utils import isiterable

from . import coordinate_frames
from .util import ModelDimensionalityError, CoordinateFrameError
from .selector import *


__all__ = ['WCS']


class WCS(object):

    """
    Basic WCS class.

    Parameters
    ----------
    output_frame : str, `~gwcs.coordinate_frames.CoordinateFrame`
        A coordinates object or a string name.
    input_frame : str, `~gwcs.coordinate_frames.CoordinateFrame`
        A coordinates object or a string name.
    forward_transform : `~astropy.modeling.Model` or a list
    A model to do the transform between ``input_frame`` and ``output_frame``.
        A list of (frame, transform) tuples where ``frame`` is the starting frame and
        ``transform`` is the transform from this frame to the next one or ``output_frame``.
    name : str
        a name for this WCS
    """

    def __init__(self, output_frame, input_frame='detector',
                 forward_transform=None, name=""):
        self._coord_frames = {}
        self._pipeline = []
        self._input_frame, frame_obj = self._get_frame_name(input_frame)
        self._coord_frames[self._input_frame] = frame_obj
        self._output_frame, frame_obj = self._get_frame_name(output_frame)
        self._coord_frames[self._output_frame] = frame_obj
        self._name = name
        if forward_transform is not None:
            if isinstance(forward_transform, Model):
                self._pipeline = [(self._input_frame, forward_transform.copy()),
                                  (self._output_frame, None)]
            elif isinstance(forward_transform, list):
                for item in forward_transform:
                    name, frame_obj = self._get_frame_name(item[0])
                    self._coord_frames[name] = copy.deepcopy(frame_obj)
                    self._pipeline.append((name, item[1]))
            else:
                raise TypeError("Expected forward_transform to be a model or a "
                                "(frame, transform) list, got {0}".format(
                                    type(forward_transform)))
        else:
            self._pipeline = [(self._input_frame, None),
                              (self._output_frame, None)]

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
        from_name, from_obj = self._get_frame_name(from_frame)
        to_name, to_obj = self._get_frame_name(to_frame)

        # if from_name not in self.available_frames:
        #raise ValueError("Frame {0} is not in the available frames".format(from_frame))
        # if to_name not in self.available_frames:
        #raise ValueError("Frame {0} is not in the available frames".format(to_frame))
        try:
            from_ind = self._get_frame_index(from_name)
        except ValueError:
            raise CoordinateFrameError("Frame {0} is not in the available frames".format(from_name))
        try:
            to_ind = self._get_frame_index(to_name)
        except ValueError:
            raise CoordinateFrameError("Frame {0} is not in the available frames".format(to_name))
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
        self._pipeline[from_ind] = (self._pipeline[from_ind], transform)

    @property
    def forward_transform(self):
        """
        Return the total forward transform - from input to output coordinate frame.

        """

        if self._pipeline:
            if self._pipeline[-1] != (self._output_frame, None):
                self._pipeline.append((self._output_frame, None))
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
            name = frame.wcs_name
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
        return self._coord_frames

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
        try:
            return self._coord_frames[self._output_frame].unit
        except AttributeError:
            return None

    @property
    def output_frame(self):
        """Return the output coordinate frame."""
        if self._coord_frames[self._output_frame] is not None:
            return self._coord_frames[self._output_frame]
        else:
            return self._output_frame

    @property
    def input_frame(self):
        """Return the input coordinate frame."""
        if self._coord_frames[self._input_frame] is not None:
            return self._coord_frames[self._input_frame]
        else:
            return self._input_frame

    @property
    def name(self):
        """Return the name for this WCS."""
        return self._name

    @name.setter
    def name(self, value):
        """Set the name for the WCS."""
        self._name = value

    def __str__(self):
        from astropy.table import Table
        col1 = [item[0] for item in self._pipeline]
        col2 = []
        for item in self._pipeline:
            model = item[1]
            if model is not None:
                if model.name != "":
                    col2.append(model.name)
                else:
                    col2.append(model.__class__.__name__)
            else:
                col2.append(None)
        t = Table([col1, col2], names=['From',  'Transform'])
        return str(t)

    def __repr__(self):
        fmt = "<WCS(output_frame={0}, input_frame={1}, forward_transform={2})>".format(
            self.output_frame, self.input_frame, self.forward_transform)
        return fmt

    def footprint(self, axes, center=True):
        """
        Return the footprint of the observation in world coordinates.

        Parameters
        ----------
        axes : tuple of floats
            size of image
        center : bool
            If `True` use the center of the pixel, otherwise use the corner.

        Returns
        -------
        coord : (4, 2) array of (*x*, *y*) coordinates.
            The order is counter-clockwise starting with the bottom left corner.
        """
        naxis1, naxis2 = axes  # extend this to more than 2 axes
        if center == True:
            corners = np.array([[1, 1],
                                [1, naxis2],
                                [naxis1, naxis2],
                                [naxis1, 1]], dtype=np.float64)
        else:
            corners = np.array([[0.5, 0.5],
                                [0.5, naxis2 + 0.5],
                                [naxis1 + 0.5, naxis2 + 0.5],
                                [naxis1 + 0.5, 0.5]], dtype=np.float64)
        result = self.__call__(corners[:, 0], corners[:, 1])
        return np.asarray(result).T
        #result = np.vstack(self.__call__(corners[:,0], corners[:,1])).T
        # try:
        # return self.output_coordinate_system.world_coordinates(result[:,0], result[:,1])
        # except:
        # return result
