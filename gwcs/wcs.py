# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import division, print_function

import functools
import numpy as np
from astropy.extern import six
from astropy.io import fits
from astropy.modeling import models
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
    output_coordinate_system : str, `~gwcs.coordinate_frames.CoordinateFrame`
        A coordinates object or a string label
    input_coordinate_system : str, `~gwcs.coordinate_frames.CoordinateFrame`
        A coordinates object or a string label
    forward_transform : `~astropy.modeling.Model`
        a model to do the forward transform
    name : str
        a name for this WCS
    """
    def __init__(self, output_coordinate_system,  input_coordinate_system='detector',
                 forward_transform=None, name=""):
        self._coord_frames = {}
        self._pipeline = {}
        if isinstance(input_coordinate_system, six.string_types):
            self._input_coordinate_system = input_coordinate_system
            self._coord_frames[self._input_coordinate_system] = None
        else:
            self._input_coordinate_system = input_coordinate_system.name
            self._coord_frames[self._input_coordinate_system] = input_coordinate_system
        if isinstance(output_coordinate_system, six.string_types):
            self._output_coordinate_system = output_coordinate_system
            self._coord_frames[self._output_coordinate_system] = None
        else:
            self._output_coordinate_system = output_coordinate_system.name
            self._coord_frames[self._output_coordinate_system] = output_coordinate_system
        self._name = name
        if forward_transform is not None:
            self.add_transform(self.input_coordinate_system,
                               self.output_coordinate_system,
                               forward_transform)

    @property
    def unit(self):
        """The unit of the coordinates in the output coordinate system."""
        try:
            return self._coord_frames[self._output_coordinate_system].unit
        except AttributeError:
            return None

    @property
    def output_coordinate_system(self):
        """Return the output coordinate system."""
        if self._coord_frames[self._output_coordinate_system] is not None:
            return self._coord_frames[self._output_coordinate_system]
        else:
            return self._output_coordinate_system

    @property
    def input_coordinate_system(self):
        """Return the input coordinate system."""
        if self._coord_frames[self._input_coordinate_system] is not None:
            return self._coord_frames[self._input_coordinate_system]
        else:
            return self._input_coordinate_system

    @property
    def forward_transform(self):
        """ Return the total forward transform - from input to output coordinate system."""
        return self.get_transform(self._input_coordinate_system, self._output_coordinate_system)

    @property
    def backward_transform(self):
        """
        Return the total backward transform if available - from output to input coordinate system.
        """
        try:
            backward = self.forward_transform.inverse
            return backward
        except NotImplementedError:
            return None

    @property
    def name(self):
        """Name for this WCS."""
        return self._name

    @name.setter
    def name(self, value):
        """Set the name for the WCS."""
        self._name = value

    def __call__(self, *args):
        """
        Executes the forward transform.

        args : float or array-like
            Inputs in the input coordinate system, separate inputs for each dimension.

        """
        if self.forward_transform is not None:
            return self.forward_transform(*args)
        #if self.output_coordinate_system is not None:
            #return self.output_coordinate_system.world_coordinates(*result)
        #else:
            #return result

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
        from_frame : str or `~gwcs.coordinate_frame.CoordinateFrame`
            Initial coordinate frame.
        to_frame : str, or instance of `~gwcs.cordinate_frames.CoordinateFrame`
            Coordinate frame into which to transform.
        args : float
            input coordinates to transform
        """
        transform = self.get_transform(from_frame, to_frame)
        return transform(*args)


    def get_transform(self, from_frame, to_frame):
        """
        Return a transform between two coordinate frames

        Parameters
        ----------
        from_frame : str or `~gwcs.coordinate_frame.CoordinateFrame`
            Initial coordinate frame.
        to_frame : str, or instance of `~gwcs.cordinate_frames.CoordinateFrame`
            Coordinate frame into which to transform.
        """
        if not self._pipeline:
            return None
        if not isinstance(from_frame, six.string_types):
            from_frame = from_frame.name
        if not isinstance(to_frame, six.string_types):
            to_frame = to_frame.name
        if from_frame not in self.available_frames:
            raise ValueError("Frame {0} is not in the available frames".format(from_frame))
        if to_frame not in self.available_frames:
            raise ValueError("Frame {0} is not in the available frames".format(to_frame))
        transforms = []
        frame = from_frame
        while frame != to_frame:
            frame, transform = self._pipeline[frame].items()[0]
            transforms.append(transform)

        return functools.reduce(lambda x, y: x | y , transforms)

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

    def add_transform(self, from_frame, to_frame, transform):
        """
        Add a transform between two coordinate frames.

        At least one of the frames must already exist in the pipeline.
        If both already exist, the transform relaces the existing transform.

        Parameters
        ----------
        from_frame : str or `~gwcs.coordinate_frame.CoordinateFrame`
            Coordinate frame.
        to_frame : str, or instance of `~gwcs.cordinate_frames.CoordinateFrame`
            Coordinate frame.
        transform : `~astropy.modeling.Model`
            Transform.
        """
        if isinstance(from_frame, six.string_types):
            from_name = from_frame
            from_frame_obj = None
        else:
            from_name = from_frame.name
            from_frame_obj = from_frame

        if isinstance(to_frame, six.string_types):
            to_name = to_frame
            to_frame_obj = None
        else:
            to_name = to_frame.name
            to_frame_obj = to_frame
        self._pipeline[from_name] = {to_name: transform}
        self._coord_frames[from_name] = from_frame_obj
        self._coord_frames[to_name] = to_frame_obj

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
        naxis1, naxis2 = axes # extend this to more than 2 axes
        if center == True:
            corners = np.array([[1, 1],
                                [1, naxis2],
                                [naxis1, naxis2],
                                [naxis1, 1]], dtype = np.float64)
        else:
            corners = np.array([[0.5, 0.5],
                                [0.5, naxis2 + 0.5],
                                [naxis1 + 0.5, naxis2 + 0.5],
                                [naxis1 + 0.5, 0.5]], dtype = np.float64)
        return self.__call__(corners[:,0], corners[:,1])
        #result = np.vstack(self.__call__(corners[:,0], corners[:,1])).T
        #try:
            #return self.output_coordinate_system.world_coordinates(result[:,0], result[:,1])
        #except:
            #return result

