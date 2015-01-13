# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import division
import numpy as np
from astropy.io import fits
from astropy.modeling import models
from astropy.utils import isiterable

from . import coordinate_systems
from .util import ModelDimensionalityError, CoordinateFrameError
from .selector import *


__all__ = ['WCS']


class WCS(object):
    """
    Basic WCS class

    Parameters
    ----------
    output_coordinate_system : str, gwcs.coordinate_systems.Frame
        A coordinates object or a string label
    input_coordinate_system : str, gwcs.coordinate_systems.Frame
        A coordinates object or a string label
    forward_transform : astropy.modeling.Model
        a model to do the forward transform
    name : str
        a name for this WCS
    """
    def __init__(self, output_coordinate_system,  input_coordinate_system='detector',
                 forward_transform=None, name=""):
        self._forward_transform = forward_transform
        self._input_coordinate_system = input_coordinate_system
        self._output_coordinate_system = output_coordinate_system
        self._name = name
        '''
        if forward_transform is not None and input_coordinate_system is not None \
           and output_coordinate_system is not None:
            self._pipeline.add_transform(self._input_coordinate_system.__class__,
                                         self._output_coordinate_system.__class__,
                                         forward_transform)
        '''

    @property
    def unit(self):
        return self._output_coordinate_system._unit

    @property
    def output_coordinate_system(self):
        return self._output_coordinate_system

    @property
    def input_coordinate_system(self):
        return self._input_coordinate_system

    @property
    def forward_transform(self):
        return self._forward_transform

    @forward_transform.setter
    def forward_transform(self, value):
        self._forward_transform = value.copy()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def __call__(self, *args):
        if self._forward_transform is not None:
            result = self._forward_transform(*args)
        if self.output_coordinate_system is not None:
            return self.output_coordinate_system.world_coordinates(*result)
        else:
            return result

    def invert(self, *args, **kwargs):
        '''
        args : x, y
        **kwargs: x0, y0, maxiter ,...
        '''
        try:
            return self.forward_transform.inverse(*args)
        except (NotImplementedError, KeyError):
            return self._invert(*args, **kwargs)

    def _invert(self, x, y, x0, y0, **kwargs):
        raise NotImlementeError

    def transform(self, fromsys, tosys, *args):
        """
        Perform coordinate transformation beteen two frames inclusive.

        Parameters
        ----------
        fromsys : CoordinateFrame
            an instance of CoordinateFrame
        tosys : CoordinateFrame
            an instance of CoordinateFrame
        args : float
            input coordinates to transform
        """
        transform = self._forward_transform[fromsys : tosys]
        return transform(*args)


    def get_transform(self, fromsys, tosys):
        """
        Return a transform between two coordinate frames

        Parameters
        ----------
        fromsys : CoordinateFrame
            an instance of CoordinateFrame
        tosys : CoordinateFrame
            an instance of CoordinateFrame
        """
        try:
            return self._forward_transform[fromsys : tosys]
        except ValueError:
            try:
                transform = self._forward_transform[tosys : fromsys]
            except ValueError:
                return None
            try:
                return transform.inverse
            except NotImplementedError:
                return None

    @property
    def available_frames(self):
        return self.forward_transform.submodel_names

    def footprint(self, axes, center=True):
        """
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

