# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
import astropy.time

from asdf import yamlutil
from ..gwcs_types import GWCSType
from ..coordinate_frames import (Frame2D, CoordinateFrame, CelestialFrame,
                                 SpectralFrame, TemporalFrame, CompositeFrame,
                                 StokesFrame)
from ..wcs import WCS, Step


_REQUIRES = ['astropy']


__all__ = ["WCSType", "CelestialFrameType", "CompositeFrameType", "FrameType",
           "SpectralFrameType", "StepType", "TemporalFrameType", "StokesFrameType"]


class WCSType(GWCSType):
    name = "wcs"
    requires = _REQUIRES
    types = [WCS]
    version = '1.1.0'

    @classmethod
    def from_tree(cls, node, ctx):
        name = node['name']
        steps = [(x.frame, x.transform) for x in node['steps']]
        return WCS(steps, name=name)

    @classmethod
    def to_tree(cls, gwcsobj, ctx):
        return {'name': gwcsobj.name,
                'steps': gwcsobj.pipeline
                }

    @classmethod
    def assert_equal(cls, old, new):
        from asdf.tests import helpers
        assert old.name == new.name # nosec
        assert len(old.available_frames) == len(new.available_frames) # nosec
        for old_step, new_step in zip(
                old.pipeline, new.pipeline):
            helpers.assert_tree_match(old_step.frame, new_step.frame)
            helpers.assert_tree_match(old_step.transform, new_step.transform)


class StepType(dict, GWCSType):
    name = "step"
    requires = _REQUIRES
    version = '1.1.0'
    types = [Step]

    @classmethod
    def from_tree(cls, node, ctx):
        return Step(frame=node['frame'], transform=node.get('transform', None))

    @classmethod
    def to_tree(cls, step, ctx):
        return {'frame': step.frame,
                'transform': step.transform}


class FrameType(GWCSType):
    name = "frame"
    requires = _REQUIRES
    types = [CoordinateFrame]
    version = '1.0.0'

    @classmethod
    def _from_tree(cls, node, ctx):
        kwargs = {'name': node['name']}

        if 'axes_type' in node and 'naxes' in node:
            kwargs.update({
                'axes_type': node['axes_type'],
                'naxes': node['naxes']})

        if 'axes_names' in node:
            kwargs['axes_names'] = node['axes_names']

        if 'reference_frame' in node:
            kwargs['reference_frame'] = yamlutil.tagged_tree_to_custom_tree(
                node['reference_frame'], ctx)

        if 'axes_order' in node:
            kwargs['axes_order'] = tuple(node['axes_order'])

        if 'unit' in node:
            kwargs['unit'] = tuple(
                yamlutil.tagged_tree_to_custom_tree(node['unit'], ctx))

        if 'axis_physical_types' in node:
            kwargs['axis_physical_types'] = tuple(node['axis_physical_types'])

        return kwargs

    @classmethod
    def _to_tree(cls, frame, ctx):

        node = {}

        node['name'] = frame.name

        # We want to check that it is exactly this type and not a subclass
        if type(frame) is CoordinateFrame:
            node['axes_type'] = frame.axes_type
            node['naxes'] = frame.naxes

        if frame.axes_order is not None:
            node['axes_order'] = list(frame.axes_order)

        if frame.axes_names is not None:
            node['axes_names'] = list(frame.axes_names)

        if frame.reference_frame is not None:
            node['reference_frame'] = yamlutil.custom_tree_to_tagged_tree(
                frame.reference_frame, ctx)

        if frame.unit is not None:
            node['unit'] = yamlutil.custom_tree_to_tagged_tree(
                list(frame.unit), ctx)

        if frame.axis_physical_types is not None:
            node['axis_physical_types'] = list(frame.axis_physical_types)

        return node

    @classmethod
    def _assert_equal(cls, old, new):
        from asdf.tests import helpers
        assert old.name == new.name  # nosec
        assert old.axes_order == new.axes_order  # nosec
        assert old.axes_names == new.axes_names   # nosec
        assert type(old.reference_frame) is type(new.reference_frame)  # nosec
        assert old.unit == new.unit  # nosec

        if old.reference_frame is not None:
            for name in old.reference_frame.get_frame_attr_names().keys():
                helpers.assert_tree_match(
                    getattr(old.reference_frame, name),
                    getattr(new.reference_frame, name))

    @classmethod
    def assert_equal(cls, old, new):
        cls._assert_equal(old, new)

    @classmethod
    def from_tree(cls, node, ctx):
        node = cls._from_tree(node, ctx)
        return CoordinateFrame(**node)

    @classmethod
    def to_tree(cls, frame, ctx):
        return cls._to_tree(frame, ctx)


class Frame2DType(FrameType):
    name = "frame2d"
    types = [Frame2D]

    @classmethod
    def from_tree(cls, node, ctx):
        node = cls._from_tree(node, ctx)
        return Frame2D(**node)


class CelestialFrameType(FrameType):
    name = "celestial_frame"
    types = [CelestialFrame]

    @classmethod
    def from_tree(cls, node, ctx):
        node = cls._from_tree(node, ctx)
        return CelestialFrame(**node)

    @classmethod
    def to_tree(cls, frame, ctx):
        return cls._to_tree(frame, ctx)

    @classmethod
    def assert_equal(cls, old, new):
        cls._assert_equal(old, new)

        assert old.reference_position == new.reference_position  # nosec


class SpectralFrameType(FrameType):
    name = "spectral_frame"
    types = [SpectralFrame]
    version = "1.0.0"

    @classmethod
    def from_tree(cls, node, ctx):
        node = cls._from_tree(node, ctx)

        if 'reference_position' in node:
            node['reference_position'] = node['reference_position'].upper()

        return SpectralFrame(**node)

    @classmethod
    def to_tree(cls, frame, ctx):
        node = cls._to_tree(frame, ctx)

        if frame.reference_position is not None:
            node['reference_position'] = frame.reference_position.lower()

        return node


class CompositeFrameType(FrameType):
    name = "composite_frame"
    types = [CompositeFrame]

    @classmethod
    def from_tree(cls, node, ctx):
        if len(node) != 2:
            raise ValueError("CompositeFrame has extra properties")

        name = node['name']
        frames = node['frames']

        return CompositeFrame(frames, name)

    @classmethod
    def to_tree(cls, frame, ctx):
        return {
            'name': frame.name,
            'frames': yamlutil.custom_tree_to_tagged_tree(frame.frames, ctx)
        }

    @classmethod
    def assert_equal(cls, old, new):
        from asdf.tests import helpers
        assert old.name == new.name  # nosec
        for old_frame, new_frame in zip(old.frames, new.frames):
            helpers.assert_tree_match(old_frame, new_frame)


class TemporalFrameType(FrameType):
    name = "temporal_frame"
    requires = _REQUIRES
    types = [TemporalFrame]
    version = '1.0.0'


    @classmethod
    def from_tree(cls, node, ctx):
        node = cls._from_tree(node, ctx)
        return TemporalFrame(**node)

    @classmethod
    def to_tree(cls, frame, ctx):
        return cls._to_tree(frame, ctx)

    @classmethod
    def assert_equal(cls, old, new):
        assert old.name == new.name  # nosec
        assert old.axes_order == new.axes_order  # nosec
        assert old.axes_names == new.axes_names  # nosec
        assert old.unit == new.unit  # nosec

        assert old.reference_frame == new.reference_frame  # nosec


class StokesFrameType(FrameType):
    name = "stokes_frame"
    types = [StokesFrame]

    @classmethod
    def from_tree(cls, node, ctx):
        node = cls._from_tree(node, ctx)
        return StokesFrame(**node)

    @classmethod
    def _to_tree(cls, frame, ctx):

        node = {}

        node['name'] = frame.name
        if frame.axes_order:
            node['axes_order'] = list(frame.axes_order)

        return node

    @classmethod
    def assert_equal(cls, old, new):
        from asdf.tests import helpers
        assert old.name == new.name  # nosec
        assert old.axes_order == new.axes_order  # nosec
