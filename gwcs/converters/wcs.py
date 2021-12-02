# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-

from asdf.extension import Converter


__all__ = ["WCSConverter", "CelestialFrameConverter", "CompositeFrameConverter",
           "FrameConverter", "SpectralFrameConverter", "StepConverter",
           "TemporalFrameConverter", "StokesFrameConverter"]


class WCSConverter(Converter):
    tags = ["tag:stsci.edu:gwcs/wcs-*"]
    types = ["gwcs.wcs.WCS"]

    def from_yaml_tree(self, node, tag, ctx):
        from ..wcs import WCS
        return WCS(node['steps'], name=node['name'])

    def to_yaml_tree(self, gwcsobj, tag, ctx):
        return {
            'name': gwcsobj.name,
            'steps': gwcsobj.pipeline
        }


class StepConverter(Converter):
    tags = ["tag:stsci.edu:gwcs/step-*"]
    types = ["gwcs.wcs.Step"]

    def from_yaml_tree(self, node, tag, ctx):
        from ..wcs import Step
        return Step(frame=node['frame'], transform=node.get('transform', None))

    def to_yaml_tree(self, step, tag, ctx):
        return {
            'frame': step.frame,
            'transform': step.transform
        }


class FrameConverter(Converter):
    tags = ["tag:stsci.edu:gwcs/frame-*"]
    types = ["gwcs.coordinate_frames.CoordinateFrame"]

    def _from_yaml_tree(self, node, tag, ctx):
        kwargs = {'name': node['name']}

        if 'axes_type' in node and 'naxes' in node:
            kwargs.update({
                'axes_type': node['axes_type'],
                'naxes': node['naxes']})

        if 'axes_names' in node:
            kwargs['axes_names'] = node['axes_names']

        if 'reference_frame' in node:
            kwargs['reference_frame'] = node['reference_frame']

        if 'axes_order' in node:
            kwargs['axes_order'] = tuple(node['axes_order'])

        if 'unit' in node:
            kwargs['unit'] = tuple(node['unit'])

        if 'axis_physical_types' in node:
            kwargs['axis_physical_types'] = tuple(node['axis_physical_types'])

        return kwargs

    def _to_yaml_tree(self, frame, tag, ctx):
        from ..coordinate_frames import CoordinateFrame

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
            node['reference_frame'] = frame.reference_frame

        if frame.unit is not None:
            node['unit'] = list(frame.unit)

        if frame.axis_physical_types is not None:
            node['axis_physical_types'] = list(frame.axis_physical_types)

        return node

    def from_yaml_tree(self, node, tag, ctx):
        from ..coordinate_frames import CoordinateFrame
        node = self._from_yaml_tree(node, tag, ctx)
        return CoordinateFrame(**node)

    def to_yaml_tree(self, frame, tag, ctx):
        return self._to_yaml_tree(frame, tag, ctx)


class Frame2DConverter(FrameConverter):
    tags = ["tag:stsci.edu:gwcs/frame2d-*"]
    types = ["gwcs.coordinate_frames.Frame2D"]

    def from_yaml_tree(self, node, tag, ctx):
        from ..coordinate_frames import Frame2D
        node = self._from_yaml_tree(node, tag, ctx)
        return Frame2D(**node)


class CelestialFrameConverter(FrameConverter):
    tags = ["tag:stsci.edu:gwcs/celestial_frame-*"]
    types = ["gwcs.coordinate_frames.CelestialFrame"]

    def from_yaml_tree(self, node, tag, ctx):
        from ..coordinate_frames import CelestialFrame
        node = self._from_yaml_tree(node, tag, ctx)
        return CelestialFrame(**node)


class SpectralFrameConverter(FrameConverter):
    tags = ["tag:stsci.edu:gwcs/spectral_frame-*"]
    types = ["gwcs.coordinate_frames.SpectralFrame"]

    def from_yaml_tree(self, node, tag, ctx):
        from ..coordinate_frames import SpectralFrame
        node = self._from_yaml_tree(node, tag, ctx)

        if 'reference_position' in node:
            node['reference_position'] = node['reference_position'].upper()

        return SpectralFrame(**node)

    def to_yaml_tree(self, frame, tag, ctx):
        node = self._to_yaml_tree(frame, tag, ctx)

        if frame.reference_position is not None:
            node['reference_position'] = frame.reference_position.lower()

        return node


class CompositeFrameConverter(FrameConverter):
    tags = ["tag:stsci.edu:gwcs/composite_frame-*"]
    types = ["gwcs.coordinate_frames.CompositeFrame"]

    def from_yaml_tree(self, node, tag, ctx):
        from ..coordinate_frames import CompositeFrame
        if len(node) != 2:
            raise ValueError("CompositeFrame has extra properties")

        name = node['name']
        frames = node['frames']

        return CompositeFrame(frames, name)

    def to_yaml_tree(self, frame, tag, ctx):
        return {
            'name': frame.name,
            'frames': frame.frames
        }


class TemporalFrameConverter(FrameConverter):
    tags = ["tag:stsci.edu:gwcs/temporal_frame-*"]
    types = ["gwcs.coordinate_frames.TemporalFrame"]

    def from_yaml_tree(self, node, tag, ctx):
        from ..coordinate_frames import TemporalFrame
        node = self._from_yaml_tree(node, tag, ctx)
        return TemporalFrame(**node)


class StokesFrameConverter(FrameConverter):
    tags = ["tag:stsci.edu:gwcs/stokes_frame-*"]
    types = ["gwcs.coordinate_frames.StokesFrame"]

    def from_yaml_tree(self, node, tag, ctx):
        from ..coordinate_frames import StokesFrame
        node = self._from_yaml_tree(node, tag, ctx)
        return StokesFrame(**node)

    def to_yaml_tree(self, frame, tag, ctx):
        node = {}

        node['name'] = frame.name
        if frame.axes_order:
            node['axes_order'] = list(frame.axes_order)

        return node
