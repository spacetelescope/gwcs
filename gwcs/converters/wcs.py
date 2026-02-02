# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import annotations

import warnings
from contextlib import suppress
from typing import TYPE_CHECKING

from asdf.extension import Converter
from asdf_astropy.converters.transform.core import (
    TransformConverterBase,
    parameter_to_value,
)

if TYPE_CHECKING:
    from gwcs.coordinate_frames import CoordinateFrame

__all__ = [
    "CelestialFrameConverter",
    "CompositeFrameConverter",
    "FITSImagingWCSConverter",
    "Frame2DConverter",
    "FrameConverter",
    "SpectralFrameConverter",
    "StepConverter",
    "StokesFrameConverter",
    "TemporalFrameConverter",
    "WCSConverter",
]


class WCSConverter(Converter):
    tags = ("tag:stsci.edu:gwcs/wcs-*",)
    types = ("gwcs.wcs.WCS",)

    def from_yaml_tree(self, node, tag, ctx):
        from gwcs.wcs import WCS, GwcsBoundingBoxWarning

        gwcsobj = WCS(node["steps"], name=node["name"])
        if "pixel_shape" in node:
            gwcsobj.pixel_shape = node["pixel_shape"]

        # Ignore the warning about the bounding box order for data read from a
        # file. This is causing issues with files from MAST.
        with suppress(AttributeError), warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=GwcsBoundingBoxWarning)
            _ = gwcsobj.bounding_box

        return gwcsobj

    def to_yaml_tree(self, gwcsobj, tag, ctx):
        return {
            "name": gwcsobj.name,
            "steps": gwcsobj.pipeline,
            "pixel_shape": gwcsobj.pixel_shape,
        }


class StepConverter(Converter):
    tags = ("tag:stsci.edu:gwcs/step-*",)
    types = ("gwcs.wcs.Step",)

    def from_yaml_tree(self, node, tag, ctx):
        from gwcs.coordinate_frames import EmptyFrame
        from gwcs.wcs import Step

        frame: CoordinateFrame | str = node["frame"]
        if isinstance(frame, str):
            frame = EmptyFrame(name=frame)

        return Step(frame=frame, transform=node.get("transform", None))

    def to_yaml_tree(self, step, tag, ctx):
        from gwcs.coordinate_frames import EmptyFrame

        return {
            "frame": step.frame.name
            if isinstance(step.frame, EmptyFrame)
            else step.frame,
            "transform": step.transform,
        }


class FrameConverter(Converter):
    tags = ("tag:stsci.edu:gwcs/frame-*",)
    types = ("gwcs.coordinate_frames.CoordinateFrame",)

    def _from_yaml_tree(self, node, tag, ctx):
        kwargs = {"name": node["name"]}

        if "axes_type" in node and "naxes" in node:
            kwargs.update({"axes_type": node["axes_type"], "naxes": node["naxes"]})

        if "axes_names" in node:
            kwargs["axes_names"] = node["axes_names"]

        if "reference_frame" in node:
            kwargs["reference_frame"] = node["reference_frame"]

        if "axes_order" in node:
            kwargs["axes_order"] = tuple(node["axes_order"])

        if "unit" in node:
            kwargs["unit"] = tuple(node["unit"])

        if "axis_physical_types" in node:
            kwargs["axis_physical_types"] = tuple(node["axis_physical_types"])

        return kwargs

    def _to_yaml_tree(self, frame: CoordinateFrame, tag, ctx):
        from gwcs.coordinate_frames import CoordinateFrame

        node = {}

        node["name"] = frame.name

        # We want to check that it is exactly this type and not a subclass
        if type(frame) is CoordinateFrame:
            node["axes_type"] = [
                str(axis_type) for axis_type in frame.raw_properties.axes_type
            ]
            node["naxes"] = frame.naxes

        if frame.axes_order is not None:
            node["axes_order"] = list(frame.axes_order)

        if frame.raw_properties.axes_names is not None:
            node["axes_names"] = list(frame.raw_properties.axes_names)

        if frame.reference_frame is not None:
            node["reference_frame"] = frame.reference_frame

        if frame.raw_properties.unit is not None:
            node["unit"] = list(frame.raw_properties.unit)

        if frame.raw_properties.axis_physical_types is not None:
            node["axis_physical_types"] = list(frame.raw_properties.axis_physical_types)

        return node

    def from_yaml_tree(self, node, tag, ctx):
        from gwcs.coordinate_frames import CoordinateFrame

        node = self._from_yaml_tree(node, tag, ctx)
        return CoordinateFrame(**node)

    def to_yaml_tree(self, frame, tag, ctx):
        return self._to_yaml_tree(frame, tag, ctx)


class Frame2DConverter(FrameConverter):
    tags = ("tag:stsci.edu:gwcs/frame2d-*",)
    types = ("gwcs.coordinate_frames.Frame2D",)

    def from_yaml_tree(self, node, tag, ctx):
        from gwcs.coordinate_frames import Frame2D

        node = self._from_yaml_tree(node, tag, ctx)
        return Frame2D(**node)


class CelestialFrameConverter(FrameConverter):
    tags = ("tag:stsci.edu:gwcs/celestial_frame-*",)
    types = ("gwcs.coordinate_frames.CelestialFrame",)

    def from_yaml_tree(self, node, tag, ctx):
        from gwcs.coordinate_frames import CelestialFrame

        node = self._from_yaml_tree(node, tag, ctx)
        return CelestialFrame(**node)


class SpectralFrameConverter(FrameConverter):
    tags = ("tag:stsci.edu:gwcs/spectral_frame-*",)
    types = ("gwcs.coordinate_frames.SpectralFrame",)

    def from_yaml_tree(self, node, tag, ctx):
        from gwcs.coordinate_frames import SpectralFrame

        node = self._from_yaml_tree(node, tag, ctx)

        return SpectralFrame(**node)


class CompositeFrameConverter(FrameConverter):
    tags = ("tag:stsci.edu:gwcs/composite_frame-*",)
    types = ("gwcs.coordinate_frames.CompositeFrame",)

    def from_yaml_tree(self, node, tag, ctx):
        from gwcs.coordinate_frames import CompositeFrame

        if len(node) != 2:
            msg = "CompositeFrame has extra properties"
            raise ValueError(msg)

        name = node["name"]
        frames = node["frames"]

        return CompositeFrame(frames, name)

    def to_yaml_tree(self, frame, tag, ctx):
        return {"name": frame.name, "frames": frame.frames}


class TemporalFrameConverter(FrameConverter):
    tags = ("tag:stsci.edu:gwcs/temporal_frame-*",)
    types = ("gwcs.coordinate_frames.TemporalFrame",)

    def from_yaml_tree(self, node, tag, ctx):
        from gwcs.coordinate_frames import TemporalFrame

        node = self._from_yaml_tree(node, tag, ctx)
        return TemporalFrame(**node)


class StokesFrameConverter(FrameConverter):
    tags = ("tag:stsci.edu:gwcs/stokes_frame-*",)
    types = ("gwcs.coordinate_frames.StokesFrame",)

    def from_yaml_tree(self, node, tag, ctx):
        from gwcs.coordinate_frames import StokesFrame

        node = self._from_yaml_tree(node, tag, ctx)
        return StokesFrame(**node)

    def to_yaml_tree(self, frame, tag, ctx):
        node = {}

        node["name"] = frame.name
        if frame.axes_order:
            node["axes_order"] = list(frame.axes_order)

        return node


class FITSImagingWCSConverter(TransformConverterBase):
    tags = ("tag:stsci.edu:gwcs/fitswcs_imaging-*",)
    types = ("gwcs.fitswcs.FITSImagingWCSTransform",)

    def from_yaml_tree_transform(self, node, tag, ctx):
        from gwcs.fitswcs import FITSImagingWCSTransform

        return FITSImagingWCSTransform(
            node["projection"],
            crpix=node["crpix"],
            crval=node["crval"],
            cdelt=node["cdelt"],
            pc=node["pc"],
        )

    def to_yaml_tree_transform(self, model, tag, ctx):
        return {
            "crpix": parameter_to_value(model.crpix),
            "crval": parameter_to_value(model.crval),
            "cdelt": parameter_to_value(model.cdelt),
            "pc": parameter_to_value(model.pc),
            "projection": model.projection,
        }
