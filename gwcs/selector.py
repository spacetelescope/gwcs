from __future__ import division, print_function

import copy
import numpy as np
import warnings

from astropy.modeling.core import Model
from astropy.utils.misc import isiterable


from .util import RegionError


class SelectorModel(Model):
    """
    Parameters
    ----------
    labels : a list of strings or objects
    transforms : a list of transforms
        transforms match labels
    """
    _param_names = ()

    linear = False
    fittable = False

    def __init__(self, labels, transforms):
        # labels should be unique
        if (np.unique(labels)).size != np.asarray(labels).size:
            raise ValueError("Labels should be unique.")
        # every label should have a corresponding transform
        if len(labels) != len(transforms):
            raise ValueError("Expected labels and transforms to be of equal length.")
        if " " in labels or 0 in labels:
            raise ValueError('"0" and " " are not allowed as labels.')
        self._selector = dict(zip(labels, transforms))
        super(SelectorModel, self).__init__(n_models=1)

    @property
    def labels(self):
        return self._selector.keys()

    @property
    def transforms(self):
        return self._selector.values()



class RegionsSelector(SelectorModel):
    """
    Parameters
    ----------
    regions_mask : ndarray
        An array where regions are indicated by int or str labels.
        " " and 0 indicate a pixel on the detector which is not within any region.
        Evaluating the model in these locations returns NaN or ``undefined_transform_value`` if provided.
    labels : list
        A list of str or int labels in the same order as transforms.
        It does not contain 0 or " "  and matches all labels in regions_mask.
    transforms : list of transforms
        a transform is an instance of modeling.Model or a callable
        which performs the transformation from the region's coordinate system
        to some other coordinate system
    undefined_transform_value : float
        A value to be returned for locations not within any region.
        Default is np.NaN.

    """
    inputs = ('x', 'y')#, 'region')
    outputs = ('lat', 'lon', 'lambda')
    #inputs = ()
    #outputs = ()

    def __init__(self, regions_mask, labels, transforms, undefined_transform_value=np.nan, coord_sys='DetectorFrame'):
        self.regions_mask = regions_mask.copy()
        self.undefined_transform_value = undefined_transform_value
        # get the list of region labels
        labels_mask = self.labels_from_mask(regions_mask)
        if not np.in1d(labels_mask, labels, assume_unique=True).all() or \
           not np.in1d(labels, labels_mask, assume_unique=True).all():
            raise ValueError("Labels don't match regions_mask.")

        super(RegionsSelector, self).__init__(labels, transforms)#,
                                              #n_inputs=3, n_outputs=3)

    @staticmethod
    def labels_from_mask(regions_mask):
        """
        Parameters
        ----------
        regions_mask : ndarray
            An array where regions are indicated by int or str labels.
            " " and 0 indicate a pixel on the detector which is not within any region.
            Evaluating the model in these locations returns NaN or
            ``undefined_transform_value`` if provided.
        """
        labels = np.unique(regions_mask).tolist()
        try:
            labels.remove(0)
        except ValueError:
            pass
        try:
            labels.remove('')
        except ValueError:
            pass
        return labels

    @classmethod
    def reconstruct_wcs(cls, rid_wcs, wcs_info):
        reference_regions_wcs = copy.deepcopy(wcs_info)
        reference_regions_wcs['CRVAL'][0] += rid_wcs['CRVAL_DELTA'][0]
        reference_regions_wcs['CRVAL'][1] += rid_wcs['CRVAL_DELTA'][1]
        reference_regions_wcs['CRPIX'][0] += rid_wcs['CRPIX_DELTA'][0]
        reference_regions_wcs['CRPIX'][1] += rid_wcs['CRPIX_DELTA'][1]
        return reference_regions_wcs


    def evaluate(self, x, y):#region=None
        """
        Parameters
        ----------
        x : float
            Input pixel coordinate
        y : float
            Input pixel coordinate
        region : int or str
            Region id
        """
        #if region is not None:
        #if not np.isnan(region):
        #    return self._selector[region.item()](x, y)
        # use input_values for indexing the output arrays
        input_values = (x, y)
        idx = np.empty(x.shape, dtype=np.int)
        idy = np.empty(y.shape, dtype=np.int)
        idx = np.around(x, out=idx)
        idy = np.around(y, out=idy)
        result = self.regions_mask[idy, idx]
        if not isiterable(result) or isinstance(result, np.string_):
            if result != 0 and result != '':
                result = self._selector[result](*input_values)
                if np.isscalar(result[0]):
                    result = [np.array(ar) for ar in result]
                return result
            else:
                broadcast_missing_value = np.broadcast_arrays(
                    self.undefined_transform_value, input_values[0])[0]
                return [broadcast_missing_value for output in range(self.n_outputs)] #input_values

        unique_regions = self.get_unique_regions(result)
        if unique_regions == []:
            #warnings.warning('There are no regions with valid transforms.')
            #return
            raise RegionError("The input positions are not inside any region.")
        nout = self._selector[unique_regions[0]].n_outputs
        output_arrays = [np.zeros(self.regions_mask.shape, dtype=np.float) for i in range(nout)]
        for ar in output_arrays:
            ar[self.regions_mask == 0] = self.undefined_transform_value
        for i in unique_regions:
            transform = self._selector[i]
            indices = (self.regions_mask==i).nonzero()
            outputs = transform(*indices)
            for out, tr_out in zip(output_arrays, outputs):
                out[indices] = tr_out

        result = [ar[idx, idy] for ar in output_arrays]
        if np.isscalar(result[0]):
            result = [np.array(ar) for ar in result]
        return tuple(result)

    def __call__(self, *inputs, **kwargs):
        """
        Evaluate this model using the given input(s) and the parameter values
        that were specified when the model was instantiated.
        """
        import itertools
        if len(inputs) == 3:
            label = inputs[-1]
            try:
                tr = self._selector[label]
            except KeyError:
                raise RegionError("Region {0} does not exist".format(label))
            evaluate = tr.evaluate
            parameters = tr._param_sets(raw=True)
            inputs = inputs[:2]
        else:
            parameters = self._param_sets(raw=True)
            evaluate = self.evaluate
        inputs, format_info = self.prepare_inputs(*inputs, **kwargs)

        outputs = evaluate(*itertools.chain(inputs, parameters))

        if self.n_outputs == 1:
            outputs = (outputs,)
        return self.prepare_outputs(format_info, *outputs, **kwargs)

    def evaluate_region(self, region_id):
        transform = self._selector[region_id]
        indices = (self.regions_mask==region_id).nonzero()
        outputs = transform(*indices)
        for out, tr_out in zip(output_arrays, outputs):
            out[indices] = tr_out
        return out

    @staticmethod
    def get_unique_regions(mask):
        unique_regions = np.unique(mask).tolist()

        try:
            unique_regions.remove(0)
            unique_regions.remove('')
        except ValueError:
            pass
        try:
            unique_regions.remove("")
        except ValueError:
            pass
        return unique_regions

    def inverse(self, x, y):
        """
        Need to invert the regions and construct an InverseSelectorModel
        """
