# Module to wrap the old WCS in the gWCS framework

from astropy.modeling import FittableModel, Parameter


from astropy.wcs import WCS



def _get_possible_wcs_parameters(wcs):
    """
    get possible Parameters
    :param wcs:
    :return:
    """
    return [item.replace('get_', '') for item in wcs.__class__.__dict__
            if item.startswith('get_')]


def read_fits_wcs(header=None):
    """
    Parameters
    ----------
        header : astropy.io.fits header object, string, dict-like, or None, optional
            If *header* is not provided or None, the object will be
            initialized to default values.
    """
    legacy_wcs = WCS(header)


    possible_parameters = _get_possible_wcs_parameters(legacy_wcs.wcs)

    param_dict = {}
    class_dict = {}
    for param in possible_parameters:
        param_value = getattr(legacy_wcs.wcs, 'get_{0}'.format(param))()
        if param_value == []:
            continue

        param_dict[param] = param_value
        class_dict[param] = Parameter()

    class_dict['__init__'] = BaseLegacyFITSWCS.__init__

    class_dict['inputs'] = ('x', 'y')
    class_dict['outputs'] = ('a', 'b')

    wcs_legacy_model = type('LegacyWCS',(LegacyFITSWCSPix2World, ), class_dict)


    print param_dict

    return wcs_legacy_model(legacy_wcs, **param_dict)


class BaseLegacyFITSWCS(FittableModel):

    def __init__(self, legacy_wcs, **kwargs):
        super(BaseLegacyFITSWCS, self).__init__(**kwargs)
        self.legacy_wcs = legacy_wcs


    def _set_wcs_parameters(self, *args):
        self.legacy_wcs.set("these models params in here args")


class LegacyFITSWCSPix2World(BaseLegacyFITSWCS):

    def evaluate(self, x, y, *args):
        self._set_wcs_parameters(*args)
        return self.legacy_wcs.wcs_pix2world(*args)

    @property
    def inverse(self):
        return LegacyFITSWCSWorld2Pix(self.legacy_wcs)

class LegacyFITSWCSWorld2Pix(BaseLegacyFITSWCS):

    def evaluate(self, x, y, *args):
        self._set_wcs_parameters(*args)
        return self.legacy_wcs.wcs_world2pix(*args)

