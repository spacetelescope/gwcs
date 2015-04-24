# Module to wrap the old WCS in the gWCS framework

from astropy.modeling import FittableModel, Parameter
from astropy.wcs import WCS

from copy import deepcopy

from abc import ABCMeta, abstractproperty, abstractmethod

class LegacyWCSParameter(Parameter):

    @abstractproperty
    def wcs_name(self):
        raise NotImplementedError

    def __set__(self, instance, value):
        super(LegacyWCSParameter, self).__set__(instance, value)
        if self._model is not None:
            self.set_wcs_value(self._model.legacy_wcs.wcs, value)

class CDELT(LegacyWCSParameter):
    wcs_name = 'cdelt'

    @staticmethod
    def get_wcs_value(wcsprm):
        if wcsprm.has_cd:
            return None
        else:
            return wcsprm.cdelt

    def set_wcs_value(self):
        raise NotImplementedError()

class PC(LegacyWCSParameter):
    wcs_name = 'pc'

    @staticmethod
    def get_wcs_value(wcsprm):
        try:
            return wcsprm.pc
        except AttributeError:
            return None

    @staticmethod
    def set_wcs_value(wcsprm, value):
        wcsprm.pc = value

class CRVAL(LegacyWCSParameter):
    wcs_name = 'crval'

    @staticmethod
    def get_wcs_value(wcsprm):
        return wcsprm.crval

    @staticmethod
    def set_wcs_value(wcsprm, value):
        wcsprm.crval = value

class CD(LegacyWCSParameter):
    wcs_name = 'cd'

    @staticmethod
    def get_wcs_value(wcsprm):
        return wcsprm.cd

    @staticmethod
    def set_wcs_value(wcsprm, value):
        wcsprm.cd = value

legacy_wcs_parameters = LegacyWCSParameter.__subclasses__()

def read_fits_wcs(header=None, origin=0):
    """
    Parameters
    ----------
        header : astropy.io.fits header object, string, dict-like, or None, optional
            If *header* is not provided or None, the object will be
            initialized to default values.
    """
    legacy_wcs = WCS(header)

    class_dict = {}
    param_dict = {}

    for legacy_param_cls in legacy_wcs_parameters:
        legacy_param = legacy_param_cls()
        legacy_param_value = legacy_param.get_wcs_value(legacy_wcs.wcs)
        if legacy_param_value is None:
            continue
        else:
            class_dict[legacy_param.wcs_name] = legacy_param
            param_dict[legacy_param.wcs_name] = legacy_param_value

    class_dict['__init__'] = BaseLegacyFITSWCS.__init__

    class_dict['inputs'] = ('x', 'y')
    class_dict['outputs'] = ('a', 'b')

    wcs_legacy_model = type('LegacyWCS',(LegacyFITSWCSPix2World, ), class_dict)


    return wcs_legacy_model(legacy_wcs, origin=origin, **param_dict)


class BaseLegacyFITSWCS(FittableModel):
    standard_broadcasting = False

    def __init__(self, legacy_wcs, origin, **kwargs):
        super(BaseLegacyFITSWCS, self).__init__(**kwargs)
        self.legacy_wcs = legacy_wcs
        self.origin = origin

    @property
    def wcsprm(self):
        new_wcsprm = deepcopy(self.legacy_wcs.wcs)

        for param in self.parameter_descriptors:
            param.set_wcs_value(new_wcsprm, param.value)

        return new_wcsprm

    @property
    def parameter_descriptors(self):
        return [getattr(self, param_name) for param_name in self.param_names]


    def _set_wcs_parameters(self, *args):
        self.legacy_wcs.set("these models params in here args")


class LegacyFITSWCSPix2World(BaseLegacyFITSWCS):

    def evaluate(self, x, y, *args):
        self.legacy_wcs.wcs = self.wcsprm
        return self.legacy_wcs.wcs_pix2world(x, y, self.origin)

    @property
    def inverse(self):
        return LegacyFITSWCSWorld2Pix(self.legacy_wcs)

class LegacyFITSWCSWorld2Pix(BaseLegacyFITSWCS):

    def evaluate(self, x, y, *args):
        self._set_wcs_parameters(*args)
        return self.legacy_wcs.wcs_world2pix(*args)

