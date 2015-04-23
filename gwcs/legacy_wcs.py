# Module to wrap the old WCS in the gWCS framework

from astropy.modeling import FittableModel, Parameter


from astropy.wcs import WCS




def read_fits_wcs(header=None):
    """
    Parameters
    ----------
        header : astropy.io.fits header object, string, dict-like, or None, optional
            If *header* is not provided or None, the object will be
            initialized to default values.
    """
    legacy_wcs = WCS(header)


    possible_parameters = [item.replace('get_', '') for item in
                          legacy_wcs.wcs.__class__.__dict__.keys()
                          if item.startswith('get_')]

    param_dict = {}
    class_dict = {}
    for param in possible_parameters:
        param_value = getattr(legacy_wcs.wcs, 'get_{0}'.format(param))()
        if param_value == []:
            continue

        param_dict[param] = param_value
        class_dict[param] = Parameter()
    
    wcs_legacy_model = type('LegacyWCS',(LegacyFITSWCSModel, ), class_dict)

    return wcs_legacy_model(legacy_wcs, **param_dict)




class LegacyFITSWCSModel(FittableModel):

    def __init__(self, legacy_wcs, **kwargs):
        super(LegacyFITSWCSModel).__init__(**kwargs)
        self.legacy_wcs = legacy_wcs

    def evaluate(self, *args, **kwargs):
        self.legacy_wcs(*args)
