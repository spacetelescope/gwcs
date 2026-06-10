# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Spectroscopy related models.
"""

import astropy.units as u
import numpy as np
from astropy.modeling import CompoundModel, custom_model
from astropy.modeling import models as m
from astropy.modeling.core import Model
from astropy.modeling.parameters import Parameter

__all__ = [
    "AnglesFromGratingEquation3D",
    "SellmeierGlass",
    "SellmeierZemax",
    "Snell3D",
    "WavelengthFromGratingEquation",
]


class WavelengthFromGratingEquation(Model):
    r"""Solve the Grating Dispersion Law for the wavelength.

    .. Note:: This form of the equation can be used for paraxial
      (small angle approximation) as well as oblique incident angles.
      With paraxial systems the inputs are ``sin`` of the angles and it
      transforms to

      :math:`(\sin(\alpha_{in}) + \sin(\alpha_{out})) / (groove\_density * spectral\_order)`.

      With oblique angles the inputs are the direction cosines
      of the angles.

    In addition to evaluating wavelength from grating-equation inputs, this
    class also provides helper methods for constructing FITS ``-GRA``/``-GRI``
    spectral transforms:

    - `~gwcs.spectroscopy.WavelengthFromGratingEquation.refracted_angle_sine_model`
      builds the pixel-dependent refracted-angle sine custom model
    - `~gwcs.spectroscopy.WavelengthFromGratingEquation.generate_grating_spectral_transform`
      assembles the corresponding one-dimensional compound spectral transform


    Parameters
    ----------
    groove_density : int
        Grating ruling density in units of 1/length.
    spectral_order : int
        Spectral order.

    Examples
    --------
    >>> from astropy.modeling.models import math
    >>> model = WavelengthFromGratingEquation(groove_density=20000*1/u.m, spectral_order=-1)
    >>> alpha_in = (math.Deg2radUfunc() | math.SinUfunc())(.0001 * u.deg)
    >>> alpha_out = (math.Deg2radUfunc() | math.SinUfunc())(.0001 * u.deg)
    >>> lam = model(alpha_in, alpha_out)
    >>> print(lam)
    -1.7453292519934437e-10 m

    """  # noqa: E501

    _separable = False

    linear = False

    n_inputs = 2
    n_outputs = 1

    groove_density = Parameter(default=1)
    """ Grating ruling density in units of 1/m."""
    spectral_order = Parameter(default=1)
    """ Spectral order."""

    def __init__(self, groove_density, spectral_order, **kwargs):
        super().__init__(
            groove_density=groove_density, spectral_order=spectral_order, **kwargs
        )
        self.inputs = ("alpha_in", "alpha_out")
        """ Sine function of the angles or the direction cosines."""
        self.outputs = ("wavelength",)
        """ Wavelength."""

    def evaluate(self, alpha_in, alpha_out, groove_density, spectral_order):
        return (alpha_in + alpha_out) / (groove_density * spectral_order)

    @staticmethod
    def refracted_angle_sine_model(
        reference_pixel: float,
        reference_wavelength: u.Quantity,
        dispersion: u.Quantity,
        grating_density: u.Quantity,
        spectral_order: u.Quantity,
        incident_angle: u.Quantity,
        refractive_index: u.Quantity,
        refractive_index_derivative: u.Quantity,
        out_of_plane_angle: u.Quantity,
        camera_angle: u.Quantity,
    ) -> Model:
        """
        Build the pixel-dependent refracted-angle sine model for FITS grating WCS.
        """
        grism_constant = (grating_density * spectral_order) / np.cos(out_of_plane_angle)
        reference_refracted_angle = np.arcsin(
            (grism_constant * reference_wavelength)
            - refractive_index * np.sin(incident_angle)
        )
        grism_parameter_per_wavelength = (
            grism_constant
            - refractive_index_derivative * np.sin(incident_angle)
        ) / (np.cos(reference_refracted_angle) * np.cos(camera_angle) ** 2)

        @custom_model
        def refracted_angle_sine(pixel):
            wavelength_offset = ((pixel - reference_pixel) * u.pix) * dispersion
            output_angle = (
                np.arctan(
                    -np.tan(camera_angle)
                    + wavelength_offset * grism_parameter_per_wavelength
                )
                + reference_refracted_angle
                + camera_angle
            )
            return np.sin(output_angle)

        return refracted_angle_sine()

    @staticmethod
    def generate_grating_spectral_transform(
        reference_pixel: float,
        reference_wavelength: u.Quantity,
        dispersion: u.Quantity,
        grating_density: u.Quantity,
        spectral_order: u.Quantity,
        incident_angle: u.Quantity,
        refractive_index: u.Quantity = 1 * u.one,
        refractive_index_derivative: u.Quantity = 0 / u.m,
        out_of_plane_angle: u.Quantity = 0 * u.deg,
        camera_angle: u.Quantity = 0 * u.deg,
    ) -> CompoundModel:
        """
        Build a one-dimensional FITS ``-GRA``/``-GRI`` spectral transform.

        This assembles the compound spectral model by combining a constant model
        for the adjusted incident-angle sine, a pixel-dependent refracted-angle
        sine model, and `WavelengthFromGratingEquation` to compute wavelength
        from the grating equation.
        """
        adjusted_incident_angle_sine = (
            refractive_index - refractive_index_derivative * reference_wavelength
        ) * np.sin(incident_angle)
        adjusted_groove_density = (
            (grating_density * spectral_order) / np.cos(out_of_plane_angle)
            - refractive_index_derivative * np.sin(incident_angle)
        ) / spectral_order

        refracted_angle = WavelengthFromGratingEquation.refracted_angle_sine_model(
            reference_pixel=reference_pixel,
            reference_wavelength=reference_wavelength,
            dispersion=dispersion,
            grating_density=grating_density,
            spectral_order=spectral_order,
            incident_angle=incident_angle,
            refractive_index=refractive_index,
            refractive_index_derivative=refractive_index_derivative,
            out_of_plane_angle=out_of_plane_angle,
            camera_angle=camera_angle,
        )
        incident_angle = m.Const1D(amplitude=adjusted_incident_angle_sine)
        wavelength_from_grating = WavelengthFromGratingEquation(
            groove_density=adjusted_groove_density,
            spectral_order=spectral_order,
            name="Spectral",
        )

        return m.Mapping((0, 0)) | (incident_angle & refracted_angle) | wavelength_from_grating

    @property
    def return_units(self):
        if self.groove_density.unit is None:
            return None
        return {"wavelength": u.Unit(1 / self.groove_density.unit)}


class AnglesFromGratingEquation3D(Model):
    """
    Solve the 3D Grating Dispersion Law in Direction Cosine
    space for the refracted angle.

    Parameters
    ----------
    groove_density : int
        Grating ruling density in units of 1/m.
    order : int
        Spectral order.

    Examples
    --------
    >>> from astropy.modeling.models import math
    >>> model = AnglesFromGratingEquation3D(groove_density=20000*1/u.m, spectral_order=-1)
    >>> alpha_in = (math.Deg2radUfunc() | math.SinUfunc())(.0001 * u.deg)
    >>> beta_in = (math.Deg2radUfunc() | math.SinUfunc())(.0001 * u.deg)
    >>> lam = 2e-6 * u.m
    >>> alpha_out, beta_out, gamma_out = model(lam, alpha_in, beta_in)
    >>> print(alpha_out, beta_out, gamma_out)
    0.04000174532925199 -1.7453292519934436e-06 0.9991996098716049

    """  # noqa: E501

    _separable = False
    linear = False

    n_inputs = 3
    n_outputs = 3

    groove_density = Parameter(default=1)
    """ Grating ruling density in units 1/ length."""

    spectral_order = Parameter(default=1)
    """ Spectral order."""

    def __init__(self, groove_density, spectral_order, **kwargs):
        super().__init__(
            groove_density=groove_density, spectral_order=spectral_order, **kwargs
        )
        self.inputs = ("wavelength", "alpha_in", "beta_in")
        """ Wavelength and 2 angle coordinates going into the grating."""

        self.outputs = ("alpha_out", "beta_out", "gamma_out")
        """ Two angles coming out of the grating. """

    def evaluate(self, wavelength, alpha_in, beta_in, groove_density, spectral_order):
        if alpha_in.shape != beta_in.shape:
            msg = "Expected input arrays to have the same shape."
            raise ValueError(msg)

        if isinstance(groove_density, u.Quantity):
            alpha_in = u.Quantity(alpha_in)
            beta_in = u.Quantity(beta_in)

        alpha_out = -groove_density * spectral_order * wavelength + alpha_in
        beta_out = -beta_in
        gamma_out = np.sqrt(1 - alpha_out**2 - beta_out**2)
        return alpha_out, beta_out, gamma_out

    @property
    def input_units(self):
        if self.groove_density.unit is None:
            return None
        return {
            "wavelength": 1 / self.groove_density.unit,
            "alpha_in": u.Unit(1),
            "beta_in": u.Unit(1),
        }


class Snell3D(Model):
    """
    Snell model in 3D form.

    Inputs are index of refraction and direction cosines.

    Returns
    -------
    alpha_out, beta_out, gamma_out : float
        Direction cosines.
    """

    _separable = False
    linear = False

    n_inputs = 4
    n_outputs = 3

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inputs = ("n", "alpha_in", "beta_in", "gamma_in")
        self.outputs = ("alpha_out", "beta_out", "gamma_out")

    @staticmethod
    def evaluate(n, alpha_in, beta_in, gamma_in):
        # Apply Snell's law through front surface,
        # eq 5.3.3 II in Nirspec docs
        alpha_out = alpha_in / n
        beta_out = beta_in / n
        gamma_out = np.sqrt(1.0 - alpha_out**2 - beta_out**2)
        return alpha_out, beta_out, gamma_out


class SellmeierGlass(Model):
    """
    Sellmeier equation for glass.

    Parameters
    ----------
    B_coef : ndarray
        Iterable of size 3 containing B coefficients.
    C_coef : ndarray
        Iterable of size 3 containing c coefficients in
        units of ``u.um**2``.

    Returns
    -------
    n : float
        Refractive index.

    Examples
    --------
    >>> import astropy.units as u
    >>> b_coef = [0.58339748, 0.46085267, 3.8915394]
    >>> c_coef = [0.00252643, 0.010078333, 1200.556] * u.um**2
    >>> model = SellmeierGlass(b_coef, c_coef)
    >>> model(2 * u.m)
        <Quantity 2.43634758>

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Sellmeier_equation

    Notes
    -----

    Model formula:

        .. math::

            n(\\lambda)^2 = 1 + \\frac{(B1 * \\lambda^2 )}{(\\lambda^2 - C1)} +
            \\frac{(B2 * \\lambda^2 )}{(\\lambda^2 - C2)} +
            \\frac{(B3 * \\lambda^2 )}{(\\lambda^2 - C3)}

    """

    _separable = False
    standard_broadcasting = False
    linear = False

    n_inputs = 1
    n_outputs = 1

    B_coef = Parameter(default=np.array([1, 1, 1]))
    """ B1, B2, B3 coefficients. """
    C_coef = Parameter(default=np.array([0, 0, 0]))
    """ C1, C2, C3 coefficients in units of um ** 2. """

    def __init__(self, B_coef, C_coef, **kwargs):
        super().__init__(B_coef, C_coef)
        self.inputs = ("wavelength",)
        self.outputs = ("n",)

    @staticmethod
    def evaluate(wavelength, B_coef, C_coef):
        B1, B2, B3 = B_coef[0]
        C1, C2, C3 = C_coef[0]

        return np.sqrt(
            1.0
            + B1 * wavelength**2 / (wavelength**2 - C1)
            + B2 * wavelength**2 / (wavelength**2 - C2)
            + B3 * wavelength**2 / (wavelength**2 - C3)
        )

    @property
    def input_units(self):
        if self.C_coef.unit is None:
            return None
        return {"wavelength": u.um}


class SellmeierZemax(Model):
    """
    Sellmeier equation used by Zemax.

    Parameters
    ----------
    temperature : float
        Temperature of the material in ``u.Kelvin``.
    ref_temperature : float
        Reference emperature of the glass in ``u.Kelvin``.
    ref_pressure : float
        Reference pressure in ATM.
    pressure : float
        Measured pressure in ATM.
    B_coef : ndarray
        Iterable of size 3 containing B coefficients.
    C_coef : ndarray
        Iterable of size 3 containing C coefficients in
        units of ``u.um**2``.
    D_coef : ndarray
        Iterable of size 3 containing constants to describe the
        behavior of the material.
    E_coef : ndarray
        Iterable of size 3 containing constants to describe the
        behavior of the material.

    Returns
    -------
    n : float
        Refractive index.

    """

    _separable = False
    standard_broadcasting = False
    linear = False

    n_inputs = 1
    n_outputs = 1

    temperature = Parameter(default=0)
    ref_temperature = Parameter(default=0)
    ref_pressure = Parameter(default=0)
    pressure = Parameter(default=0)
    B_coef = Parameter(default=[1, 1, 1])
    C_coef = Parameter(default=[0, 0, 0])
    D_coef = Parameter(default=[0, 0, 0])
    E_coef = Parameter(default=[1, 1, 1])

    def __init__(
        self,
        temperature=temperature,
        ref_temperature=ref_temperature,
        ref_pressure=ref_pressure,
        pressure=pressure,
        B_coef=B_coef,
        C_coef=C_coef,
        D_coef=D_coef,
        E_coef=E_coef,
        **kwargs,
    ):
        super().__init__(
            temperature=temperature,
            ref_temperature=ref_temperature,
            ref_pressure=ref_pressure,
            pressure=pressure,
            B_coef=B_coef,
            C_coef=C_coef,
            D_coef=D_coef,
            E_coef=E_coef,
            **kwargs,
        )
        self.inputs = ("wavelength",)
        self.outputs = ("n",)

    def evaluate(
        self,
        wavelength,
        temp,
        ref_temp,
        ref_pressure,
        pressure,
        B_coef,
        C_coef,
        D_coef,
        E_coef,
    ):
        """
        Input ``wavelength`` is in units of microns.
        """
        if isinstance(temp, u.Quantity):
            temp = temp.to(u.Celsius)
            ref_temp = ref_temp.to(u.Celsius)
        else:
            KtoC = 273.15  # kelvin to celsius conversion
            temp -= KtoC
            ref_temp -= KtoC
        delta = temp - ref_temp
        D0, D1, D2 = D_coef[0]
        E0, E1, lam_tk = E_coef[0]

        nref = (
            1.0
            + (
                6432.8
                + 2949810.0 * wavelength**2 / (146.0 * wavelength**2 - 1.0)
                + (5540.0 * wavelength**2) / (41.0 * wavelength**2 - 1.0)
            )
            * 1e-8
        )
        # T should be in C, P should be in ATM
        nair_obs = 1.0 + ((nref - 1.0) * pressure) / (1.0 + (temp - 15.0) * 3.4785e-3)
        nair_ref = 1.0 + ((nref - 1.0) * ref_pressure) / (
            1.0 + (ref_temp - 15) * 3.4785e-3
        )

        # Compute the relative index of the glass at Tref and Pref using
        # Sellmeier equation I.
        lamrel = wavelength * nair_obs / nair_ref
        nrel = SellmeierGlass.evaluate(lamrel, B_coef, C_coef)
        # Convert the relative index of refraction at the reference temperature
        # and pressure to absolute.
        nabs_ref = nrel * nair_ref

        # Compute the absolute index of the glass
        delnabs = (0.5 * (nrel**2 - 1.0) / nrel) * (
            D0 * delta
            + D1 * delta**2
            + D2 * delta**3
            + (E0 * delta + E1 * delta**2) / (lamrel**2 - lam_tk**2)
        )
        nabs_obs = nabs_ref + delnabs

        # Define the relative index at the system's operating T and P.
        return nabs_obs / nair_obs
