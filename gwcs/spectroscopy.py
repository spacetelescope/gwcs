# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Spectroscopy related models.
"""

import astropy.units as u
import numpy as np
from astropy.modeling.core import Model
from astropy.modeling.parameters import Parameter

__all__ = [
    "AnglesFromGratingEquation3D",
    "RefractedAngleSineModel",
    "SellmeierGlass",
    "SellmeierZemax",
    "Snell3D",
    "WavelengthFromGratingEquation",
]


class RefractedAngleSineModel(Model):
    """
    Compute the sine of the refracted angle for FITS ``-GRA``/``-GRI`` spectra.

    Given a pixel coordinate, converts it to a wavelength offset from the
    reference pixel, computes the corresponding refracted angle via the grating
    geometry, and returns its sine — suitable for feeding directly into
    `~gwcs.spectroscopy.WavelengthFromGratingEquation`.

    Parameters
    ----------
    reference_pixel : float
        Pixel coordinate of the reference point (0-indexed).
    reference_wavelength : float or `~astropy.units.Quantity`, optional
        Wavelength at the reference pixel. If a bare number is given, units of
        ``m`` are assumed. Defaults to ``0 m``.
    dispersion : float or `~astropy.units.Quantity`, optional
        Wavelength dispersion per pixel. If a bare number is given, units of
        ``m/pix`` are assumed. Defaults to ``0 m/pix``.
    groove_density : float or `~astropy.units.Quantity`, optional
        Grating ruling density in units of 1/length. If a bare number is
        given, units of ``1/m`` are assumed. Defaults to ``1 /m``.
    spectral_order : float or `~astropy.units.Quantity`, optional
        Spectral order. If a bare number is given, it is treated as
        dimensionless (``u.one``). Defaults to ``1``.
    incident_angle : float or `~astropy.units.Quantity`, optional
        Incident grating angle. If a bare number is given, units of degrees
        are assumed. Defaults to ``0 deg``.
    refractive_index : float or `~astropy.units.Quantity`, optional
        Refractive index at the reference wavelength. Dimensionless; if a bare
        number is given it is treated as dimensionless (``u.one``). Defaults
        to ``1``.
    refractive_index_derivative : float or `~astropy.units.Quantity`, optional
        Derivative of refractive index with respect to wavelength. If a bare
        number is given, units of ``1/m`` are assumed. Defaults to ``0 /m``.
    out_of_plane_angle : float or `~astropy.units.Quantity`, optional
        Out-of-plane grating angle. If a bare number is given, units of
        degrees are assumed. Defaults to ``0 deg``.
    camera_angle : float or `~astropy.units.Quantity`, optional
        Camera angle. If a bare number is given, units of degrees are
        assumed. Defaults to ``0 deg``.
    """

    _separable = False
    linear = False

    n_inputs = 1
    n_outputs = 1

    reference_pixel = Parameter(default=0, description="Reference pixel (0-indexed).")
    reference_wavelength = Parameter(
        default=0 * u.m, description="Reference wavelength."
    )
    dispersion = Parameter(
        default=0 * u.m / u.pix,
        description="Wavelength dispersion per pixel.",
    )
    groove_density = Parameter(default=1 / u.m, description="Grating ruling density.")
    spectral_order = Parameter(default=1, description="Spectral order.")
    incident_angle = Parameter(default=0 * u.deg, description="Incident grating angle.")
    refractive_index = Parameter(default=1, description="Refractive index.")
    refractive_index_derivative = Parameter(
        default=0 / u.m,
        description="Derivative of refractive index with respect to wavelength.",
    )
    out_of_plane_angle = Parameter(
        default=0 * u.deg,
        description="Out-of-plane grating angle.",
    )
    camera_angle = Parameter(
        default=0 * u.deg,
        description="Camera angle.",
    )

    def __init__(
        self,
        reference_pixel: float = 0,
        reference_wavelength: float | u.Quantity = 0 * u.m,
        dispersion: float | u.Quantity = 0 * u.m / u.pix,
        groove_density: float | u.Quantity = 1 / u.m,
        spectral_order: float | u.Quantity = 1,
        incident_angle: float | u.Quantity = 0 * u.deg,
        refractive_index: float | u.Quantity = 1,
        refractive_index_derivative: float | u.Quantity = 0 / u.m,
        out_of_plane_angle: float | u.Quantity = 0 * u.deg,
        camera_angle: float | u.Quantity = 0 * u.deg,
        **kwargs,
    ) -> None:
        # Coerce bare numbers to Quantities with assumed units. Assumed units
        # are documented in the Parameters section of the class docstring.
        if not isinstance(reference_wavelength, u.Quantity):
            reference_wavelength = reference_wavelength * u.m
        if not isinstance(dispersion, u.Quantity):
            dispersion = dispersion * u.m / u.pix
        if not isinstance(groove_density, u.Quantity):
            groove_density = groove_density / u.m
        if not isinstance(spectral_order, u.Quantity):
            spectral_order = spectral_order * u.one
        if not isinstance(incident_angle, u.Quantity):
            incident_angle = incident_angle * u.deg
        if not isinstance(refractive_index, u.Quantity):
            refractive_index = refractive_index * u.one
        if not isinstance(refractive_index_derivative, u.Quantity):
            refractive_index_derivative = refractive_index_derivative / u.m
        if not isinstance(out_of_plane_angle, u.Quantity):
            out_of_plane_angle = out_of_plane_angle * u.deg
        if not isinstance(camera_angle, u.Quantity):
            camera_angle = camera_angle * u.deg
        super().__init__(
            reference_pixel=reference_pixel,
            reference_wavelength=reference_wavelength,
            dispersion=dispersion,
            groove_density=groove_density,
            spectral_order=spectral_order,
            incident_angle=incident_angle,
            refractive_index=refractive_index,
            refractive_index_derivative=refractive_index_derivative,
            out_of_plane_angle=out_of_plane_angle,
            camera_angle=camera_angle,
            **kwargs,
        )
        self.inputs = ("pixel",)
        self.outputs = ("alpha_out",)

    @staticmethod
    def evaluate(
        pixel,
        reference_pixel,
        reference_wavelength,
        dispersion,
        groove_density,
        spectral_order,
        incident_angle,
        refractive_index,
        refractive_index_derivative,
        out_of_plane_angle,
        camera_angle,
    ):
        grism_constant = (groove_density * spectral_order) / np.cos(out_of_plane_angle)
        reference_refracted_angle = np.arcsin(
            (grism_constant * reference_wavelength)
            - refractive_index * np.sin(incident_angle)
        )
        grism_parameter_per_wavelength = (
            grism_constant - refractive_index_derivative * np.sin(incident_angle)
        ) / (np.cos(reference_refracted_angle) * np.cos(camera_angle) ** 2)
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


class WavelengthFromGratingEquation(Model):
    r"""Solve the Grating Dispersion Law for the wavelength.

    .. Note:: This form of the equation can be used for paraxial
      (small angle approximation) as well as oblique incident angles.
      With paraxial systems the inputs are ``sin`` of the angles and it
      transforms to

      :math:`(\sin(\alpha_{in}) + \sin(\alpha_{out})) / (groove\_density * spectral\_order)`.

      With oblique angles the inputs are the direction cosines
      of the angles.

    This model also supports the FITS ``-GRA``/``-GRI`` spectral-coordinate
    formalism when the optional grating configuration parameters are supplied.
    In that usage, the first input remains the incident-angle sine term and
    the second input remains the refracted-angle sine term, while the model
    applies the grism-specific correction terms internally. When the optional
    parameters are left at their default values, the original two-input
    grating-equation behavior is preserved.

    Parameters
    ----------
    groove_density : int or `~astropy.units.Quantity`
        Grating ruling density in units of 1/length. If a bare number is
        given, units of ``1/m`` are assumed.
    spectral_order : int or `~astropy.units.Quantity`
        Spectral order. If a bare number is given, it is treated as
        dimensionless (``u.one``).
    reference_wavelength : float or `~astropy.units.Quantity`, optional
        Wavelength at the reference pixel in units of ``m`` for FITS ``-GRA``/``-GRI`` mode.
        If a bare number is given, units of ``m`` are assumed.
        Defaults to ``0 m``.
    refractive_index : float or `~astropy.units.Quantity`, optional
        Refractive index at the reference wavelength for FITS
        ``-GRA``/``-GRI`` mode. Dimensionless; if a bare number is given it
        is treated as dimensionless (``u.one``). Defaults to ``1``.
    refractive_index_derivative : float or `~astropy.units.Quantity`, optional
        Derivative of refractive index with respect to wavelength for FITS
        ``-GRA``/``-GRI`` mode. If a bare number is given, units of ``1/m``
        are assumed. Defaults to ``0 /m``.
    out_of_plane_angle : float or `~astropy.units.Quantity`, optional
        Out-of-plane grating angle for FITS ``-GRA``/``-GRI`` mode. If a
        bare number is given, units of degrees are assumed.
        Defaults to ``0 deg``.

    Examples
    --------
    >>> from astropy.modeling.models import math
    >>> model = WavelengthFromGratingEquation(groove_density=20000*1/u.m, spectral_order=-1)
    >>> alpha_in = (math.Deg2radUfunc() | math.SinUfunc())(.0001 * u.deg)
    >>> alpha_out = (math.Deg2radUfunc() | math.SinUfunc())(.0001 * u.deg)
    >>> lam = model(alpha_in, alpha_out)
    >>> print(lam)
    -1.7453292519934437e-10 m

    # FITS ``-GRA``/``-GRI`` mode with externally computed alpha terms
    >>> groove_density = 23000 * 1 / u.m
    >>> spectral_order = 90 * u.one
    >>> alpha_out_model = RefractedAngleSineModel(
    ...     reference_pixel=217,
    ...     reference_wavelength=854.1738582455826 * u.nm,
    ...     dispersion=0.0022975580183395555 * u.nm / u.pix,
    ...     groove_density=23000 * 1 / u.m,
    ...     spectral_order=90 * u.one,
    ...     incident_angle=65.696 * u.deg,
    ...     refractive_index=1.25 * u.one,
    ...     refractive_index_derivative=1000 * 1 / u.m,
    ...     out_of_plane_angle=1.5 * u.deg,
    ...     camera_angle=0.8 * u.deg,
    ... )
    >>> wavelength_model = WavelengthFromGratingEquation(
    ...     groove_density=groove_density,
    ...     spectral_order=spectral_order,
    ...     reference_wavelength=854.1738582455826 * u.nm,
    ...     refractive_index=1.25 * u.one,
    ...     refractive_index_derivative=1000 * 1 / u.m,
    ...     out_of_plane_angle=1.5 * u.deg,
    ... )
    >>> pixels = np.array([0.0, 100.0, 217.0, 300.0, 511.0])
    >>> alpha_in = np.sin(65.696 * u.deg)
    >>> alpha_out = alpha_out_model(pixels)
    >>> lam = wavelength_model(alpha_in, alpha_out)
    >>> print(lam.to(u.nm))
    [853.6750296  853.90496873 854.17385825 854.36451764 854.84886375] nm

    """  # noqa: E501

    _separable = False

    linear = False

    n_inputs = 2
    n_outputs = 1

    groove_density = Parameter(default=1)
    """ Grating ruling density in units of 1/m."""
    spectral_order = Parameter(default=1)
    """ Spectral order."""
    reference_wavelength = Parameter(default=0 * u.m)
    """ Reference wavelength for FITS ``-GRA``/``-GRI`` mode."""
    refractive_index = Parameter(default=1)
    """ Refractive index at the reference wavelength."""
    refractive_index_derivative = Parameter(default=0 / u.m)
    """ Derivative of refractive index with respect to wavelength."""
    out_of_plane_angle = Parameter(default=0 * u.deg)
    """ Out-of-plane grating angle for FITS ``-GRA``/``-GRI`` mode."""

    def __init__(
        self,
        groove_density: float | u.Quantity,
        spectral_order: float | u.Quantity,
        reference_wavelength: float | u.Quantity = 0 * u.m,
        refractive_index: float | u.Quantity = 1,
        refractive_index_derivative: float | u.Quantity = 0 / u.m,
        out_of_plane_angle: float | u.Quantity = 0 * u.deg,
        **kwargs,
    ) -> None:
        # Coerce bare numbers to Quantities with assumed SI units so that
        # unit arithmetic in evaluate() is always valid. Assumed units are
        # documented in the Parameters section of the class docstring.
        if not isinstance(groove_density, u.Quantity):
            groove_density = groove_density / u.m
        if not isinstance(spectral_order, u.Quantity):
            spectral_order = spectral_order * u.one
        if not isinstance(reference_wavelength, u.Quantity):
            reference_wavelength = reference_wavelength * u.m
        if not isinstance(refractive_index, u.Quantity):
            refractive_index = refractive_index * u.one
        if not isinstance(refractive_index_derivative, u.Quantity):
            refractive_index_derivative = refractive_index_derivative / u.m
        if not isinstance(out_of_plane_angle, u.Quantity):
            out_of_plane_angle = out_of_plane_angle * u.deg
        super().__init__(
            groove_density=groove_density,
            spectral_order=spectral_order,
            reference_wavelength=reference_wavelength,
            refractive_index=refractive_index,
            refractive_index_derivative=refractive_index_derivative,
            out_of_plane_angle=out_of_plane_angle,
            **kwargs,
        )
        self.inputs = ("alpha_in", "alpha_out")
        """ Sine function of the angles or the direction cosines."""
        self.outputs = ("wavelength",)
        """ Wavelength."""

    def evaluate(
        self,
        alpha_in,
        alpha_out,
        groove_density,
        spectral_order,
        reference_wavelength,
        refractive_index,
        refractive_index_derivative,
        out_of_plane_angle,
    ):
        """
        Evaluate the grating equation or FITS grating-transform mode.

        In both grating-equation and FITS ``-GRA``/``-GRI`` usage,
        ``alpha_in`` and ``alpha_out`` are the direct model inputs. In FITS
        grating mode, the model applies the refractive-index and
        out-of-plane-angle corrections to the incident-angle term internally.

        Parameters
        ----------
        alpha_in : float
            Sine of the incident angle.
        alpha_out : float
            Sine of the refracted angle.
        groove_density : `~astropy.units.Quantity`
            Grating ruling density in units of ``1/m``.
        spectral_order : `~astropy.units.Quantity`
            Spectral order (dimensionless).
        reference_wavelength : `~astropy.units.Quantity`
            Wavelength at the reference pixel, in units of ``m``.
        refractive_index : `~astropy.units.Quantity`
            Refractive index at the reference wavelength (dimensionless).
        refractive_index_derivative : `~astropy.units.Quantity`
            Derivative of refractive index with respect to wavelength, in units
            of ``1/m``.
        out_of_plane_angle : `~astropy.units.Quantity`
            Out-of-plane grating angle, in units of ``deg``.

        Returns
        -------
        numpy.ndarray or astropy.units.Quantity
            The wavelength computed from the grating equation. Units are the inverse
            of ``groove_density`` units (e.g., meters if groove_density is in 1/m).
            Values are physically meaningful only for wavelengths within the detector
            bandpass (instrument-dependent).
        """
        adjusted_incident_angle_sine = (
            refractive_index - refractive_index_derivative * reference_wavelength
        ) * alpha_in

        groove_density_term = (groove_density * spectral_order) / np.cos(
            out_of_plane_angle
        )
        refractive_correction_term = refractive_index_derivative * alpha_in

        adjusted_groove_density = groove_density_term - refractive_correction_term

        return (adjusted_incident_angle_sine + alpha_out) / (adjusted_groove_density)

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

        # Direction cosines are always dimensionless. Strip any Quantity
        # wrapper so downstream models (e.g. JWST's Logical/mask_slit) that
        # compare against plain floats don't hit astropy 8+ UnitConversionError.
        if isinstance(alpha_out, u.Quantity):
            alpha_out = alpha_out.to_value(u.one)
        if isinstance(beta_out, u.Quantity):
            beta_out = beta_out.to_value(u.one)
        if isinstance(gamma_out, u.Quantity):
            gamma_out = gamma_out.to_value(u.one)

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
