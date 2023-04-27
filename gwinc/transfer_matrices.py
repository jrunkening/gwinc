import numpy as np
from numpy import linalg as la
import scipy.constants as const


class Squeezer:
    """
    The `Squeezer` is represented by the operator :math:`\mathcal{S}(\sigma, \phi)`.

    :math:`\mathcal{S}(\sigma, \phi) = \mathcal{R}(\phi) \mathcal{S}(\sigma, \phi) \mathcal{R}(-\phi)`
    :math:`= \mathcal{R}_\phi \mathcal{S}_\sigma \mathcal{R}_\phi^\dagger`

    Args:
        * `self` (`Squeezer`): A squeezer.
        * `squeezing_factor` (`float`): Squeezing factor in dB.
        * `squeezing_angle,` (`float`): Squeezing angle in rad.
        * `injection_loss` (`float`): The loss that combined all of the \
            input losses together into a single frequency-independent loss.
    """

    def __init__(
        self,
        squeezing_factor,
        squeezing_angle,
        injection_loss
    ) -> None:
        self.squeezing_factor = squeezing_factor / (20*np.log10(np.exp(1)))
        self.squeezing_angle = squeezing_angle
        self.injection_loss = injection_loss
        self.transfer_coefficient_injection_loss = np.sqrt(1-self.injection_loss)
