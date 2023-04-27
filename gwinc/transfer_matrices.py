import numpy as np
from numpy import linalg as la
import scipy.constants as const


class Squeezer:
    r"""
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

        self.squeezing = np.array([
            [np.exp(self.squeezing_factor), 0],
            [0, np.exp(-self.squeezing_factor)]
        ])
        self.rotation = np.array([
            [np.cos(self.squeezing_angle), -np.sin(self.squeezing_angle)],
            [np.sin(self.squeezing_angle), np.cos(self.squeezing_angle)]
        ])
        self.rotation_dagger = np.array([
            [np.cos(self.squeezing_angle), np.sin(self.squeezing_angle)],
            [-np.sin(self.squeezing_angle), np.cos(self.squeezing_angle)]
        ])
        self.transfer_coefficient_injection_loss = np.sqrt(1-self.injection_loss)

    def transfer(self):
        return self.transfer_coefficient_injection_loss * \
            self.rotation.dot(self.squeezing).dot(self.rotation_dagger)
