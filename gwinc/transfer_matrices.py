import abc
from typing import List
import numpy as np
from numpy import linalg as la
import scipy.constants as const


class AbstractComponent(abc.ABC):
    @abc.abstractmethod
    def gen_transfer_matrix(self) -> np.array:
        return NotImplemented


def build_transfer_matrix(components: List[AbstractComponent]):
    if len(components) < 2:
        return components[0].gen_transfer_matrix()

    return lambda signal_sideband_frequency: la.multi_dot([
        component.gen_transfer_matrix()(signal_sideband_frequency)
        for component in components[::-1]
    ])


class Squeezer(AbstractComponent):
    """
    The `Squeezer` is represented by the operator :math:`\\mathcal{S}(\\sigma, \\phi)`.

    :math:`\\mathcal{S}(\\sigma, \\phi) = \\mathcal{R}(\\phi) \\mathcal{S}(\\sigma, \\phi) \\mathcal{R}(-\\phi)`
    :math:`= \\mathcal{R}_\\phi \\mathcal{S}_\\sigma \\mathcal{R}_\\phi^\\dagger`

    Args:
        * `self` (`Squeezer`): A squeezer.
        * `squeezing_factor` (`float`): Squeezing factor in dB.
        * `squeezing_angle` (`float`): Squeezing angle in rad.
        * `squared_injection_loss` (`float`): The loss that combined all of the \
            input losses together into a single frequency-independent loss.
    """

    def __init__(
        self,
        squeezing_factor: float, # \\sigma
        squeezing_angle: float, # \\phi
        squared_injection_loss: float # \\Lambda_{inj}^2
    ) -> None:
        super().__init__()
        self.squeezing_factor = squeezing_factor / (20*np.log10(np.exp(1)))
        self.squeezing_angle = squeezing_angle
        self.squared_injection_loss = squared_injection_loss

        self.squeezing = np.array([
            [np.exp(self.squeezing_factor), 0],
            [0, np.exp(-self.squeezing_factor)]
        ]) # \\mathcal{S}
        self.rotation = np.array([
            [np.cos(self.squeezing_angle), -np.sin(self.squeezing_angle)],
            [np.sin(self.squeezing_angle), np.cos(self.squeezing_angle)]
        ]) # \\mathcal{R}
        self.rotation_dagger = np.array([
            [np.cos(self.squeezing_angle), np.sin(self.squeezing_angle)],
            [-np.sin(self.squeezing_angle), np.cos(self.squeezing_angle)]
        ]) # \\mathcal{R}^\\dagger
        self.injection_loss_transfer_coefficient = np.sqrt(1-self.squared_injection_loss) # \\tau_{inj}

    def gen_transfer_matrix(self) -> np.array:
        """
        Generate the transfer matrix :math:`\\mathcal{T}_{inj}` \
            of the squeezer (injection loss considered).

        Args:
            * `self` (`Squeezer`): A squeezer.

        Return:
            * (`np.array`): The transfer matrix.
        """

        return lambda _: self.injection_loss_transfer_coefficient * \
            self.rotation.dot(self.squeezing).dot(self.rotation_dagger)


class FilterCavity(AbstractComponent):
    """
    The filter cavity.

    Args:
        * `self` (`FilterCavity`): A filter cavity.
        * `squared_input_mirror_transmission` (`float`):
        * `squared_round_trip_loss` (`float`):
        * `filter_cavity_length` (`float`):
        * `carrier_wavelength` (`float`):
        * `detuning` (`float`):
        * `filter_cavity_length_error` (`float`):
    """

    def __init__(
        self,
        squared_input_mirror_transmission: float, # t_{in}^2
        squared_round_trip_loss: float, # \\Lambda_{rt}^2
        filter_cavity_length: float, # L_{fc}
        carrier_wavelength: float, # \\lambda
        detuning: float,
        filter_cavity_length_error: float,
        sum_of_all_squeezed_filter_cavity_higher_order_mode_coupling_coefficients: float, # \\Sigma |a_n|^2
        sum_of_all_squeezed_local_oscillator_higher_order_mode_coupling_coefficients: float, # \\Sigma |c_n|^2
        mode_mismatch_phase_ambiguity: float, # \\phi_{mm}
    ) -> None:
        super().__init__()
        self.squared_input_mirror_transmission = squared_input_mirror_transmission
        self.squared_round_trip_loss = squared_round_trip_loss
        self.filter_cavity_length = filter_cavity_length
        self.carrier_wavelength = carrier_wavelength
        self.filter_cavity_length_error = filter_cavity_length_error

        self.carrier_frequency = 2*const.pi*const.c / self.carrier_wavelength # \\omega_0
        self.filter_cavity_detuning = detuning * 2*const.pi + \
            self.filter_cavity_length_error * self.carrier_frequency/self.filter_cavity_length # \\Delta \\omega_{fc}
        self.input_mirror_reflectivity = np.sqrt(1 - self.squared_input_mirror_transmission) # r_{in}
        self.fsr_frequency = const.c / (2*self.filter_cavity_length) # f_{FSR}
        self.bandwith = self.fsr_frequency * (self.squared_input_mirror_transmission + self.squared_round_trip_loss)/2 # \\gamma_{fc}
        self.round_trip_reflectivity = np.exp(-self.bandwith / self.fsr_frequency)
        self.conversion_matrix = 1/np.sqrt(2) * np.array([
            [1, 1],
            [-1j, 1j]
        ]) # \\mathcal{A}_2

        self.squeezed_filter_cavity_mode_coupling = np.sqrt(
            1 - sum_of_all_squeezed_filter_cavity_higher_order_mode_coupling_coefficients
        ) # a_0
        self.squeezed_local_oscillator_mode_coupling = np.sqrt(
            1 - sum_of_all_squeezed_local_oscillator_higher_order_mode_coupling_coefficients
        ) # c_0
        self.local_oscillator_filter_cavity_mode_coupling = \
            self.squeezed_filter_cavity_mode_coupling*self.squeezed_filter_cavity_mode_coupling + np.sqrt(
                (1 - np.square(self.squeezed_filter_cavity_mode_coupling)) * \
                (1 - np.square(self.squeezed_filter_cavity_mode_coupling))
            ) * np.exp(1j * mode_mismatch_phase_ambiguity) # b_0
        self.first_order_mode_mismatch =  self.squeezed_filter_cavity_mode_coupling * \
            np.conj(self.local_oscillator_filter_cavity_mode_coupling) # t_{00}
        self.higher_order_mode_mismatch = self.squeezed_local_oscillator_mode_coupling - \
             self.first_order_mode_mismatch # t_{mm}

    def get_round_trip_phase(self, signal_sideband_frequency: float) -> float:
        """
        Get round trip phase :math:`\\Phi(\\Omega)`.

        Args:
            * `self` (`FilterCavity`): A filter cavity.
            * `signal_sideband_frequency` (`float`): The signal sideband frequency.

        Return:
            * (`float`): round trip phase
        """

        return (signal_sideband_frequency - self.filter_cavity_detuning) / self.fsr_frequency

    def get_reflectivity(self, signal_sideband_frequency: float) -> float:
        round_trip_factor = self.round_trip_reflectivity * \
            np.exp(-1j*self.get_round_trip_phase(signal_sideband_frequency))

        return self.input_mirror_reflectivity - \
            (self.squared_input_mirror_transmission/self.input_mirror_reflectivity) * \
            round_trip_factor/(1 - round_trip_factor)

    def gen_filter_cavity_transfer_matrix(self) -> np.array:
        """
        Generate the transfer matrix :math:`\\mathcal{T}_{fc}` \
            of the filter cavity.

        Args:
            * `self` (`FilterCavity`): A filter cavity.

        Return:
            * (`np.array`): The transfer matrix.
        """

        return lambda signal_sideband_frequency: self.conversion_matrix.dot(np.array([
            [self.get_reflectivity(signal_sideband_frequency), 0],
            [0, np.conj(self.get_reflectivity(-signal_sideband_frequency))]
        ])).dot(la.inv(self.conversion_matrix))

    def gen_mode_matching_transfer_matrix(self) -> np.array:
        """
        Generate the transfer matrix :math:`\\mathcal{T}_{mm}` \
            of the mode matching.

        Args:
            * `self` (`FilterCavity`): A filter cavity.

        Return:
            * (`np.array`): The transfer matrix.
        """

        return lambda _: self.conversion_matrix.dot(np.array([
            [self.higher_order_mode_mismatch, 0],
            [0, np.conj(self.higher_order_mode_mismatch)]
        ])).dot(la.inv(self.conversion_matrix))

    def gen_transfer_matrix(self) -> np.array:
        """
        Generate the transfer matrix :math:`t_{00} \\mathcal{T}_{fc} + \\mathcal{T}_{mm}` \
            of the filter cavity.

        Args:
            * `self` (`FilterCavity`): A filter cavity.

        Return:
            * (`np.array`): The transfer matrix.
        """

        return lambda signal_sideband_frequency: \
            self.first_order_mode_mismatch * self.gen_filter_cavity_transfer_matrix()(signal_sideband_frequency) + \
            self.gen_mode_matching_transfer_matrix()(signal_sideband_frequency)


class Readout(AbstractComponent):
    def __init__(
        self,
        squared_readout_loss: float # \\Lambda_{ro}^2
    ) -> None:
        super().__init__()
        self.squared_readout_loss = squared_readout_loss
        self.readout_loss_transfer_coefficient = np.sqrt(1 - self.squared_readout_loss) # \\tuo_{ro}
        self.conversion_matrix = 1/np.sqrt(2) * np.array([
            [1, 1],
            [-1j, 1j]
        ]) # \\mathcal{A}_2

    def gen_transfer_matrix(self) -> np.array:
        return lambda _: self.readout_loss_transfer_coefficient * np.identity(
            self.conversion_matrix.shape[0]
        )
