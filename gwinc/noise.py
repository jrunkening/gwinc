import numpy as np
from numpy import linalg as la


def local_oscillator_field(amplitude: float, measurement_phase: float):
    return amplitude * np.array([np.sin(measurement_phase), np.cos(measurement_phase)])


def build_noise(transfer_matrix, measurement_amplitude: float, measurement_phase: float):
    return lambda signal_sideband_frequency: np.square(la.norm(
        local_oscillator_field(measurement_amplitude, measurement_phase).conj().T.dot(
            np.squeeze(transfer_matrix(signal_sideband_frequency)).dot(
                np.identity(2)
            )
        )
    ))
