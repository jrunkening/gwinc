import numpy as np
from gwinc.transfer_matrices import Squeezer, FilterCavity


def test_squeezer():
    s = Squeezer(
        squeezing_factor = 10,
        squeezing_angle = 0,
        squared_injection_loss = 0.32
    )
    s.gen_transfer_matrix()(0)


def test_filter_cavity():
    s = FilterCavity(
        squared_input_mirror_transmission = 0.00136,
        squared_round_trip_loss = 120e-6,
        filter_cavity_length = 300,
        carrier_wavelength = 1064,
        detuning = 0.1,
        filter_cavity_length_error = 0,
        sum_of_all_squeezed_filter_cavity_higher_order_mode_coupling_coefficients = 0.06,
        sum_of_all_squeezed_local_oscillator_higher_order_mode_coupling_coefficients = 0.02,
        mode_mismatch_phase_ambiguity = 2*np.pi
    )
    s.gen_transfer_matrix()(0)
