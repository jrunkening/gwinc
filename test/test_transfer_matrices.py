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
    )
    s.gen_transfer_matrix()(0)
