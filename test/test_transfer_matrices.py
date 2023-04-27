from gwinc.transfer_matrices import Squeezer

def test_squeezer():
    s = Squeezer(10, 0, 0.32)
    s.transfer()
