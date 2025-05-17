import numpy as np

def test_find_rings(paf2):
    # Create a bond matrix for a 5-membered ring: atoms 0-1-2-3-4-0
    rings = paf2.bonding.find_rings()

    #print(rings)
    # Sort and compare for consistency
    #sorted_rings = sorted([sorted(ring) for ring in rings])

    # Its a challenging example
    expected_rings = [[1, 2, 3, 4, 5, 0], [87, 88, 89, 90, 91, 37], [24, 25, 26, 27, 28, 29], [14, 23, 24, 29, 30, 15], [71, 72, 73, 74, 75, 18], [8, 13, 12, 11, 10, 9], [7, 8, 9, 14, 15], [1, 6, 7, 15, 19, 2]]
    
    assert expected_rings == rings
    assert len(rings) == 8  # Should be only one ring

def test_ring_center(paf2):
    ring0 = paf2.bonding.find_rings()[0]
    center = paf2.bonding.get_ring_center(ring0)
    assert np.allclose(center, np.asarray([1.207545, 4.19927442, -1.40707352]))