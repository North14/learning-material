## Monte Carlo Catan
- [Monte Carlo Catan](#monte-carlo/monte-carlo-catan.py)

Simple demonstration of monte carlo simulations being used to generate the best placement in the board game CATAN.
The application generates a random Catan starting board,
simulates N 2d6 dice throws,
and calculates the expected resources.
In the end the application calculates the best starting positions,
based on the total amount of gathered resources.

The implementation of the hexagon board uses the axial/cubed coordinate system (q, r(, s)).
Corners are calculated using a 3 times scaled cubed coordinate systems with corners being the permutation of +-(2, -1, -1).
E.g. (q, r, s) = (2, 1, -3)
>>> get_corners_from_axial((2, 1))
[(8, 2, -10), (7, 4, -11), (5, 5, -10), (4, 4, -8), (5, 2, -7), (7, 1, -8)]

The following system allows a coordinate system that calculates the location of every corner for each coordinate,
which may share a corner coordinate with a neighbour, while avoiding collisions.

