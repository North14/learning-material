#!/usr/bin/env python3
# Written by: Filip Andersson (filip@artifact.se)
# Date: Aug 2025
# Updated: 
#
# Description:
# This script simulates a Monte Carlo method for Catan dice throws and generates a random Catan board.
# In addition, it simulates resource generation and calculates the best starting position based on the combinations of tile numbers.


import numpy as np
# import matplotlib.pyplot as plt
# 
# from matplotlib.animation import FuncAnimation

import logging
logger = logging.getLogger(__name__)


def monte_carlo_dice_throws(n_dice=2, n_sides=6, n_trials=100_000):
    logger.info(f"Running Monte Carlo simulation for {n_trials} trials with {n_dice} dice, each with {n_sides} sides")
    throws = np.random.randint(1, high=n_sides+1, size=(n_trials, n_dice))
    # logger.debug(throws)
    return np.mean(throws), np.std(throws), throws.sum(axis=1)

def generate_random_board():
    # Generate a random board configuration for Catan
    logger.info("Generating random Catan board configuration")

    # Generate and shuffle the tiles
    resources = np.array(['wood'] * 4 + ['brick'] * 3 + ['sheep'] * 4 + 
                        ['wheat'] * 4 + ['ore'] * 3 + ['desert'] * 1)
    resources = np.random.permutation(resources)
    logger.debug(f"Tiles: {resources}")

    # Generate and shuffle the numbers
    numbers = np.array([2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12])
    numbers = np.random.permutation(numbers)
    logger.debug(f"Shuffled number: {numbers}")
    
    tiles = []
    num_idx = 0
    for r in resources:
        if r == "desert":
            tiles.append((r, None))  # no number on desert
        else:
            tiles.append((r, numbers[num_idx]))
            num_idx += 1
    
    # Arrange in Catan hex shape (3-4-5-4-3)
    rows = [3, 4, 5, 4, 3]
    board = []
    idx = 0
    for count in rows:
        row = []
        for _ in range(count):
            row.append(tiles[idx])
            idx += 1
        board.append(row)

    logger.debug(f"Generated board: {board}")
    return board, np.array(tiles)

def resources_from_dice(dice_array, board):
    # Placeholder for resource calculation based on dice dice
    if np.isscalar(dice_array):
        dice_array = np.array([dice_array])

    results = []
    for dice in dice_array:
        if dice == np.int64(7):
            logger.debug("dice is 7, no resources returned (robber)")
            results.append([(np.str_('desert'), None)])
        else:
            matching_resources = board.get(dice, [])
            logger.debug(f"Resources matching dice {dice}: {matching_resources}")
            results.append(matching_resources)
    return results

def preprocess_resource_board(board):
    mapping = {}
    for row in board:
        for resource, number in row:
            if number is None:
                mapping.setdefault(np.int64(7), []).append((resource, None))
            else:
                mapping.setdefault(number, []).append((resource, number))
    return mapping

def list_to_double_coord(col, row, row_length):
    # Convert row, col coordinates to double coordinates
    # This assumes a hexagonal grid with the following properties:
    # - Each row has a different number of columns (3-4-5-4-3)
    # - The hexagonal grid is rotated pointy top
    # - The desired outcome is doublewidth
    # https://www.redblobgames.com/grids/hexagons/
    q = (col * 2) - (row_length - 3)
    if (q + row) % 2 != 0: # constraint (col + row) mod 2 == 0
        raise ValueError("Invalid coordinates: q + row must be even for hexagonal grid")
    return tuple((int(q), int(row)))

def doublewidth_to_axial(col, row):
    return tuple((int((col - row) / 2), int(row)))

def axial_to_doublewidth(q, r):
    return tuple((int(2 * q + r), int(r)))

def qr_to_s(q, r):
    return int(-q-r)


def get_neigbours_from_coord(coord):
    # Get the 6 neighbours of a hexagon given its double coordinates (Col, Row)
    neighbour_directions = [
    (+2, 0), (+1, -1), (-1, -1),
    (-2, 0), (-1, +1), (+1, +1)
    ]
    q, r = coord
    return [(q + dq, r + dr) for dq, dr in neighbour_directions]

def get_corners_from_doublewidth(coord):
    corner_directions = [
            (-1, 0, 'N'), (0, +1, 'S'), (+1, 0, 'N'),
            (0, -1, 'S'), (0, -1, 'N'), (+1, 0, 'S')
    ]
    q, r = coord
    return [(q + dq, r + dr, direction) for dq, dr, direction in corner_directions]

def validate_corners(corners):
    for q, r, _ in corners:
        if (q + r) % 2 == 0:
            raise ValueError("Invalid corner coordinates: col + row must be even for hexagonal grid")
    logger.debug(f"Validated corners: {corners}")


def list_valid_coords(board):
    coordinates = []
    for r, row in enumerate(board):
        for c in range(len(row)):
            double_coord = list_to_double_coord(col=c, row=r, row_length=len(row))
            coordinates.append(double_coord)
    return coordinates

def get_neighbours_from_board(board, coordinates):
    neighbour_map = {}
    print(f"Running neighbour finder for coordinates: {coordinates}")
    for r, row in enumerate(board):
        for c, tile in enumerate(row):
            double_coord = list_to_double_coord(col=c, row=r, row_length=len(row))
            neighbours = get_neigbours_from_coord(double_coord)
            for neighbour in neighbours:
                if neighbour in coordinates:
                    neighbour_map.setdefault(neighbour, []).append((tile, double_coord))
    return neighbour_map

def get_corners_from_board(board):
    corners_map = {}
    coordinates_list = list_valid_coords(board)
    for i, coord in enumerate(coordinates_list):
        logger.debug(f"Coordinate: {coord}, Axial: {doublewidth_to_axial(*coord)}")
        corners = get_corners_from_doublewidth(coord)
        # validate_corners(corners)
        corners_map[coord] = corners
    logger.debug(f"Corners map: {corners_map}, for {len(corners_map)} coordinates")
    return corners_map

def reverse_mapping_corners(corners_map):
    reverse_map = {}
    for coord, corners in corners_map.items():
        for corner in corners:
            if corner not in reverse_map:
                reverse_map[corner] = []
            reverse_map[corner].append(coord)
    logger.debug(f"Reverse corners mapping: {reverse_map}")
    return reverse_map

def analyze_corners(corners_map):
    reverse_corners = reverse_mapping_corners(corners)
    shared_2_hexes = {}
    shared_3_hexes = {}
    
    for corner, hex_list in reverse_corners.items():
        if len(hex_list) == 2:
            shared_2_hexes[corner] = hex_list
        elif len(hex_list) == 3:
            shared_3_hexes[corner] = hex_list
    results = [shared_2_hexes, shared_3_hexes]
    logger.debug(f"Shared corners with 2 hexes: {shared_2_hexes}")
    logger.debug(f"Shared corners with 3 hexes: {shared_3_hexes}")
    return results

def initiate_logging():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(funcName)s : %(message)s', filename='monte-carlo-catan.log', filemode='w', level=logging.DEBUG)

if __name__ == '__main__':
    initiate_logging()
    board, board_array = generate_random_board()
    logger.info("Generated random Catan board")
    logger.debug(f"Board: {board}")
    logger.debug(f"Board array: {board_array}")
    dice_mean, dice_std, dice_results = monte_carlo_dice_throws()
    preprocessed_board = preprocess_resource_board(board)
    gained_resources = resources_from_dice(dice_results[:10], preprocessed_board)  # Example for robber
    logger.info(f"Resources gained from first 10 dice rolls: {gained_resources}")
    corners = get_corners_from_board(board)
    analyzed_corners = analyze_corners(corners)
    visualize_corner_distribution(analyzed_corners[0])  # Frequency of corners shared by 2 hexes
    visualize_corner_distribution(analyzed_corners[1])  # Frequency of corners shared by 3 hexes
    # full_neighbours = get_neighbours_from_board(board, coordinates_list)

    # logger.info("Full neighbours mapping: %s", full_neighbours)

    # three_hex_combinations = []
    # for neighbour, touching in full_neighbours.items():
    #     if len(touching) >= 3:
    #         combo = tuple(sorted(touching, key=lambda x: x[1]))  # sort by (row,col)
    #         three_hex_combinations.append(combo)
    # 
    # # Deduplicate
    # three_hex_combinations = list(set(three_hex_combinations))

    # print("Number of tiles with 3 neighbours:", len(three_hex_combinations))
    # for i, combo in enumerate(three_hex_combinations):
    #     print(f"Combination {i}: {combo}")
    print(f"Simulation completed with {len(dice_results)} dice rolls")
    print(f"Average dice sum: {dice_mean:.3f} ± {dice_std:.3f}")
    # print(f"Resource generation rates: {resource_rates}")
