#!/usr/bin/env python3

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
    resources = {
        "wood": 4,
        "brick": 3,
        "sheep": 4,
        "wheat": 4,
        "ore": 3,
        "desert": 1
    }
    
    # Expand into list
    tiles = np.array(
        sum(([resource] * count for resource, count in resources.items()), [])
    )
    tiles = np.random.permutation(tiles)
    logger.debug(f"Tiles: {tiles}")

    numbers = np.array([2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12])
    numbers = np.random.permutation(numbers)
    logger.debug(f"Shuffled number: {numbers}")
    
    assigned_tiles = []
    num_idx = 0
    for t in tiles:
        if t == "desert":
            assigned_tiles.append((t, None))  # no number on desert
        else:
            assigned_tiles.append((t, numbers[num_idx]))
            num_idx += 1
    
    # Arrange in Catan hex shape (3-4-5-4-3)
    rows = [3, 4, 5, 4, 3]
    board = []
    idx = 0
    for count in rows:
        row = []
        for _ in range(count):
            row.append(assigned_tiles[idx])
            idx += 1
        board.append(row)

    logger.debug(f"Generated board: {board}")
    return board

def resources_from_dice(dice, board):
    # Placeholder for resource calculation based on dice dice
    if dice == np.int64(7):
        # logger.debug("dice is 7, no resources returned (robber)")
        return [(np.str_('desert'), None)]
    matching_resources = []
    for row in board:
        for tile in row:
            if tile[1] == dice:
                matching_resources.append(tile)
    logger.debug(f"Resources matching dice {dice}: {matching_resources}")
    return matching_resources

def preprocess_resource_board(board):
    mapping = {}
    for row in board:
        for resource, number in row:
            if number is None:
                mapping.setdefault(np.int64(7), []).append((resource, None))
            else:
                mapping.setdefault(number, []).append((resource, number))
    return mapping

def list_to_hex(col, row, row_length, cubed=False):
    # Convert row, col coordinates to axial hex coordinates
    # https://www.redblobgames.com/grids/hexagons/
    q = col - (row_length - 3)
    if cubed:
        return tuple((int(q), int(row), int(-row-q)))
    return tuple((int(q), int(row)))

def axial_to_double(coord):
    # https://www.redblobgames.com/grids/hexagons/
    q, r = coord
    return 2 * q + r, r

def get_corners_from_coord(coord):
    # Get the 6 corners of a hexagon given its axial coordinates (q, r)
    corner_directions = [
    (+2, 0), (+1, -1), (-1, -1),
    (-2, 0), (-1, +1), (+1, +1)
    ]
    q, r = coord
    return [(q + dq, r + dr) for dq, dr in corner_directions]

def get_corners_from_board(board):
    corner_map = {}
    coordinates = []
    logger.debug(enumerate(board))
    for r, row in enumerate(board):
        logger.debug(f"Counting row {r} with {len(row)} tiles")
        for c, tile in enumerate(row):
            hex_coord = axial_to_double(list_to_hex(col=c, row=r, row_length=len(row), cubed=False))
            coordinates.append(hex_coord)
            logger.debug(f"Tile {tile} at row {r}, col {c} has axial coordinates ({hex_coord})")
            corners = get_corners_from_coord(hex_coord)
            # logger.debug(f"Corners for tile at ({hex_coord}): {corners}")
            for corner in corners:
                if corner in coordinates:
                    # logger.debug(f"Corner {corner} already exists in coordinates")
                    corner_map.setdefault(corner, []).append((tile, hex_coord))
    return corner_map



def initiate_logging():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(funcName)s : %(message)s', filename='monte-carlo-catan.log', filemode='w', level=logging.DEBUG)

if __name__ == '__main__':
    initiate_logging()
    board = generate_random_board()
    dice_mean, dice_std, dice = monte_carlo_dice_throws(n_trials=5)
    gained_resources = [resources_from_dice([s], board) for s in dice]
    logger.info(gained_resources)
    # preprocessed_board = preprocess_resource_board(board)
    # logger.info(preprocessed_board)
    logger.info(board)
    full_corners = get_corners_from_board(board)

    three_hex_combinations = []
    for corner, touching in full_corners.items():
        if len(touching) == 3:
            combo = tuple(sorted(touching, key=lambda x: x[1]))  # sort by (row,col)
            three_hex_combinations.append(combo)

    # Deduplicate
    three_hex_combinations = list(set(three_hex_combinations))

    for i, combo in enumerate(three_hex_combinations):
        print(f"Combination {i}: {combo}")
    print("Number of 3-hex corners:", len(three_hex_combinations))
    for combo in three_hex_combinations[:5]:
        print(combo)

