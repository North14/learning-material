#!/usr/bin/env python3
# Written by: Filip Andersson (filip@artifact.se)
# Date: Aug 2025
# Updated: 
#
# Description:
# This script simulates a Monte Carlo method for Catan dice throws and generates a random Catan board.
# In addition, it simulates resource generation and calculates the best starting position based on the combinations of tile numbers.


import numpy as np
rng = np.random.default_rng(42)
# import matplotlib.pyplot as plt
# 
# from matplotlib.animation import FuncAnimation

from collections import Counter

import logging
logger = logging.getLogger(__name__)


def monte_carlo_dice_throws(n_dice=2, n_sides=6, n_trials=100_000):
    logger.info(f"Running Monte Carlo simulation for {n_trials} trials with {n_dice} dice, each with {n_sides} sides")
    throws = rng.integers(1, n_sides+1, size=(n_trials, n_dice), dtype=np.int16)
    logger.debug(f"First 10 throws: {throws[:10]}")
    return throws.mean(), throws.std(), throws.sum(axis=1)

def generate_random_board():
    # Generate a random board configuration for Catan
    logger.info("Generating random Catan board configuration")

    # Generate and shuffle the tiles
    resources = np.array(['wood'] * 4 + ['brick'] * 3 + ['sheep'] * 4 + 
                        ['wheat'] * 4 + ['ore'] * 3 + ['desert'] * 1)
    rng.shuffle(resources)
    logger.debug(f"Tiles: {resources}")

    # Generate and shuffle the numbers
    numbers = np.array([2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12])
    rng.shuffle(numbers)
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
    return board

def resources_from_dice(dice_array, board):
    # Placeholder for resource calculation based on dice dice
    logger.info("Calculating resources from dice rolls")
    dice_array = np.atleast_1d(dice_array)

    # Special case: robber (7)
    is_robber = dice_array == 7
    robber_res = np.array([[("desert", None)]] * is_robber.sum(), dtype=object)

    # Non-robber dice
    non_robber = dice_array[~is_robber].astype(int)
    resources = np.array([board.get(d, []) for d in non_robber], dtype=object)

    # Recombine in original order
    result = np.empty(len(dice_array), dtype=object)
    result[is_robber] = [[("desert", None)]]
    result[~is_robber] = resources
    logger.debug(f"Resources from dice rolls finished")
    return result
#    results = []
#    for dice in dice_array:
#        if dice == 7:
#            # logger.debug("dice is 7, no resources returned (robber)")
#            results.append([(np.str_('desert'), None)])
#        else:
#            matching_resources = board.get(dice, [])
#            # logger.debug(f"Resources matching dice {dice}: {matching_resources}")
#            results.append(matching_resources)
#    return results

def preprocess_resource_board(board):
    mapping = {}
    for row in board:
        for resource, number in row:
            if number is None:
                mapping.setdefault(np.int16(7), []).append((resource, None))
            else:
                mapping.setdefault(number, []).append((resource, number))
    return mapping

def rc_to_axial(col, row, row_length):
    # This assumes a hexagonal grid with the following properties:
    # - Each row has a different number of columns (3-4-5-4-3)
    # - The hexagonal grid is rotated pointy top
    # - The desired outcome is axial coordinates
    # - Cubed coordinates can be calculated as (q, r, s) where s = -q - r
    # https://www.redblobgames.com/grids/hexagons/
    # Convert a numpy array of double coordinates to cubed coordinates
    return tuple((int(col - (row_length - 3)), int(row)))

def get_corners_from_cubed(coord):
    corner_directions = [
        (-1, 0, 1), (0, -1, 1), (1, -1, 0),
        (+1, 0, -1), (0, +1, -1), (-1, +1, 0)
    ]
    q, r = coord
    s = -q - r
    return [(q + dq, r + dr, s + ds) for dq, dr, ds in corner_directions]


def list_valid_coords(board):
    coordinates = []
    for r, row in enumerate(board):
        for c in range(len(row)):
            double_coord = rc_to_axial(col=c, row=r, row_length=len(row))
            coordinates.append(double_coord)
    return coordinates

def get_corners_from_board(board):
    corners_map = {}
    coordinates_list = list_valid_coords(board)
    for coord in coordinates_list:
        logger.debug(f"Coordinate: {coord}")
        corners = get_corners_from_cubed(coord)
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
    reverse_corners = reverse_mapping_corners(corners_map)
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

def find_best_settlements(shared_3_hexes, board, gained_resources):
    coord_to_tile = {}
    for r_idx, row in enumerate(board):
        for c_idx, tile in enumerate(row):
            logger.debug(f"Processing tile at row {r_idx}, col {c_idx}: {tile}")
            double_coord = rc_to_axial(col=c_idx, row=r_idx, row_length=len(row))
            coord_to_tile[double_coord] = tile
    logger.debug(f"Coordinate to tile mapping: {coord_to_tile}")

    roll_counts = Counter(gained_resources)

    tile_scores = {}
    for coord, tile in coord_to_tile.items():
        resource, number = tile
        tile_scores[coord] = roll_counts.get(number, 0)

    settlement_scores = []
    for corner, hex_coords in shared_3_hexes.items():
        score = sum(tile_scores.get(hc, 0) for hc in hex_coords)
        
        tiles_at_corner = [coord_to_tile.get(hc) for hc in hex_coords]
        tiles_at_corner = [(tile[0], tile[1], np.int64(tile_scores.get(hc, 0))) for hc, tile in zip(hex_coords, tiles_at_corner) if tile]
        
        settlement_scores.append({
            'corner': corner,
            'total_score': score,
            'tiles': tiles_at_corner
        })

    sorted_settlements = sorted(settlement_scores, key=lambda x: x['total_score'], reverse=True)
    logger.info(f"Best settlements found: {sorted_settlements[:3]}") # Log top 3
    
    return sorted_settlements



def initiate_logging():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(funcName)s : %(message)s', filename='monte-carlo-catan.log', filemode='w', level=logging.DEBUG)

if __name__ == '__main__':
    initiate_logging()

    # Generate a random Catan board and run the Monte Carlo simulation
    board = generate_random_board()
    dice_mean, dice_std, dice_results = monte_carlo_dice_throws(n_dice=2, n_sides=6, n_trials=10_000_000)
    # def monte_carlo_dice_throws(n_dice=2, n_sides=6, n_trials=100_000):

    preprocessed_board = preprocess_resource_board(board)
    print(f"Preprocessed board: {preprocessed_board}")
    print(f"Unprocessed board: {board}")
    gained_resources = resources_from_dice(dice_results, preprocessed_board)  # Example for robber
    logger.info(f"Resources gained from first 10 dice rolls: {gained_resources}")

    # Find all possible 2-way and 3-way corners
    corners_map = get_corners_from_board(board)
    analyzed_corners = analyze_corners(corners_map)

    best_settlements = find_best_settlements(analyzed_corners[1], board, dice_results)

    print(f"\n" + "="*50)
    print(f"Generated board: {board}")

    print(f"\n" + "="*50)
    print(f"Preprocessed board: {preprocessed_board}")
    
    print(f"\n" + "="*50)
    print(f"Dice: {dice_results[:10]}")
    print(f"Gained resources from first 10 dice rolls: {gained_resources[:10]}")

    # print(f"\n" + "="*50)
    # print(f"Corners mapping: {corners_map}")
    # print(f"Shared corners with 2 hexes: {analyzed_corners[0]}")
    # print(f"Shared corners with 3 hexes: {analyzed_corners[1]}")

    if best_settlements:
        if len(best_settlements) > 5:
            for i in range(5):
                best_placement = best_settlements[i]
                print("\n" + "="*50)
                print(f"Settlement Rank Based on Simulation: {i+1}")
                print(f"Total simulated resource gain: {best_placement['total_score']:,}")
                print("Tiles at this location:")
                for resource, number, score in best_placement['tiles']:
                    print(f"- {resource.capitalize()} ({number or 'N/A'}) with score {score:,}")
    else:
        print("No valid settlement locations were found.")

    print(f"\n" + "="*50)
    convert_name_scheme = {
        'wood': 'T',
        'brick': 'B',
        'sheep': 'S',
        'wheat': 'W',
        'ore': 'O',
        'desert': 'D'
    }
    for row in board:
        row_string = ""
        for tile in row:
            row_string += f"{convert_name_scheme[tile[0]].capitalize()}{tile[1] or '7'}  "
        print(f"{row_string: ^50s}")

    print(f"\n" + "="*50)
    print(f"Simulation completed with {len(dice_results):,} dice rolls")
    print(f"Average dice sum: {dice_mean:.3f} ± {dice_std:.3f}")
    # print(f"Resource generation rates: {resource_rates}")
