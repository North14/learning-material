#!/usr/bin/env python3
# Written by: Filip Andersson (filip@artifact.se)
# Date: Aug 2025
# Updated: 
#
# Description:
# This script simulates a Monte Carlo method for Catan dice throws and generates a random Catan board.
# In addition, it simulates resource generation and calculates the best starting position based on the combinations of tile numbers.


import numpy as np
rng = np.random.default_rng()
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
    dice_array = np.atleast_1d(dice_array).astype(int)  # normalize to int

    # Build lookup array from board keys (works for Catan dice: 2–12 + robber 7)
    max_dice = max(board.keys())
    lookup = np.empty(max_dice + 1, dtype=object)
    for k, v in board.items():
        lookup[int(k)] = v

    return lookup[dice_array]

def preprocess_resource_board(board):
    mapping = {}
    for row in board:
        for resource, number in row:
            if number is None:
                mapping.setdefault(7, []).append((resource, None))
            else:
                mapping.setdefault(int(number), []).append((resource, number))
    return mapping

def rc_to_axial(col, row, row_length):
    # This assumes a hexagonal grid with the following properties:
    # - Each row has a different number of columns (3-4-5-4-3)
    # - The hexagonal grid is rotated pointy top
    # - The desired outcome is axial coordinates
    # - Cubed coordinates can be calculated as (q, r, s) where s = -q - r
    # https://www.redblobgames.com/grids/hexagons/
    # Convert a numpy array of double coordinates to cubed coordinates
    return tuple((int(col - (row_length - 3)), int(-row)))

def axial_to_cubed(q, r):
    s = -q - r
    return (q, r, s)

def get_corners_from_axial(coord):
    # Given a coordinate in axial format (q, r), return the corners in cubed format
    # https://www.redblobgames.com/grids/parts/

    # corner_directions = [
    #     (-1, 0, 1), (0, -1, 1), (1, -1, 0),
    #     (+1, 0, -1), (0, +1, -1), (-1, +1, 0)
    # ]
    corner_directions = [
        ( 2,-1,-1), ( 1, 1,-2), (-1, 2,-1),
        (-2, 1, 1), (-1,-1, 2), ( 1,-2, 1)
    ]
    
    qc, rc, sc = axial_to_cubed(*coord)
    base = (3*qc, 3*rc, 3*sc)
    return [(base[0]+dx, base[1]+dy, base[2]+dz) for (dx,dy,dz) in corner_directions]

def corner_to_hexes(corner_coord):
    """
    Given a corner coordinate, return the hexes that share this corner.
    Corner coordinates are in scaled cube coordinates (3x scale).
    """
    cx, cy, cz = corner_coord
    
    corner_directions = [
        ( 2,-1,-1), ( 1, 1,-2), (-1, 2,-1),
        (-2, 1, 1), (-1,-1, 2), ( 1,-2, 1)
    ]
    
    hexes = []
    for dx, dy, dz in corner_directions:
        hx = (cx - dx) // 3
        hy = (cy - dy) // 3
        hz = (cz - dz) // 3
        
        if hx + hy + hz == 0:
            hexes.append((hx, hy))
    return hexes

def get_corner_neighbors(coord):
    """
    Find all hexes that share at least one corner with the given hex.
    Returns a set of (q, r) tuples in axial coordinates.
    """
    corners = get_corners_from_axial(coord)
    neighbors = set()
    
    for corner in corners:
        sharing_hexes = corner_to_hexes(corner)
        for hex_coord in sharing_hexes:
            if hex_coord != coord:
                neighbors.add(hex_coord)
    
    return neighbors

def list_valid_coords(board):
    coordinates = []
    for r, row in enumerate(board):
        for c in range(len(row)):
            axial_coord = rc_to_axial(col=c, row=r, row_length=len(row))
            coordinates.append(axial_coord)
    return coordinates

def get_corners_from_board(board):
    corners_map = {}
    corner_neighbors_map = {}
    coordinates_list = list_valid_coords(board)
    for coord in coordinates_list:
        logger.debug(f"Coordinate: {coord}")
        corners = get_corners_from_axial(coord)
        corners_map[coord] = corners
        corner_neighbors_map[coord] = get_corner_neighbors(coord)
    logger.debug(f"Corners map: {corners_map}, for {len(corners_map)} coordinates")
    logger.debug(f"Corner neighbors map: {corner_neighbors_map}, for {len(corner_neighbors_map)} coordinates")
    return corners_map, corner_neighbors_map

def reverse_mapping_corners(corners_map):
    reverse_map = {}
    for coord, corners in corners_map.items():
        for corner in corners:
            if corner not in reverse_map:
                reverse_map[corner] = []
            reverse_map[corner].append(coord)
    logger.debug(f"Reverse corners mapping: {reverse_map}")
    return reverse_map

def analyze_corners(corners_data):
    corners_map, corner_neighbors_map = corners_data
    reverse_corners = reverse_mapping_corners(corners_map)
    shared_2_hexes = {}
    shared_3_hexes = {}
    
    for corner, hex_list in reverse_corners.items():
        if len(hex_list) == 2:
            shared_2_hexes[corner] = hex_list
        elif len(hex_list) == 3:
            shared_3_hexes[corner] = hex_list
    results = [shared_2_hexes, shared_3_hexes, corner_neighbors_map]
    logger.debug(f"Shared corners with 2 hexes: {shared_2_hexes}")
    if len(shared_2_hexes) != 14:
        raise ValueError(f"Expected 12 shared corners with 2 hexes, found {len(shared_2_hexes)}")
    logger.debug(f"Shared corners with 3 hexes: {shared_3_hexes}")
    if len(shared_3_hexes) != 22:
        raise ValueError(f"Expected 24 shared corners with 3 hexes, found {len(shared_3_hexes)}")
    return results

def find_best_settlements(shared_3_hexes, board, gained_resources):
    logger.info("Finding the best settlement placements")
    coord_to_tile = {}
    for r_idx, row in enumerate(board):
        logger.debug(f"Processing row {r_idx}: {row}")
        for c_idx, tile in enumerate(row):
            double_coord = rc_to_axial(col=c_idx, row=r_idx, row_length=len(row))
            coord_to_tile[double_coord] = tile

    logger.debug(f"Coordinate to tile mapping: {coord_to_tile}")

    max_val = np.max(gained_resources)
    roll_counts = np.bincount(gained_resources, minlength=max_val+1)

    tile_scores = {}
    for coord, tile in coord_to_tile.items():
        resource, number = tile
        if number is None:
            tile_scores[coord] = 0
        else:
            tile_scores[coord] = roll_counts[number] if number < len(roll_counts) else 0

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
    dice_mean, dice_std, dice_results = monte_carlo_dice_throws(n_dice=2, n_sides=6, n_trials=1_000_000)
    # def monte_carlo_dice_throws(n_dice=2, n_sides=6, n_trials=100_000):

    preprocessed_board = preprocess_resource_board(board)
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
