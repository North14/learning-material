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
    logger.debug(throws)
    return np.mean(throws), np.std(throws), throws

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
    tiles = []
    for resource, count in resources.items():
        tiles.extend([resource] * count)
    np.random.shuffle(tiles)
    logger.debug(f"Tiles: {tiles}")

    numbers = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]
    np.random.shuffle(numbers)
    logger.debug(f"Shuffled number: {numbers}")
    
    assigned_tiles = []
    num_idx = 0
    for t in tiles:
        logger.debug(f"Assigning tile {t} with number index {num_idx} and number {numbers[num_idx]}")
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

def initiate_logging():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(funcName)s : %(message)s', filename='monte-carlo-catan.log', filemode='w', level=logging.DEBUG)
    # logging.basicConfig(filename='monte-carlo-catan.log', filemode='w', level=logging.DEBUG)

if __name__ == '__main__':
    initiate_logging()
    generate_random_board()
    monte_carlo_dice_throws()
