#!/usr/bin/env python3
# https://github.com/luca-pugnetti/Tabu_Search_Python


from gurobipy import Model, GRB, quicksum␍
import random␍
import matplotlib.pyplot as plt␍
import sys␍
from collections import deque␍
␍
sys.stdout = open("output.txt", "w")␍
␍
␍
␍
jobs = [␍
    {"id": 1, "p": 6, "d": 9},␍
    {"id": 2, "p": 4, "d": 12},␍
    {"id": 3, "p": 8, "d": 15},␍
    {"id": 4, "p": 2, "d": 8},␍
    {"id": 5, "p": 10, "d": 20},␍
    {"id": 6, "p": 3, "d": 22},␍
]␍
␍
n = len(jobs)␍
␍
model = Model("Job Sequencing")␍
x = model.addVars(n, n, vtype=GRB.BINARY, name="x")␍
C = model.addVars(n, vtype=GRB.CONTINUOUS, name="C")␍
T = model.addVars(n, vtype=GRB.CONTINUOUS, name="T")␍
model.setObjective(quicksum(T[i] for i in range(n)), GRB.MINIMIZE)␍
␍
for i in range(n):␍
    model.addConstr(quicksum(x[i, j] for j in range(n)) == 1)␍
for j in range(n):␍
    model.addConstr(quicksum(x[i, j] for i in range(n)) == 1)␍
for j in range(n):␍
    model.addConstr(C[j] == quicksum(jobs[i]["p"] * x[i, k] for i in range(n) for k in range(j + 1)))␍
for j in range(n):␍
    model.addConstr(T[j] >= C[j] - quicksum(jobs[i]["d"] * x[i, j] for i in range(n)))␍
    model.addConstr(T[j] >= 0)␍
␍
model.optimize()␍
␍
if model.status == GRB.OPTIMAL:␍
    print("\nOptimal Tardiness:", model.objVal)␍
    solution = []␍
    for j in range(n):␍
        for i in range(n):␍
            if x[i, j].x > 0.5:␍
                solution.append(jobs[i]["id"])␍
                ␍
print(f"Optimal sequence: {solution}")␍
␍
def calculate_tardiness(sequence):␍
    tardiness = 0␍
    completion_time = 0␍
    for job in sequence:␍
        completion_time += job["p"]␍
        tardiness += max(0, completion_time - job["d"])␍
    return tardiness␍
␍
def initial_solution(jobs):␍
    return random.sample(jobs, len(jobs))␍
␍
def get_neighborhood(solution):␍
    neighbors = []␍
    for i in range(len(solution)):␍
        for j in range(i + 1, len(solution)):␍
            neighbor = solution[:]␍
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]␍
            move = [solution[i]["id"], solution[j]["id"], i, j]␍
            neighbors.append((neighbor, move))␍
    return neighbors␍
␍
def choose_move(neighborhood, tabu_list, objective_criteria_counter):␍
    best_tardiness_normal = float("inf")␍
    best_tardiness_tabu = float("inf")␍
    best_tardiness_alltabu = float("inf")␍
    best_move_normal = None␍
    best_move_tabu = None␍
    best_move_alltabu = None␍
    best_neighbor_normal = None␍
    best_neighbor_tabu = None␍
    best_neighbor_alltabu = None␍
    criteria = None␍
    all_tabu = True␍
    ␍
    for neighbor, move in neighborhood:␍
        tardiness = calculate_tardiness(neighbor)␍
        move_upDown = [move[0], move[1], move[2], move[3]]␍
        move_downUp = [move[1], move[0], move[3], move[2]]␍
␍
        if move_upDown not in tabu_list and move_downUp not in tabu_list:␍
            all_tabu = False␍
            break␍
        ␍
        else:␍
            ␍
            if tardiness < best_tardiness_normal:␍
                best_tardiness_alltabu = tardiness␍
                best_move_alltabu = move␍
                best_neighbor_alltabu = neighbor␍
            ␍
    if all_tabu:␍
        criteria = "default"␍
        print(f"  Default aspiration criteria applied {best_move_tabu}")␍
        return best_move_alltabu, best_neighbor_alltabu, best_tardiness_alltabu, criteria␍
␍
    for neighbor, move in neighborhood:␍
        tardiness = calculate_tardiness(neighbor)␍
        move_upDown = [move[0], move[1], move[2], move[3]]␍
        move_downUp = [move[1], move[0], move[3], move[2]]␍
␍
        if move_upDown not in tabu_list and move_downUp not in tabu_list:␍
            if tardiness < best_tardiness_normal:␍
                best_tardiness_normal = tardiness␍
                best_move_normal = move␍
                best_neighbor_normal = neighbor␍
        else:␍
            if tardiness < best_tardiness_tabu:␍
                best_tardiness_tabu = tardiness␍
                best_move_tabu = move␍
                best_neighbor_tabu = neighbor␍
␍
    if best_tardiness_normal < best_tardiness_tabu or objective_criteria_counter > 0 or len(tabu_list) < 5:␍
        return best_move_normal, best_neighbor_normal, best_tardiness_normal, criteria␍
    else:␍
        print(f"  Objective aspiration criteria applied for the move {best_move_tabu}")␍
        criteria = "objective"␍
        return best_move_tabu, best_neighbor_tabu, best_tardiness_tabu, criteria␍
␍
␍
def update_tabu_list(tabu_list, move, tabu_tenure):␍
    ␍
    if move not in tabu_list:␍
        tabu_list.append(move)␍
        ␍
    if len(tabu_list) > tabu_tenure:␍
        tabu_list.pop(0)␍
␍
def tabu_search(jobs, max_iterations=100):␍
    current_solution = initial_solution(jobs)␍
    best_solution = current_solution␍
    tabu_list = []␍
    best_tardiness = calculate_tardiness(best_solution)␍
    tabu_tenure = random.randint(5, 30)␍
    current_tardiness_history = [best_tardiness]␍
    best_tardiness_history = [best_tardiness]␍
    objective_criteria_counter = 0␍
    ␍
    print("\nInitial Solution:")␍
    print(f"  Sequence: {[job['id'] for job in current_solution]} | Tardiness: {best_tardiness}")␍
    print(f"  Tabu tenure: {tabu_tenure}")␍
␍
    for iteration in range(max_iterations):␍
        print(f"\nIteration {iteration + 1}:")␍
        ␍
        neighborhood = get_neighborhood(current_solution)␍
        ␍
        best_move, best_neighbor, best_neighbor_tardiness, criteria = choose_move(neighborhood, tabu_list, objective_criteria_counter)␍
        ␍
        if criteria == 'objective':␍
            objective_criteria_counter = tabu_tenure//2␍
        ␍
        current_solution = best_neighbor␍
        update_tabu_list(tabu_list, best_move, tabu_tenure)␍
        ␍
        if best_neighbor_tardiness < best_tardiness:␍
            best_solution = best_neighbor␍
            best_tardiness = best_neighbor_tardiness␍
        ␍
        current_tardiness_history.append(best_neighbor_tardiness)␍
        best_tardiness_history.append(best_tardiness)␍
␍
        print(f"  Move Selected: {best_move}")␍
        print(f"  Current Sequence: {[job['id'] for job in current_solution]} | Tardiness: {best_neighbor_tardiness}")␍
        print(f"  Best Solution: {[job['id'] for job in best_solution]} | Best Tardiness: {best_tardiness}")␍
        print(f"  Tabu List: {tabu_list}")␍
        ␍
        if objective_criteria_counter >= 1:␍
            objective_criteria_counter -= 1␍
        ␍
    iterations = range(0, tabu_tenure*3+1)␍
    plt.figure(figsize=(40, 20))␍
    plt.plot(iterations ,current_tardiness_history[:tabu_tenure*3+1], label="Current Tardiness", marker="o", linestyle="-")␍
    plt.plot(iterations , best_tardiness_history[:tabu_tenure*3+1], label="Best Tardiness", marker="o", linestyle="--")␍
    plt.xticks(iterations)␍
    plt.xlabel("Iterations", fontsize = 25)␍
    plt.ylabel("Tardiness", fontsize = 25)␍
    plt.title("Tabu Search Progress", fontsize = 30)␍
    plt.legend(fontsize = 25)␍
    plt.grid(True)␍
    plt.savefig("tabu_search_progress.png")␍
    plt.show()␍
␍
    return best_solution, best_tardiness␍
␍
best_solution, best_tardiness = tabu_search(jobs)␍
print("\nFinal Results:")␍
print(f"Best Solution: {[job['id'] for job in best_solution]} | Best Tardiness: {best_tardiness}")
