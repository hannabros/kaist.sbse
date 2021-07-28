import os, sys
import argparse
import random
import math
import time
import pickle

parser = argparse.ArgumentParser(description='Travelling Sales Person Problem Arguments')

parser.add_argument('-p', '--population', default=None, type=int, help='Population')
parser.add_argument('-f', '--fit-limit', default=100, type=int, help='Fitness Evaluation Limit') 
parser.add_argument('-l', '--logic', default='lc', type=str, help='Name of Logic')
parser.add_argument('--tsp-file', default=str, help='TSP file path')

args = parser.parse_args()

def get_coordinates(file):
    coor = {}
    with open(file, 'r') as f:
        lines = f.read().splitlines()
        header_index = lines.index('NODE_COORD_SECTION')
        eof_index = lines.index('EOF')
        for line in lines[header_index+1:eof_index]:
            tmp = [float(el) for el in line.strip().split(' ') if el != '']
            coor[int(tmp[0])-1] = tmp[1:]
    return coor

# return two indexes of coordinates and distance between those coordinates ex) {(0, 1): 10, (0, 2): 5, ...}
def get_distances(start_coor, end_coor):
    distance = math.sqrt(sum([(p - q)**2 for (p, q) in zip(start_coor, end_coor)]))
    return distance

# solution : sequence of index of coordinates
def fitness(solution, coor):
    total_dist = 0
    for idx, coor_idx in enumerate(solution[:-1]):
        start, end = coor_idx, solution[idx+1]
        if end < start:
            tmp = start
            start = end
            end = tmp
        total_dist += get_distances(coor[start], coor[end])
    return total_dist

def init_random(coor):    
    return random.sample(list(range(len(coor))), len(coor))

def get_neighbors(solution):
    neighbors = []
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            neighbour = solution[:]
            neighbour[i] = solution[j]
            neighbour[j] = solution[i]
            neighbors.append(neighbour)
    return neighbors
    
# should add limit instance
# should change to local search (global search currently)
def local_search(coor, gen):
    init_sol = init_random(coor)
    neighbors = get_neighbors(init_sol)
    init_length = fitness(init_sol, coor)

    step = 0
    for neighbor in neighbors:
        if step > gen:
            break
        new_length = fitness(neighbor, coor)
        if new_length < init_length:
            solution = neighbor
            init_length = new_length
        step += 1
    
    return solution, new_length

def crossover(sol1, sol2, coor):
    p = random.randint(1, len(sol1) - 1)
    o1 = sol1[0][:p] + sol2[0][p:]
    o2 = sol2[0][:p] + sol1[0][p:]
    return (o1, fitness(o1, coor)), (o2, fitness(o2, coor))

def mutate(solution, rate, coor):
    for i in range(len(solution[0])):
        if random.random() < rate:
            randint = random.randint(0, len(solution[0])-1)
            if randint not in solution[0]:
                solution[0][i] = randint
    return (solution[0], fitness(solution[0], coor))

def selection(population, k=2):
    candidates = random.sample(population, k=k)
    return sorted(candidates, key=lambda x: x[1], reverse=True)[0]

def ga(coor, popsize, gen):
    pop = []
    for i in range(popsize):
        sol = init_random(coor)
        length = fitness(sol, coor)
        pop.append((sol, length))
    count = 0
    while count < gen:
        offspring = []
        while len(offspring) < len(pop):
            p1 = selection(pop, k=4)
            p2 = selection(pop, k=4)
            o1, o2 = crossover(p1, p2, coor)
            o1 = mutate(o1, 0.1, coor)
            o2 = mutate(o2, 0.1, coor)
            offspring.append(o1)
            offspring.append(o2)
        pop.extend(offspring)
        pop = sorted(pop, key=lambda x: x[0], reverse=True)[:popsize]
        count += 1
    return pop[0]

def get_nearest(idx, new_coor, distances):
    nearest_idx = []
    nearest_dist = 10e10
    for key, value in new_coor.items():
        if key != idx:
            #dist = get_distances(new_coor[idx], new_coor[key])
            dist = distances[tuple(sorted((idx, key)))][0]
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_idx = []
                nearest_idx.append(key)
            elif dist == nearest_dist:
                nearest_idx.append(key)
    return nearest_idx

def near_neighbor(coor, distances):
    coor_tmp = coor.copy()
    solution = []
    init_idx = random.randint(0, len(coor)-1)
    solution.append(init_idx)
    while len(solution) != len(coor):
        nearest_idx = get_nearest(init_idx, coor_tmp, distances)
        # if len(nearest_idx) > 1:
        #     print(len(nearest_idx))
        next_idx = random.sample(nearest_idx, 1)[0]
        distances[tuple(sorted((init_idx, next_idx)))][1] += 1
        solution.append(next_idx)
        coor_tmp.pop(init_idx, None)
        init_idx = next_idx
    total_dist = fitness(solution, coor)
    return solution, total_dist, distances

def main(file, logic, limit):
    coor = get_coordinates(file)
    start_time = time.time()
    distances = dict()
    for key, value in coor.items():
        if key != len(coor):
            next_idx = key + 1
            for i in range(next_idx, len(coor)):
                distances[(key, i)] = get_distances(key, i)
    end_time = time.time()
    print(end_time - start_time)

    if logic == 'lc':
        limit = 50
        solution = local_search(coor, limit)
    elif logic == 'ga':
        popsize = 100
        limit = 1200
        solution = ga(coor, popsize, limit)
    elif logic == 'nn':
        solution = near_neighbor(coor)

    return solution

if __name__ == "__main__":
    # sol, step = main(file=args.tsp_file, logic=args.logic, limit=args.fit_limit)
    # print(sol, step)
    coor = get_coordinates(args.tsp_file)
    start_time = time.time()
    distances = dict()
    for key, value in coor.items():
        if key != len(coor):
            next_idx = key + 1
            for i in range(next_idx, len(coor)):
                distances[(key, i)] = [get_distances(coor[key], coor[i]), 0]
    end_time = time.time()
    print(end_time - start_time)
    trial = 0
    limit = 100
    while trial != limit:
        print(f'trial {int(trial) + 1} th')
        solution, total_dist, distances = near_neighbor(coor, distances)
        trial += 1
    with open('./dist_cnt_rl11849.pkl', 'wb') as f:
        pickle.dump(distances, f)
