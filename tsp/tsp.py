import os, sys
import argparse
import random
import math

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
def get_distances(coor):
    dist = {}
    total_len = len(coor)
    for i in range(total_len):
        for j in range(i+1, total_len):
            length = math.sqrt(sum([(p - q)**2 for (p, q) in zip(coor[i], coor[j])]))
            dist[(i, j)] = length
    return dist

# solution : sequence of index of coordinates
def fitness(dist, solution):
    total_length = 0
    for idx, coor_idx in enumerate(solution[:-1]):
        start, end = coor_idx, solution[idx+1]
        if end < start:
            tmp = start
            start = end
            end = tmp
        total_length += dist[(start, end)]
    return total_length

def init_random(coor):    
    return random.sample(list(range(len(coor))), len(coor))

def get_neighbors(solution):
    neighbors = []
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            neighbour = solution.copy()
            neighbour[i] = solution[j]
            neighbour[j] = solution[i]
            neighbors.append(neighbour)
    return neighbors
    
# should add limit instance
# should change to local search (global search currently)
def local_search(coor, dist):
    init_sol = init_random(coor)
    neighbors = get_neighbors(init_sol)
    init_length = fitness(dist, init_sol)

    step = 0
    for neighbor in neighbors:
        new_length = fitness(dist, neighbor)
        if new_length < init_length:
            solution = neighbor
            init_length = new_length
        step += 1
    
    return solution, new_length

def crossover(sol1, sol2, dist):
    p = random.randint(1, len(sol1) - 1)
    o1 = sol1[0][:p] + sol2[0][p:]
    o2 = sol2[0][:p] + sol1[0][p:]
    return (o1, fitness(dist, o1)), (o2, fitness(dist, o2))

def mutate(solution, rate, dist):
    for i in range(len(solution[0])):
        if random.random() < rate:
            randint = random.randint(0, len(solution[0])-1)
            if randint not in solution[0]:
                solution[0][i] = randint
    return (solution[0], fitness(dist, solution[0]))

def selection(population, k=2):
    candidates = random.sample(population, k=k)
    return sorted(candidates, key=lambda x: x[1], reverse=True)[0]

def ga(coor, dist, popsize, gen):
    pop = []
    for i in range(popsize):
        sol = init_random(coor)
        length = fitness(dist, sol)
        pop.append((sol, length))
    count = 0
    while count < gen:
        offspring = []
        while len(offspring) < len(pop):
            p1 = selection(pop, k=4)
            p2 = selection(pop, k=4)
            o1, o2 = crossover(p1, p2, dist)
            o1 = mutate(o1, 0.1, dist)
            o2 = mutate(o2, 0.1, dist)
            offspring.append(o1)
            offspring.append(o2)
        pop.extend(offspring)
        pop = sorted(pop, key=lambda x: x[0], reverse=True)[:popsize]
        count += 1
    return pop[0]

def main(file, logic):
    coor = get_coordinates(file)
    dist = get_distances(coor)

    if args.logic == 'lc':
        solution = local_search(coor, dist)
    elif args.logic == 'ga':
        popsize = 10
        gen = 50
        solution = ga(coor, dist, popsize, gen)

    return solution

if __name__ == "__main__":
    '''
    sol, step = main(file=args.tsp_file, logic=args.logic)
    print(sol, step)
    '''
    # visualization
    import matplotlib.pyplot as plt
    coor = get_coordinates(file=args.tsp_file)
    x, y = [], []
    for key, value in coor.items():
        x.append(value[0])
        y.append(value[1])
    plt.plot(x, y, 'o', color='black')
