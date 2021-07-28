import os, sys
import argparse
import random
import math
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
    
def get_nearest(idx, new_coor):
    nearest_idx = []
    nearest_dist = 10000
    for key, value in new_coor.items():
        if key != idx:
            dist = get_distances(new_coor[idx], new_coor[key])
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_idx = []
                nearest_idx.append(key)
            elif dist == nearest_dist:
                nearest_idx.append(key)

    return nearest_idx

def normal_inverse(dist):
    inverse_dist = [1/d for d in dist]
    total = sum(inverse_dist)
    n_inverse = [float(d)/total for d in inverse_dist]
    return n_inverse

def with_prb(idx, keys, distances):
    inverse_p = []
    for k in keys:
        cnt = distances[tuple(sorted((idx, k)))][1]
        inverse_p.append(cnt/100)
    print(inverse_p)
    return inverse_p

def get_nearest2(idx, new_coor, distances):
    nearest_idx = []
    keys = []
    dist_list = []
    for key, value in new_coor.items():
        if key != idx:
            # dist = get_distances(new_coor[idx], new_coor[key])
            dist = distances[tuple(sorted((idx, key)))][0]
            keys.append(key)
            dist_list.append(dist)
    if 0.0 in distances:
        nearest_idx = keys[dist_list.index(0.0)]
    else:
        # nearest_idx = random.choices(keys, normal_inverse(dist_list))
        nearest_idx = random.choices(keys, with_prb(idx, keys, distances))
    return nearest_idx

def near_neighbor(coor, distances):
    coor_tmp = coor.copy()
    solution = []
    init_idx = random.randint(0, len(coor)-1)
    solution.append(init_idx)
    while len(solution) != len(coor):
        nearest_idx = get_nearest2(init_idx, coor_tmp, distances)
        if isinstance(nearest_idx, list):
            next_idx = random.sample(nearest_idx, 1)[0]
        else:
            next_idx = nearest_idx
        solution.append(next_idx)
        coor_tmp.pop(init_idx, None)
        init_idx = next_idx
    total_dist = fitness(solution, coor)
    return solution, total_dist

def main(file, logic, limit):
    coor = get_coordinates(file)
    #dist = get_distances(coor)

    if logic == 'nn':
        solution = near_neighbor(coor)

    return solution

if __name__ == "__main__":
    # sol, step = main(file=args.tsp_file, logic=args.logic, limit=args.fit_limit)
    # print(sol, step)
    coor = get_coordinates(args.tsp_file)
    with open('./dist_cnt_rl11849.pkl', 'rb') as f:
        distances = pickle.load(f)

    solution, total_dist = near_neighbor(coor, distances)
    print(solution, total_dist)

    