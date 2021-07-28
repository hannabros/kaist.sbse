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
    nearest_dist = 10e10
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

def near_neighbor(coor, sub, limit):
    total_dist = 0
    total_sol = []
    init_idx = random.randint(0, len(coor)-1)
    coor_tmp = coor.copy()
    sub_length = len(coor) // sub
    sub_length_list = [sub_length] * sub
    if len(coor) % sub != 0:
        sub_length_list.append(len(coor) % sub)
    
    for sl_idx, sl in enumerate(sub_length_list):
        trial = 0
        init_dist = 10e10
        while trial != limit:
            tmp_sol = []
            coor_partial_tmp = coor_tmp.copy()
            tmp_init_idx = init_idx
            for i in range(sl):
                if len(coor_partial_tmp) == 1:
                    break
                nearest_idx = get_nearest(tmp_init_idx, coor_partial_tmp)
                next_idx = random.sample(nearest_idx, 1)[0]
                tmp_sol.append(next_idx)
                coor_partial_tmp.pop(tmp_init_idx, None)
                tmp_init_idx = next_idx
            partial_dist = fitness(tmp_sol, coor)
            if partial_dist < init_dist:
                print(f'previous partial distance was {init_dist}. partial distance changed to {partial_dist}')
                init_dist = partial_dist
                partial_sol = tmp_sol
            trial += 1
        total_dist += partial_dist
        total_sol.extend(partial_sol)
        for sol_idx in partial_sol[:-1]:
            coor_tmp.pop(sol_idx, None)
        init_idx = random.sample(get_nearest(partial_sol[-1], coor_tmp), 1)[0]
        coor_tmp.pop(partial_sol[-1], None)
    return total_sol, total_dist

def main(file, logic, limit):
    coor = get_coordinates(file)
    #dist = get_distances(coor)
    
    if logic == 'nn':
        sub = 50
        limit = 5
        solution = near_neighbor(coor, sub, limit)

    return solution

if __name__ == "__main__":
    sol, step = main(file=args.tsp_file, logic=args.logic, limit=args.fit_limit)
    print(sol, step)