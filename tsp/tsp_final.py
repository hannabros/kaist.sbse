import os, sys
import argparse
import random
import math
import time

from numpy.core.fromnumeric import trace

parser = argparse.ArgumentParser(description='Travelling Sales Person Problem Arguments')

parser.add_argument('-p', '--population', default=None, type=int, help='Population')
parser.add_argument('-f', '--fit-limit', default=100, type=int, help='Fitness Evaluation Limit') 
parser.add_argument('-l', '--logic', default='greedy', type=str, help='Name of Logic')
parser.add_argument('-s', '--sub', default=10, type=int, help='# of partial length')
parser.add_argument('--tsp-file', default=str, help='TSP file path')
parser.add_argument('--output', default=str, help='TSP output path')

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

def save(path, solution, total_dist):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    filename = int(time.time())
    with open(os.path.join(path, f'solution_{filename}.txt'), 'w') as f:
        for idx in solution:
            f.write(str(idx+1))
            f.write('\n')
    
    if os.path.isfile(os.path.join(path, 'distance.best')):
        with open(os.path.join(path, 'distance.best'), 'r') as f:
            lines = f.readlines()
        best_filename = lines[0].split()[0]
        best_distance = float(lines[0].split()[1])
        if total_dist < best_distance:
            best_distance = total_dist
            best_filename = filename
    else:
        best_filename = filename
        best_distance = total_dist
    with open(os.path.join(path, 'distance.best'), 'w') as f:
        f.writelines(f'{best_filename} {best_distance}')    

class Fitness():
    def __init__(self):
        pass
        
    def get_distances(self, start_coor, end_coor):
        distance = math.sqrt(sum([(p - q)**2 for (p, q) in zip(start_coor, end_coor)]))
        return distance

    def fitness(self, solution, coor, **kwargs):
        total_dist = 0
        for idx, coor_idx in enumerate(solution[:-1]):
            start, end = coor_idx, solution[idx+1]
            # if end < start:
            #     tmp = start
            #     start = end
            #     end = tmp
            total_dist += self.get_distances(coor[start], coor[end])
        if not 'partial' in kwargs:
            total_dist += self.get_distances(coor[solution[0]], coor[solution[-1]])
        return total_dist

class NN():
    def __init__(self, coor, sub, limit):
        self.ff = Fitness()
        self.coor = coor
        self.init_idx = random.randint(0, len(coor)-1)
        self.init_dist = self.ff.fitness(random.sample([co for co in range(len(self.coor))], len(self.coor)), self.coor)
        self.sub = sub
        self.limit = limit

    def get_nearest(self, idx, new_coor):
        nearest_idx = []
        init_dist = self.init_dist
        for key, value in new_coor.items():
            if key != idx:
                dist = self.ff.get_distances(new_coor[idx], new_coor[key])
                if dist < init_dist:
                    init_dist = dist
                    nearest_idx = []
                    nearest_idx.append(key)
                elif dist == init_dist:
                    nearest_idx.append(key)
        return nearest_idx

    def get_nearest_prob(self, idx, new_coor):
        nearest_idx = []
        keys = []
        distances = []
        for key, value in new_coor.items():
            if key != idx:
                dist = self.ff.get_distances(new_coor[idx], new_coor[key])
                keys.append(key)
                distances.append(dist)
        if 0.0 in distances:
            nearest_idx = keys[distances.index(0.0)]
        else:
            nearest_idx = random.choices(keys, self.normal_inverse(distances))
        return nearest_idx

    def normal_inverse(self, dist):
        inverse_dist = [1/d for d in dist]
        total = sum(inverse_dist)
        n_inverse = [float(d)/total for d in inverse_dist]
        return n_inverse

    def greedy_solve(self):
        solution = []
        init_dist = self.init_dist
        coor_tmp = self.coor.copy()
        init_idx = self.init_idx
        solution.append(init_idx)
        for i in range(self.limit):
            while len(solution) != len(self.coor):
                nearest_idx = self.get_nearest(init_idx, coor_tmp)
                next_idx = random.sample(nearest_idx, 1)[0]
                solution.append(next_idx)
                coor_tmp.pop(init_idx, None)
                init_idx = next_idx
            total_dist = self.ff.fitness(solution, self.coor)
            if total_dist < init_dist:
                init_dist = total_dist
        total_dist = init_dist
        return solution, total_dist

    def prob_solve(self):
        solution = []
        init_dist = self.init_dist
        coor_tmp = self.coor.copy()
        init_idx = self.init_idx
        solution.append(init_idx)
        for i in range(self.limit):
            while len(solution) != len(self.coor):
                nearest_idx = self.get_nearest_prob(init_idx, coor_tmp)
                if isinstance(nearest_idx, list):
                    next_idx = random.sample(nearest_idx, 1)[0]
                else:
                    next_idx = nearest_idx
                solution.append(next_idx)
                coor_tmp.pop(init_idx, None)
                init_idx = next_idx
            total_dist = self.ff.fitness(solution, self.coor)
            if total_dist < init_dist:
                init_idst = total_dist
        total_dist = init_idst
        return solution, total_dist

    def partial_solve(self):
        solution = []
        # total_dist = 0
        init_idx = self.init_idx
        coor_tmp = self.coor.copy()
        sub_length = len(self.coor) // self.sub
        sub_length_list = [sub_length] * self.sub
        if len(self.coor) % self.sub != 0:
            sub_length_list.append(len(self.coor) % self.sub)
        
        for sl_idx, sl in enumerate(sub_length_list):
            trial = 0
            init_dist = self.init_dist
            while trial != self.limit:
                tmp_sol = []
                coor_partial_tmp = coor_tmp.copy()
                tmp_init_idx = init_idx
                for i in range(sl):
                    if len(coor_partial_tmp) == 1:
                        break
                    nearest_idx = self.get_nearest(tmp_init_idx, coor_partial_tmp)
                    next_idx = random.sample(nearest_idx, 1)[0]
                    tmp_sol.append(next_idx)
                    coor_partial_tmp.pop(tmp_init_idx, None)
                    tmp_init_idx = next_idx
                partial_dist = self.ff.fitness(tmp_sol, self.coor, partial=True)
                if partial_dist < init_dist:
                    print(f'previous partial distance was {init_dist}. partial distance changed to {partial_dist}')
                    init_dist = partial_dist
                    partial_sol = tmp_sol
                trial += 1
            # total_dist += partial_dist
            solution.extend(partial_sol)
            for sol_idx in partial_sol[:-1]:
                coor_tmp.pop(sol_idx, None)
            init_idx = random.sample(self.get_nearest(partial_sol[-1], coor_tmp), 1)[0]
            coor_tmp.pop(partial_sol[-1], None)
        total_dist = self.ff.fitness(solution, self.coor)
        return solution, total_dist

if __name__ == "__main__":
    coor = get_coordinates(args.tsp_file)

    # with open('/Users/hangyeolsun/workspace/kaist.sbse/tsp/solution/rl11849/solution_0000.txt', 'r') as f:
    #     lines = f.read().splitlines()
    # ff = Fitness()
    # distance = ff.fitness([int(l)-1 for l in lines], coor)
    # print(distance)   

    if args.logic == 'greedy_nn':
        sub = None
        limit = args.fit_limit
        nn = NN(coor, sub, limit)
        solution, total_dist = nn.greedy_solve()
    elif args.logic == 'prob_nn':
        sub = None
        limit = args.fit_limit
        nn = NN(coor, sub, limit)
        solution, total_dist = nn.prob_solve()
    elif args.logic == 'partial_nn':
        sub = args.sub
        limit = args.fit_limit
        nn = NN(coor, sub, limit)
        solution, total_dist = nn.partial_solve()
    else:
        sys.exit(1)

    print(solution, total_dist)
    output_path = os.path.join(args.output, args.tsp_file.split('/')[-1].split('.')[0])
    save(output_path, solution, total_dist)