import random
import math
secret = [8,3,7,4,2,5,4,7]
secret_len = len(secret)
# fitness: tells us how many digits we got correct
# for example, secret = 837462, number = 827123 --> fitness: 2
def fitness(solution):
    fitness = 0
    for sec, sol in zip(secret, solution):
        if sec == sol:
            fitness += 1
    return fitness
def random_solution(digits):
    s = []
    for j in range(digits):
        d = random.randint(0, 9)
        s.append(d)
    return (fitness(s), s)
def crossover(sol1, sol2):
    p = random.randint(1, len(sol1) - 1)
    o1 = sol1[1][:p] + sol2[1][p:]
    o2 = sol2[1][:p] + sol1[1][p:]
    return (fitness(o1), o1), (fitness(o2), o2)
def mutate(solution, rate):
    for i in range(len(solution[1])):
        if random.random() < rate:
            solution[1][i] = random.randint(0, 9)
    return (fitness(solution[1]), solution[1])
def selection(population, k=2):
    candidates = random.sample(population, k=k)
    return sorted(candidates, key=lambda x: x[0], reverse=True)[0]
## solution = (fitness, [])
def ga(popsize, gen):
    pop = []
    for i in range(popsize):
        pop.append(random_solution(secret_len))
    count = 0    
    while count < gen:
        offspring = []
        while len(offspring) < len(pop):
            p1 = selection(pop, k=4)
            p2 = selection(pop, k=4)
            o1, o2 = crossover(p1, p2)
            o1 = mutate(o1, 0.1)
            o2 = mutate(o2, 0.1)
            offspring.append(o1)
            offspring.append(o2)
        # generational selection
        pop.extend(offspring) 
        pop = sorted(pop, key=lambda x: x[0], reverse=True)[:popsize]
        count += 1
    return pop[0]
if __name__ == '__main__':
    print(ga(20, 50))