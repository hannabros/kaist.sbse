import random
import math

secret = 837491
secret_len = len(str(secret))

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
    p = random.randint(0, len(sol1) - 1)
    o1 = sol1[1][:p] + sol2[1][p:]
    o2 = sol2[1][:p] + sol1[1][p:]
    return o1, o2

def mutate(solution):
    mutated = solution[1][:]
    p = random.randint(0, len(solution))
    mutated[p] = random.randint(0, 9)
    return (fitness(mutated), mutated)

def selection(population, k=2):
    candidates = random.choices(population, k=k)
    return sorted(candidates, key=lambda x: x[1], reverse=True)[0]

##solution = (fitness, [])

def ga(popsize, gen):
    pop = []
    for i in range(popsize):
        pop.append(random_solution(secret_len))
    
    while True:
        offspring = []
        while len(offspring) < len(pop):
            p1 = selection(population, k=3)
            p2 = selection(population, k=3)

            o1, o2 = crossover(p1, p2)
            o1 = mutate(o1)
            o2 = mutate(o2)

            offspring.append(o1)
            offspring.append(o2)

        pop = sorted(pop, key=lambda x: x[0], reversed=True)
        offspring = sorted(offspring, key=lambda x : x[0], reversed=True)

        offspring[:10] = pop[:10]

        pop = offspring
        pop = sorted(pop, key=lambda x: x[0], reversed=True)
        print(pop[0])

        if pop[0][0] == secret_len:
            break

if __name__ == "__main__":
    ga(80, 20)
