import random
import math
MAX=100
LEN = 5
BUDGET = 200000
# Create a hiddn vector of length LEN
hidden = []
for i in range(LEN):
    hidden.append(random.randrange(MAX))

# compute the Euclidean distance between solution and hidden vector
def evaluate(solution):
    assert len(solution) == len(hidden)
    return math.sqrt(sum([(p - q)**2 for (p, q) in zip(hidden, solution)]))

def init_random(len, max):
    v = []
    for i in range(len):
        v.append(random.randrange(max))
    return v

def get_neighbour(sol):
    neighbours = []
    for i in range(len(sol)):
        n1 = sol[:]
        n1[i] += 1
        n2 = sol[:]
        n2[i] -= 1
        neighbours.append(n1)
        neighbours.append(n2)
    return neighbours

def hillclimbing(budget):
    # initial solution = random
    sol = init_random(LEN, MAX)
    current_fitness = evaluate(sol)
    spent = 0
    while True:
        neighbours = get_neighbour(sol)
        climb = False
        for n in neighbours:
            nf = evaluate(n)
            spent += 1
            if nf < current_fitness:
                sol = n[:]
                current_fitness = nf
                climb = True
        if not climb:
            break
    # repeat
        # get neighbors
        # if there is a better neighbor, move
        # otherwise return current solution
    return sol, spent

EXP = 0
PATTERN = 1
PLUS = 0
MINUS = 1
def avm(budget):
    spent = 0
    current_var = 0
    mode = EXP

    sol = init_random(LEN, MAX)
    fit_c = evaluate(sol)

    changed = [0] * LEN

    while spent < budget:
        if mode == EXP:
            sol_p = sol[:]
            sol_p[current_var] += 1
            sol_m = sol[:]
            sol_m[current_var] -= 1

            fit_p = evaluate(sol_p)
            spent += 1
            fit_m = evaluate(sol_m)
            spent += 1
            if fit_p < fit_c:
                direction = PLUS
                sol = sol_p[:]
                fit_c = fit_p
                changed[current_var] = 1
            elif fit_m < fit_c:
                direction = MINUS
                sol = sol_m[:]
                fit_c = fit_m
                changed[current_var] = 1
            else:
                changed[current_var] = 0
                current_var += 1
                if current_var == LEN:
                    if sum(changed) == 0:
                        break
                    current_var = 0

        elif mode == PATTERN:
            overshoot = FALSE
            step = 1
            while not overshoot:
                delta = 2 ** step
                if direction == PLUS:
                    new_sol = sol[:]
                    new_sol[current_var] += delta
                else:
                    new_sol = sol[:]
                    new_sol[current_var] -= delta

                fit_n = evaluate(new_sol)
                spent += 1
                if fit_n < fit_c:
                    step += 1
                    sol = new_sol[:]
                    fit_c = fit_n
                else:
                    overshoot = True
            mode = EXP
            current_var = (current_var + 1) % LEN
    return sol, spent

random_solution = init_random(LEN, MAX)
#print(random_solution)
#neighbours = get_neighbour(random_solution)
#print(neighbours)
#print(hidden)
guess, spent = hillclimbing(BUDGET)
print(evaluate(guess), spent)
guess, spent = avm(BUDGET)
print(evaluate(guess), spent)
