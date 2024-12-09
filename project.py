import networkx as nx

from pysat.solvers import Minicard
from pysat.formula import CNFPlus, IDPool
from pysat.card import CardEnc # Pour les contraintes de cardinalité

from utils import *


PRINT_Q2_CLAUSIS = False
PRINT_Q2_SOL = False
PRINT_Q5_CLAUSIS = False
PRINT_Q5_SOL = False

# Q2
def gen_solution(G: nx.Graph, k: int) -> list[tuple[int, set, set]]:
    if k == 0:
        return None

    max_t = 2 * len(G) + 1  # Theoreme 1 : taille d'une séq. pour Alcuin(G) <= 2n + 1
    vpool = IDPool(start_from=1)  # stockage des variables (s, t, 0) pour position et (s, t, 1) pour mouvement
    cnf = CNFPlus()

    def pos(s, t):
        return vpool.id((s, t, 'pos'))
    def mouv(s, t):
        return vpool.id((s, t, 'mov'))

    if PRINT_Q2_CLAUSIS: print("Construction des clauses...\n\nContraintes d'alternance du berger")
    for t in range(0, max_t):
        cnf.append([pos(-1, t), pos(-1, t + 1)])
        cnf.append([-pos(-1, t), -pos(-1, t + 1)])
    cnf.append([-pos(-1, 0)])  # Impose que le berger commence à gauche
    cnf.append([pos(-1, max_t)])  # Impose que le berger finisse à droite

    if PRINT_Q2_CLAUSIS: print("Tous les sommets commencent à 0 (gauche)")
    for s in G.nodes:
        cnf.append([-pos(s, 0)])  # Impose que chaque sommet commence à gauche

    if PRINT_Q2_CLAUSIS: print("Tous les sommets finissent à 1 (droite)")
    for s in G.nodes:
        cnf.append([pos(s, max_t)]) # Impose que chaque sommet finisse à droite

    if PRINT_Q2_CLAUSIS: print("Contraintes sur les conflits entre sommets")
    for (v, w) in G.edges:
        for t in range(0, max_t + 1): # Verifie pour toutes les configurations donc dans [0, max_t]
            cnf.append([pos(-1, t), -pos(v, t), -pos(w, t)])
            cnf.append([-pos(-1, t), pos(v, t), pos(w, t)])

    if PRINT_Q2_CLAUSIS: print("Contraintes sur la cohérence des mouvements")
    for s in G.nodes:
        for t in range(0, max_t): # mouv = [0, max_t - 1], |mouv| = max_t
            # mouv(sommet, t) -> [rive(sommet, t) = rive(berger, t)]
            cnf.append([-mouv(s, t), pos(s, t), -pos(-1, t)])
            cnf.append([-mouv(s, t), -pos(s, t), pos(-1, t)])
            # mouv(sommet, t) -> [rive(sommet, t+1) = rive(berger, t+1)]
            cnf.append([-mouv(s, t), pos(s, t + 1), -pos(-1, t + 1)])
            cnf.append([-mouv(s, t), -pos(s, t + 1), pos(-1, t + 1)])
            # [rive(sommet, t) != rive(sommet, t + 1)] -> mouv(sommet, t)
            cnf.append([-mouv(s, t), -pos(s, t), -pos(s, t + 1)])
            cnf.append([-mouv(s, t), pos(s, t), pos(s, t + 1)])
            cnf.append([mouv(s, t), -pos(s, t), pos(s, t + 1)])
            cnf.append([mouv(s, t), pos(s, t), -pos(s, t + 1)])

    if PRINT_Q2_CLAUSIS: print("Contraintes sur le nombre k de mouvements")
    for t in range(0, max_t): # mouv = [0, max_t - 1], |mouv| = max_t
        cnf.extend(CardEnc.atmost(lits=[mouv(s, t) for s in G.nodes], bound=k, vpool=vpool).clauses) # choisi au plus k sommets à déplacer par t

    if PRINT_Q2_CLAUSIS: print("Clauses construites.\n")

    # solver
    s = Minicard()
    s.append_formula(cnf.clauses, no_return=False)
    if PRINT_Q2_SOL: print("Resolution...")
    sat = s.solve()
    if PRINT_Q2_SOL: print("satisfaisable : " + str(sat))

    if sat:
        model = s.get_model()
        if PRINT_Q2_SOL: print("solution : " + str(model))

        positions = {}
        for t in range(max_t + 1):
            for s in G.nodes:
                positions[(s, t)] = model[vpool.id((s, t, 'pos')) - 1] > 0

        solution = []
        for t in range(max_t + 1):
            right, left = set(), set()
            b_t = t % 2 # gauche si pair, droite si impair
            for s in G.nodes:
                if positions[(s, t)]:
                    right.add(s)
                else:
                    left.add(s)
            solution.append((b_t, left, right))
        return solution

    return None

# Q3
def find_alcuin_number(G: nx.Graph) -> int:
    max_k, min_k = len(G), 1 # max = n est une solution sûre

    while min_k < max_k: # Recherche dichotomique
        k = (min_k + max_k) // 2 
        if gen_solution(G, k) is None:  # k n'est pas la solution
            min_k = k + 1 # sol > k
        else: # k est une solution
            max_k = k # sol <= k

    return min_k

# Q5
def gen_solution_cvalid(G: nx.Graph, k: int, c: int) -> list[tuple[int, set, set, tuple[set]]]:
    """
    Reprend le code de la question #2 et ajoute les nouvelles contraintes
    """
    if k == 0:
        return None

    max_t = 2 * len(G) + 1  # Theoreme 1 : taille d'une séq. pour Alcuin(G) <= 2n + 1
    vpool = IDPool(start_from=1)  # stockage des variables (s, t, 0) pour position et (s, t, 1) pour mouvement
    cnf = CNFPlus()

    def pos(s, t):
        return vpool.id((s, t, 'pos'))
    def mouv(s, t):
        return vpool.id((s, t, 'mov'))
    def comp(s, t, c):
        return vpool.id((s, t+1, c))

    if PRINT_Q2_CLAUSIS: print("Construction des clauses...\n\nContraintes d'alternance du berger")
    for t in range(0, max_t):
        cnf.append([pos(-1, t), pos(-1, t + 1)])
        cnf.append([-pos(-1, t), -pos(-1, t + 1)])
    cnf.append([-pos(-1, 0)])  # Impose que le berger commence à gauche
    cnf.append([pos(-1, max_t)])  # Impose que le berger finisse à droite

    if PRINT_Q2_CLAUSIS: print("Tous les sommets commencent à 0 (gauche)")
    for s in G.nodes:
        cnf.append([-pos(s, 0)])  # Impose que chaque sommet commence à gauche

    if PRINT_Q2_CLAUSIS: print("Tous les sommets finissent à 1 (droite)")
    for s in G.nodes:
        cnf.append([pos(s, max_t)]) # Impose que chaque sommet finisse à droite

    if PRINT_Q2_CLAUSIS: print("Contraintes sur les conflits entre sommets")
    for (v, w) in G.edges:
        for t in range(0, max_t + 1): # Verifie pour toutes les configurations donc dans [0, max_t]
            cnf.append([pos(-1, t), -pos(v, t), -pos(w, t)])
            cnf.append([-pos(-1, t), pos(v, t), pos(w, t)])

    if PRINT_Q2_CLAUSIS: print("Contraintes sur la cohérence des mouvements")
    for s in G.nodes:
        for t in range(0, max_t): # mouv = [0, max_t - 1], |mouv| = max_t
            # mouv(sommet, t) -> [rive(sommet, t) = rive(berger, t)]
            cnf.append([-mouv(s, t), pos(s, t), -pos(-1, t)])
            cnf.append([-mouv(s, t), -pos(s, t), pos(-1, t)])
            # mouv(sommet, t) -> [rive(sommet, t+1) = rive(berger, t+1)]
            cnf.append([-mouv(s, t), pos(s, t + 1), -pos(-1, t + 1)])
            cnf.append([-mouv(s, t), -pos(s, t + 1), pos(-1, t + 1)])
            # [rive(sommet, t) != rive(sommet, t + 1)] -> mouv(sommet, t)
            cnf.append([-mouv(s, t), -pos(s, t), -pos(s, t + 1)])
            cnf.append([-mouv(s, t), pos(s, t), pos(s, t + 1)])
            cnf.append([mouv(s, t), -pos(s, t), pos(s, t + 1)])
            cnf.append([mouv(s, t), pos(s, t), -pos(s, t + 1)])

    if PRINT_Q2_CLAUSIS: print("Contraintes sur le nombre k de mouvements")
    for t in range(0, max_t): # mouv = [0, max_t - 1], |mouv| = max_t
        cnf.extend(CardEnc.atmost(lits=[mouv(s, t) for s in G.nodes], bound=k, vpool=vpool).clauses) # choisi au plus k sommets à déplacer par t

    if PRINT_Q5_CLAUSIS: print("Contraintes sur les compartiments en fonction des mouvements")
    for s in G.nodes: # mouv(s, t) -> [comp(s, t+1, c) | c in [0, c]]
        for t in range(0, max_t):
            clause_1 = [-mouv(s, t)]
            for c in range(1, c+1):
                clause_1.append(comp(s, t, c))
            cnf.append(clause_1) # au moins un comp(s, t, c) est vrai si mouv(s, t) est vrai
            for c in range(1, c+1):
                cnf.append([mouv(s, t), -comp(s, t, c)])

    if PRINT_Q5_CLAUSIS : print("Contraintes sur l'unicité des compartiments par sommet")
    for t in range(0, max_t):
        for s in G.nodes:
            cnf.extend(CardEnc.atmost(lits=[comp(s, t, c) for c in range(1, c+1)], bound=1, vpool=vpool).clauses) # au plus un comp(s, t, c) est vrai à la fois

    if PRINT_Q5_CLAUSIS: print("Contraintes sur les conflits dans les compartiments")
    for (v,w) in G.edges:
        for t in range(0, max_t):
            for c in range(1, c+1):
                cnf.append([-comp(v, t, c), -comp(w, t, c)])

    if PRINT_Q5_CLAUSIS: print("Clauses construites.\n")

    # solver
    s = Minicard()
    s.append_formula(cnf.clauses, no_return=False)
    if PRINT_Q5_SOL: print("Resolution...")
    sat = s.solve()
    if PRINT_Q5_SOL: print("satisfaisable : " + str(sat))

    if sat:
        model = s.get_model()
        if PRINT_Q5_SOL: print("solution : " + str(model))

        positions = {}
        for t in range(max_t + 1):
            for s in G.nodes:
                positions[(s, t)] = model[vpool.id((s, t, 'pos')) - 1] > 0
        solution = []
        for t in range(max_t + 1):
            right, left = set(), set()

            comp_t = []
            if t > 0:
                for c in range(1, c+1):
                    comp_c = set()
                    for s in G.nodes:
                        if model[vpool.id((s, t, c)) - 1] > 0:
                            comp_c.add(s)
                    comp_t.append(comp_c)

            b_t = t % 2
            for s in G.nodes:
                if positions[(s, t)]:
                    right.add(s)
                else:
                    left.add(s)
            solution.append((b_t, left, right, tuple(comp_t)))
        return solution

    return None

# Q6
def find_c_alcuin_number(G: nx.Graph, c: int) -> int:
    # max_k, min_k = len(G), 1 # max = n est une solution sûre

    # while min_k < max_k: # Recherche dichotomique
    #     print('min, max =', min_k, max_k)
    #     k = (min_k + max_k) // 2 
    #     if gen_solution_cvalid(G, c, k) is None:  # k n'est pas la solution
    #         print("Failed for k = ", k)
    #         min_k = k + 1 # sol > k
    #     else: # k est une solution
    #         max_k = k # sol <= k

    # if gen_solution_cvalid(G, c, min_k) is None:
    #     return float('+inf') 
    k = 1
    going_on = True
    while going_on and k != len(G.nodes):
        if gen_solution_cvalid(G, k, c) is None:
            k += 1
        else:
            going_on = False
    if k == len(G.nodes):
        return float('+inf')
    return k


# # Test Q2 - devrait trouver une solution
# print("Test Q2")
# g = nx.Graph()
# g.add_nodes_from([1, 2, 3])
# g.add_edges_from([(1, 2), (2, 3)])
# solution = gen_solution(g, 1)
# for s in solution:
#     print(s)

# # Test Q3 - devrait trouver k = 2
# h = nx.Graph()
# h.add_nodes_from([1, 2, 3])
# h.add_edges_from([(1, 2), (2, 3)])
# solution = find_alcuin_number(h)
# print("Final k = ", solution)

# # Test Q2 - devrait renvoyer None
# h = nx.Graph()
# h.add_nodes_from([1, 2, 3, 4])
# h.add_edges_from([(2, 1), (2, 3), (2, 4)])
# gen_solution(h, 1) 

#TEST Q5 - devrait trouver une solution
# print("Test Q5")
# n = 4
# solution = gen_solution_cvalid(CompleteGraph(n), n-1, n-1)
# for s in solution:
#     print(s)

