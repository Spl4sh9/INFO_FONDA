import networkx as nx

from pysat.solvers import Minicard
from pysat.formula import CNFPlus, IDPool
from pysat.card import CardEnc

from utils import *

PRINT_Q2 = False

# Q2
def gen_solution(G: nx.Graph, k: int) -> list[tuple[int, set, set]]:
    if k == 0:
        return None

    max_t = 2 * len(G) + 1  # Theoreme 1 : taille d'une séq. pour Alcuin(G) <= 2n + 1
    vpool = IDPool(start_from=1)  # stockage des variables (s, t, 0) pour position et (s, t, 1) pour mouvement
    cnf = CNFPlus() # CNF avec cardinatlié

    def pos(s, t):
        return vpool.id((s, t, 0))
    def mouv(s, t):
        return vpool.id((s, t, 1))

    if PRINT_Q2:
        print("Construction des clauses\n")
        print("Contraintes d'alternance du berger")
    for t in range(-1, max_t):
        cnf.append([pos(-1, t), pos(-1, t + 1)])  # Créer pos(0, 0), ..., pos(0, max_t) (= berger)
        cnf.append([-pos(-1, t), -pos(-1, t + 1)])
    cnf.append([-pos(-1, 0)])  # Impose que le berger commence à gauche
    cnf.append([pos(-1, max_t)])  # Impose que le berger finisse à droite

    if PRINT_Q2:
        print("Contraintes sur la position des sommets")
    for s in G.nodes:
        for t in range(0, max_t + 1):
            cnf.append([pos(s, t), -pos(s, t)]) # Créer pos(i, 0), ..., pos(i, max_t) pour chaque sommet s à chaque t

    if PRINT_Q2:
        print("Tous les sommets commencent à 0 (gauche)")
    for s in G.nodes:
        cnf.append([-pos(s, 0)])  # Impose que chaque sommet commence à gauche

    if PRINT_Q2:
        print("Tous les sommets finissent à 1 (droite)")
    for s in G.nodes:
        cnf.append([pos(s, max_t)]) # Impose que chaque sommet finisse à droite

    if PRINT_Q2:
        print("Contraintes sur les conflits entre sommets")
    for (v, w) in G.edges:
        for t in range(0, max_t + 1): # Verifie pour toutes les configurations donc dans [0, max_t]
            cnf.append([pos(-1, t), -pos(v, t), -pos(w, t)])
            cnf.append([-pos(-1, t), pos(v, t), pos(w, t)])

    if PRINT_Q2:
        print("Contraintes sur la cohérence des mouvements")
    for s in G.nodes:
        for t in range(0, max_t): # mouv = [0, max_t - 1], |mouv| = max_t
            # mouv (en t) -> rive(sommet) = rive(berger) (en t)
            cnf.append([-mouv(s, t), pos(s, t), -pos(-1, t)])
            cnf.append([-mouv(s, t), -pos(s, t), pos(-1, t)])
            # mouv (en t) -> rive(sommet) = rive(berger) (en t + 1)
            cnf.append([-mouv(s, t), pos(s, t + 1), -pos(-1, t + 1)])
            cnf.append([-mouv(s, t), -pos(s, t + 1), pos(-1, t + 1)])
            ### IL MANQUAIT PEUT ETRE UNE CONTRAINTE SUR LE FAIT QU UN MOUVEMENT SOIT VRAI QUAND UN SOMMET EST DEPLACE
            cnf.append([-mouv(s, t), -pos(s, t), -pos(s, t + 1)])
            cnf.append([-mouv(s, t), pos(s, t), pos(s, t + 1)])
            cnf.append([mouv(s, t), -pos(s, t), pos(s, t + 1)])
            cnf.append([mouv(s, t), pos(s, t), -pos(s, t + 1)])

    if PRINT_Q2:
        print("Contraintes sur le nombre k de mouvements")
    for t in range(0, max_t): # mouv = [0, max_t - 1], |mouv| = max_t
        cnf.extend(CardEnc.atmost(lits=[mouv(s, t) for s in G.nodes], bound=k, vpool=vpool).clauses) # choisi au plus k sommets à déplacer par t

    if PRINT_Q2:
        print("Clauses construites:\n")

    # solver
    s = Minicard()
    s.append_formula(cnf.clauses, no_return=False)
    if PRINT_Q2:
        print("Resolution...")
    sat = s.solve()
    if PRINT_Q2:
        print("satisfaisable : " + str(sat))

    if sat:
        model = s.get_model()
        if PRINT_Q2:
            print("solution : " + str(model))

        positions = {}
        for t in range(max_t + 1):
            for s in G.nodes:
                positions[(s, t)] = model[vpool.id((s, t, 0)) - 1] > 0

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
    k = len(G) # avec k = n on est sûr d'avoir une solution (inutile de la tester)
    max_k, min_k = k, 1

    while min_k < max_k: # Recherche dichotomique
        k = (min_k + max_k) // 2 

        if gen_solution(G, k) is None:  # k n'est pas la solution
            min_k = k + 1 # sol > k
        else: # k est une solution
            max_k = k # sol <= k

    return min_k

# Q5
def gen_solution_cvalid(G: nx.Graph, k: int, c: int) -> list[tuple[int, set, set, tuple[set]]]:
    # À COMPLÉTER
    return

# Q6
def find_c_alcuin_number(G: nx.Graph, c: int) -> int:
    # À COMPLÉTER
    return

# Test
g = nx.Graph()
g.add_nodes_from([1, 2, 3, 4, 5, 6, 7])
g.add_edges_from([(1, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7)])
solution = find_alcuin_number(g)
print("Final k = ", solution)

# Should return None
# h = nx.Graph()
# h.add_nodes_from([1, 2, 3, 4])
# h.add_edges_from([(2, 1), (2, 3), (2, 4)])
# gen_solution(h, 1) 
