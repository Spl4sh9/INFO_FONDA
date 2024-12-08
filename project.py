import networkx as nx

from pysat.solvers import Minicard
from pysat.formula import CNFPlus, IDPool


# Q2
def gen_solution(G: nx.Graph, k: int) -> list[tuple[int, set, set]]:
    # s in G devrait iterer sur tous les sommets de G
    max_t = 2 * len(G) + 1  # Theoreme 1
    vpool = IDPool(start_from=1)  # stockage des variables (s, t)
    cnf = CNFPlus() # CNF avec cardinatlié

    def pos(v, t):
        return vpool.id((v, t, 0))
    def mouv(v, t):
        return vpool.id((v, t, 1))

    print("Construction des clauses\n")
    print("Tous les sommets commencent à 0 (gauche)")
    cnf.append([-pos(0, 0)])  # s0 = berger
    for s in G.nodes:
        cnf.append([-pos(s, 0)])

    print("Tous les sommets finissent à 1 (droite)")
    cnf.append([pos(0, max_t)])  # s0 = berger
    for s in G.nodes:
        cnf.append([pos(s, max_t)])

    print("Contraintes sur la position des sommets")
    for s in G.nodes:
        for t in range(0, max_t):
            cnf.append([pos(s, t), -pos(s, t)])

    print("Contraintes d'alternance du berger")
    for t in range(0, max_t - 1):
        cnf.append([pos(0, t), pos(0, t + 1)])
        cnf.append([-pos(0, t), -pos(0, t + 1)])

    # print("Contraintes sur la cohérence des mouvements")
    # for s in G.nodes:
    #     for t in range(0, max_t - 1): # mouv -> rive(sommet) = rive(berger)
    #         cnf.append([-mouv(s, t), pos(s, t), -pos(0, t)])
    #         cnf.append([-mouv(s, t), -pos(s, t), pos(0, t)])
    #     for t in range(0, max_t - 1): # mouv -> rive(somment) != rive(sommet+1)
    #         cnf.append([-mouv(s, t), pos(s, t), pos(s, t+1)])
    #         cnf.append([-mouv(s, t), -pos(s, t), -pos(s, t+1)])

    # print("Contraintes sur les conflits entre sommets")
    # for (v, w) in G.edges:
    #     for t in range(0, max_t): # Faut verifier parce que huh
    #         cnf.append([pos(0, t), -pos(v, t), -pos(w, t)])
    #         cnf.append([-pos(0, t), pos(v, t), pos(w, t)])

    # Verifie que is_atmost accepte une liste vide de clauses comme vrai
    for s in G.nodes: # DEBUG
        for t in range(0, max_t): # DEBUG
            cnf.append([-mouv(s, t)]) # DEBUG

    # print("Contraintes sur le nombre k de mouvements")
    # for t in range(0, max_t):
    #     cnf.append([[mouv(s, t) for s in G.nodes], k], is_atmost=True) # Check this

    print("Clauses construites:\n")
    print(cnf.clauses)  # pour afficher les clauses

    # solver
    s = Minicard()
    s.append_formula(cnf.clauses, no_return=False)

    print("Resolution...")
    sat = s.solve()
    if sat:
        print("satisfaisable : " + str(sat))
        solution = s.get_model()
        print("solution : " + str(solution))

    return None

# Q3
def find_alcuin_number(G: nx.Graph) -> int:
    # À COMPLÉTER
    return

# Q5
def gen_solution_cvalid(G: nx.Graph, k: int, c: int) -> list[tuple[int, set, set, tuple[set]]]:
    # À COMPLÉTER
    return

# Q6
def find_c_alcuin_number(G: nx.Graph, c: int) -> int:
    # À COMPLÉTER
    return

g = nx.Graph()
g.add_nodes_from([1, 2, 3])
g.add_edges_from([(1, 2), (2, 3)])
gen_solution(g,1)

# h = nx.Graph()
# h.add_nodes_from([1, 2, 3, 4, 5, 6, 7])
# h.add_edges_from([(2, 1), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7)])
# gen_solution(h, 1)
