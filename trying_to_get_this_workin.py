import networkx as nx

from pysat.solvers import Minicard
from pysat.formula import CNFPlus, IDPool
from pysat.card import CardEnc


# Q2
def gen_solution(G: nx.Graph, k: int) -> list[tuple[int, set, set]]:
    max_t = 2 * len(G) + 1  # Theoreme 1 : taille d'une séq. pour Alcuin(G) <= 2n + 1
    vpool = IDPool(start_from=1)  # stockage des variables (s, t, 0) pour position et (s, t, 1) pour mouvement
    cnf = CNFPlus() # CNF avec cardinatlié

    def pos(s, t):
        return vpool.id((s, t, 0))
    def mouv(s, t):
        return vpool.id((s, t, 1))

    print("Construction des clauses\n")
    print("Tous les sommets commencent à 0 (gauche)")
    cnf.append([-pos(0, 0)])  # Créer pos(0, 0) (= berger au t_initial)
    for s in G.nodes:
        cnf.append([-pos(s, 0)])  # Créer pos(1, 0), pos(2, 0), ..., pos(n, 0) (= chaque sommets au t_initial)

    print("Tous les sommets finissent à 1 (droite)") # t = [0, max_t], |t| = max_t + 1 (pour l'état final ou initial en fonction du point de vue)
    cnf.append([pos(0, max_t)])  # Créer pos(0, max_t) (= berger au t_final)
    for s in G.nodes:
        cnf.append([pos(s, max_t)]) # Créer pos(1, max_t), pos(2, max_t), ..., pos(n, max_t) (= chaque sommets au t_final)

    print("Contraintes sur la position des sommets")
    for s in G.nodes:
        for t in range(0, max_t + 1): # pos(i, 0) et pos(i, max_t) existent déjà
            cnf.append([pos(s, t), -pos(s, t)]) # Créer pos(i, 1), ..., pos(i, max_t - 1) pour chaque sommet s à chaque t

    print("Contraintes d'alternance du berger")
    for t in range(0, max_t): # pos(0, 0) et pos(0, max_t) existent déjà
        cnf.append([pos(0, t), pos(0, t + 1)])  # Créer pos(0, 1), ..., pos(0, max_t - 1) (= berger)
        cnf.append([-pos(0, t), -pos(0, t + 1)])

    print("Contraintes sur les conflits entre sommets")
    for (v, w) in G.edges:
        for t in range(0, max_t + 1): # on veut vérifier pour toutes les configurations donc dans [0, max_t]
            cnf.append([pos(0, t), -pos(v, t), -pos(w, t)])
            cnf.append([-pos(0, t), pos(v, t), pos(w, t)])

    print("Contraintes sur la cohérence des mouvements")
    for s in G.nodes:
        for t in range(0, max_t): # mouv = [0, max_t - 1], |mouv| = max_t
            # mouv (en t) -> rive(sommet) = rive(berger) (en t)
            cnf.append([-mouv(s, t), pos(s, t), -pos(0, t)])
            cnf.append([-mouv(s, t), -pos(s, t), pos(0, t)])
            # mouv (en t) -> rive(sommet) = rive(berger) (en t + 1)
            cnf.append([-mouv(s, t), pos(s, t + 1), -pos(0, t + 1)])
            cnf.append([-mouv(s, t), -pos(s, t + 1), pos(0, t + 1)])
            ### IL MANQUAIT PEUT ETRE UNE CONTRAINTE SUR LE FAIT QU UN MOUVEMENT SOIT VRAI QUAND UN SOMMET EST DEPLACE
            cnf.append([-mouv(s, t), -pos(s, t), -pos(s, t + 1)])
            cnf.append([-mouv(s, t), pos(s, t), pos(s, t + 1)])
            cnf.append([mouv(s, t), -pos(s, t), pos(s, t + 1)])
            cnf.append([mouv(s, t), pos(s, t), -pos(s, t + 1)])

    print("Contraintes sur le nombre k de mouvements")
    for t in range(0, max_t): # mouv = [0, max_t - 1], |mouv| = max_t
        cnf.extend(CardEnc.atmost(lits=[mouv(s, t) for s in G.nodes], bound=k, vpool=vpool).clauses) # choisi au plus k sommets à déplacer à chaque t

# Vu les print de clauses et de solution, je pense que mtn on a un probleme sur le respect de "transporter max k sommet à la fois"
# (1, 3 et 4 sont tansportés au temps 1, tous en meme temps)
# J'ai l'impression que au moins il essaye de respecter la contrainte sur les conflits (sauf qu'il le fait en cassant la contrainte sur k)

    print("Clauses construites:\n")
    print(cnf.clauses)  # pour afficher les clauses
    print(cnf.atmosts)

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
gen_solution(g,1) # Should return a solution

# h = nx.Graph()
# h.add_nodes_from([1, 2, 3, 4])
# h.add_edges_from([(2, 1), (2, 3), (2, 4)])
# gen_solution(h, 1) # Should return None
