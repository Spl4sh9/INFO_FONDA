import networkx as nx

from pysat.solvers import Minicard
from pysat.formula import CNFPlus, IDPool
from pysat.card import CardEnc # Pour les contraintes de cardinalité


PRINT_Q2_CLAUSIS = False
PRINT_Q2_SOL = False
PRINT_Q5_CLAUSIS = False
PRINT_Q5_SOL = False

# Q2
def gen_solution(G: nx.Graph, k: int) -> list[tuple[int, set, set]]:
    if k == 0: # Enigme impossible
        return None

    max_t = 2 * len(G) + 1  # Theoreme 1 : taille de la plus longue séq. pour Alcuin(G) <= 2n + 1
    vpool = IDPool(start_from=1)  # stockage des variables (s, t, 'pos') pour position et (s, t, 'mov') pour mouvement
    cnf = CNFPlus()

    # Permet plus de clarté dans le code
    def pos(s, t): 
        return vpool.id((s, t, 'pos'))
    def mouv(s, t):
        return vpool.id((s, t, 'mov'))

    if PRINT_Q2_CLAUSIS: print("Construction des clauses...")
    if PRINT_Q2_CLAUSIS: print("Contrainte d'alternance du berger")
    for t in range(0, max_t):
        cnf.append([pos(-1, t), pos(-1, t + 1)])
        cnf.append([-pos(-1, t), -pos(-1, t + 1)])
    cnf.append([-pos(-1, 0)])  # Impose que le berger commence à gauche
    cnf.append([pos(-1, max_t)])  # Impose que le berger finisse à droite

    if PRINT_Q2_CLAUSIS: print("Tous les sommets commencent à 0 (gauche)")
    for s in G.nodes:
        cnf.append([-pos(s, 0)])

    if PRINT_Q2_CLAUSIS: print("Tous les sommets finissent à 1 (droite)")
    for s in G.nodes:
        cnf.append([pos(s, max_t)])

    if PRINT_Q2_CLAUSIS: print("Contraintes sur les conflits entre sommets")
    for (v, w) in G.edges:
        for t in range(0, max_t + 1): # Verifie pour toutes les configurations => [0, max_t]
            cnf.append([pos(-1, t), -pos(v, t), -pos(w, t)])
            cnf.append([-pos(-1, t), pos(v, t), pos(w, t)])

    if PRINT_Q2_CLAUSIS: print("Contraintes sur la cohérence des mouvements")
    for s in G.nodes:
        for t in range(0, max_t): # mouvement se produisent dans l'intervalle [0, max_t - 1] (pas de mouvement à t = max_t)
            # mouv(s, t) -> [pos(s, t) = pos(berger, t)]
            cnf.append([-mouv(s, t), pos(s, t), -pos(-1, t)])
            cnf.append([-mouv(s, t), -pos(s, t), pos(-1, t)])
            # mouv(s, t) -> [pos(s, t+1) = pos(berger, t+1)]
            cnf.append([-mouv(s, t), pos(s, t + 1), -pos(-1, t + 1)])
            cnf.append([-mouv(s, t), -pos(s, t + 1), pos(-1, t + 1)])
            # [pos(s, t) != pos(s, t + 1)] -> mouv(s, t)
            cnf.append([-mouv(s, t), -pos(s, t), -pos(s, t + 1)])
            cnf.append([-mouv(s, t), pos(s, t), pos(s, t + 1)])
            cnf.append([mouv(s, t), -pos(s, t), pos(s, t + 1)])
            cnf.append([mouv(s, t), pos(s, t), -pos(s, t + 1)])

    if PRINT_Q2_CLAUSIS: print("Contraintes sur le nombre k de mouvements")
    for t in range(0, max_t): # mouvement se produisent dans l'intervalle [0, max_t - 1] (pas de mouvement à t = max_t)
        cnf.extend(CardEnc.atmost(lits=[mouv(s, t) for s in G.nodes], bound=k, vpool=vpool).clauses) # choisi au plus k sommets à déplacer

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
        for t in range(max_t + 1): # Récupère les positions de chaque sommet à chaque t
            for s in G.nodes:
                positions[(s, t)] = model[vpool.id((s, t, 'pos')) - 1] > 0 # - = 0, + = 1

        solution = []
        for t in range(max_t + 1):
            right, left = set(), set()
            b_t = t % 2 # Beger à gauche si pair, à droite si impair
            for s in G.nodes: # Ajoute les sommets à gauche (=0) ou à droite (=1)
                if positions[(s, t)]:
                    right.add(s)
                else:
                    left.add(s)
            solution.append((b_t, left, right)) # Ajoute la ieme configuration
        return solution

    return None

# Q3
def find_alcuin_number(G: nx.Graph) -> int:
    max_k, min_k = len(G), 1 # max = |S| est toujours une solution

    while min_k < max_k: # Recherche dichotomique
        k = (min_k + max_k) // 2 
        if gen_solution(G, k) is None:  # k n'est pas la solution
            min_k = k + 1 # sol > k
        else: # k est une solution
            max_k = k # sol <= k

    return min_k

# Q5
def gen_solution_cvalid(G: nx.Graph, k: int, c_: int) -> list[tuple[int, set, set, tuple[set]]]:
    """
    Reprend le code de la question #2, ajoute les nouvelles contraintes et adapte la sortie
    """
    if k == 0: # Enigme impossible
        return None

    max_t = 2 * len(G) + 1  # Theoreme 1 : taille de la plus longue séq. pour Alcuin(G) <= 2n + 1
    vpool = IDPool(start_from=1)  # stockage des littéraux (s, t, 'pos') pour position, (s, t, 'mov) pour mouvement et (s, t, c) pour compartiment
    cnf = CNFPlus()

    # Permet plus de clarté dans le code
    def pos(s, t):
        return vpool.id((s, t, 'pos'))
    def mouv(s, t):
        return vpool.id((s, t, 'mov'))
    def comp(s, t, c):
        return vpool.id((s, t+1, c)) # Les compartiments sont décalés de 1 dans le temps par rapport aux mouvements
    if PRINT_Q2_CLAUSIS: print("Construction des clauses...")
    if PRINT_Q2_CLAUSIS: print("Contrainte d'alternance du berger")
    for t in range(0, max_t):
        cnf.append([pos(-1, t), pos(-1, t + 1)])
        cnf.append([-pos(-1, t), -pos(-1, t + 1)])
    cnf.append([-pos(-1, 0)])  # Impose que le berger commence à gauche
    cnf.append([pos(-1, max_t)])  # Impose que le berger finisse à droite

    if PRINT_Q2_CLAUSIS: print("Tous les sommets commencent à 0 (gauche)")
    for s in G.nodes:
        cnf.append([-pos(s, 0)])

    if PRINT_Q2_CLAUSIS: print("Tous les sommets finissent à 1 (droite)")
    for s in G.nodes:
        cnf.append([pos(s, max_t)])

    if PRINT_Q2_CLAUSIS: print("Contraintes sur les conflits entre sommets")
    for (v, w) in G.edges:
        for t in range(0, max_t + 1): # Verifie pour toutes les configurations => [0, max_t]
            cnf.append([pos(-1, t), -pos(v, t), -pos(w, t)])
            cnf.append([-pos(-1, t), pos(v, t), pos(w, t)])

    if PRINT_Q2_CLAUSIS: print("Contraintes sur la cohérence des mouvements")
    for s in G.nodes:
        for t in range(0, max_t): # mouvement se produisent dans l'intervalle [0, max_t - 1] (pas de mouvement à t = max_t)
            # mouv(s, t) -> [pos(s, t) = pos(berger, t)]
            cnf.append([-mouv(s, t), pos(s, t), -pos(-1, t)])
            cnf.append([-mouv(s, t), -pos(s, t), pos(-1, t)])
            # mouv(s, t) -> [pos(s, t+1) = pos(berger, t+1)]
            cnf.append([-mouv(s, t), pos(s, t + 1), -pos(-1, t + 1)])
            cnf.append([-mouv(s, t), -pos(s, t + 1), pos(-1, t + 1)])
            # [pos(s, t) != pos(s, t + 1)] -> mouv(s, t)
            cnf.append([-mouv(s, t), -pos(s, t), -pos(s, t + 1)])
            cnf.append([-mouv(s, t), pos(s, t), pos(s, t + 1)])
            cnf.append([mouv(s, t), -pos(s, t), pos(s, t + 1)])
            cnf.append([mouv(s, t), pos(s, t), -pos(s, t + 1)])

    if PRINT_Q2_CLAUSIS: print("Contraintes sur le nombre k de mouvements")
    for t in range(0, max_t): # mouvement se produisent dans l'intervalle [0, max_t - 1] (pas de mouvement à t = max_t)
        cnf.extend(CardEnc.atmost(lits=[mouv(s, t) for s in G.nodes], bound=k, vpool=vpool).clauses) # choisi au plus k sommets à déplacer

    if PRINT_Q5_CLAUSIS: print("Contraintes sur les compartiments en fonction des mouvements")
    for s in G.nodes: 
        for t in range(0, max_t):
            # mouv(s, t) -> [comp(s, t+1, 1) v comp(s, t+1, 2) v ... v comp(s, t+1, c_)]
            clause_1 = [-mouv(s, t)]
            for c in range(1, c_+1): # Crée la clause [comp(s, t+1, 1) v comp(s, t+1, 2) v ... v comp(s, t+1, c_)]
                clause_1.append(comp(s, t, c))
            cnf.append(clause_1) # Si mouv(s, t) alors au moins un comp(s, t, c) est vrai

            # [comp(s, t, 1) v comp(s, t, 2) v ... v comp(s, t, c_)] -> mouv(s, t)
            for c in range(1, c_+1):
                cnf.append([mouv(s, t), -comp(s, t, c)]) # Si au moins un comp(s, t, c) est vrai alors mouv(s, t) est vrai

    if PRINT_Q5_CLAUSIS : print("Contraintes sur l'unicité des compartiments par sommet")
    for t in range(0, max_t):
        for s in G.nodes:
            cnf.extend(CardEnc.atmost(lits=[comp(s, t, c) for c in range(1, c_+1)], bound=1, vpool=vpool).clauses) # au plus un comp(s, t, c) est vrai à la fois

    if PRINT_Q5_CLAUSIS: print("Contraintes sur les conflits dans les compartiments")
    for (v,w) in G.edges:
        for t in range(0, max_t):
            for c in range(1, c_+1):
                cnf.append([-comp(v, t, c), -comp(w, t, c)]) # Au moins un des sommets en conflit ne se trouve pas dans le compartiment

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

        positions = {} # Récupère les positions de chaque sommet à chaque t
        for t in range(max_t + 1):
            for s in G.nodes:
                positions[(s, t)] = model[vpool.id((s, t, 'pos')) - 1] > 0

        solution = []
        for t in range(max_t + 1):
            right, left = set(), set()

            comp_t = [] # Récupère chaque compartiment pour le temps t
            if t > 0:
                for c in range(1, c_+1):
                    comp_c = set() # Construit le compartiment c au temps t en ajoutant les sommets s
                    for s in G.nodes:
                        if model[vpool.id((s, t, c)) - 1] > 0:
                            comp_c.add(s) 
                    comp_t.append(comp_c)

            b_t = t % 2 # Berger alterne entre gauche (0) et droite (1)
            for s in G.nodes: # Ajoute les sommets à gauche (=0) ou à droite (=1)
                if positions[(s, t)]:
                    right.add(s)
                else:
                    left.add(s)
            solution.append((b_t, left, right, tuple(comp_t))) # Ajoute la ieme configuration
        return solution

    return None

# Q6
def find_c_alcuin_number(G: nx.Graph, c: int) -> int:
    max_k, min_k = len(G), 1 # Commence avec max_k = |S| car au-delà on considère que c'est impossible (inf)
    # transporter plus de sujet qu'il n'en existe n'a pas de sens

    while min_k < max_k: # Recherche dichotomique
        k = (min_k + max_k) // 2 
        if gen_solution_cvalid(G, k, c) is None:  # k n'est pas la solution
            min_k = k + 1 # sol > k
        else: # k est une solution
            max_k = k # sol <= k

    if gen_solution_cvalid(G, min_k, c) is None: # min_k a dépassé max_k mais n'est pas la solution
        return float('+inf') # Il n'y a tout simplement pas de solution

    return min_k # min_k est la solution

