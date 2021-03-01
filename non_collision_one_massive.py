import time
from numpy import pi, cos, sin, e, tan, arctan
from clifford.g3 import blades
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from random import uniform, seed, randint
from sympy import LeviCivita as eps
from main import parity, rotate, energy, epsilon, Gram_det_2
from pytest import approx

# Works only for a,b,c,d != 0 (it is very unlikely that any one will be randomly generated in momentum 0)

e1, e2, e3 = blades['e1'], blades['e2'], blades['e3']
I = e1^e2^e3

def dot(a,b):
    dot = a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
    return dot

def multivec_to_vec(a):
    return np.array([a[1], a[2], a[3]])

def swap(S,idx_1,idx_2):

    tmp = S[idx_1]
    S[idx_1] = S[idx_2]
    S[idx_2] = tmp

    return S

def logic_statement_true_for_non_chiral(S, E, mp, mq):

    p = np.array([mp, 0, 0, 0])
    q = np.array([mq, 0, 0, 0])
    a = multivec_to_vec(S[0])
    a = np.insert(a, 0, E[0])
    b = multivec_to_vec(S[1])
    b = np.insert(b, 0, E[1])
    c = multivec_to_vec(S[2])
    c = np.insert(c, 0, E[2])
    d = multivec_to_vec(S[3])
    d = np.insert(d, 0, E[3])
    RF = p + q

    case_1 = ((epsilon(a, b, c, RF) == approx(0)) and (epsilon(a, b, d, RF) == approx(0)) and (epsilon(b, c, d, RF) == approx(0)) and (
                epsilon(a, c, d, RF) == approx(0)))

    case_2 = ((dot(a,a) == approx(dot(b,b))) and (not (a == b).all()) and (Gram_det_2(a,RF,a,RF) == approx(Gram_det_2(b,RF,b,RF))) and (Gram_det_2(a,RF,c,RF) == approx((Gram_det_2(b,RF,c,RF)))) and (Gram_det_2(a,RF,d,RF) == approx(Gram_det_2(b,RF,d,RF))))

    def case_2_symmetry(a,b,c,d):
        return ((dot(a,a) == approx(dot(b,b))) and (not (a == b).all()) and (Gram_det_2(a,RF,a,RF) == approx(Gram_det_2(b,RF,b,RF))) and (Gram_det_2(a,RF,c,RF) == approx((Gram_det_2(b,RF,c,RF) == 0))) and (Gram_det_2(a,RF,d,RF) == approx(Gram_det_2(b,RF,d,RF))))

    case_3 = case_2_symmetry(a,c,b,d)
    case_4 = case_2_symmetry(a,d,c,b)
    case_5 = case_2_symmetry(b,c,a,d)
    case_6 = case_2_symmetry(b,d,c,a)
    case_7 = case_2_symmetry(c,d,a,b)

    case_8 = ((dot(a,a) == approx(dot(b,b))) and (dot(c,c) == approx(dot(d,d))) and (Gram_det_2(a,RF,a,RF) == approx(Gram_det_2(b,RF,b,RF))) and (Gram_det_2(c,RF,c,RF) == approx(Gram_det_2(d,RF,d,RF))) and ((((dot(a+b,a+b) == approx(4*dot(a,a)) == approx(4*dot(b,b)))) and ((dot(c+d,c+d) == approx(4*dot(c,c)) == approx(4*dot(d,d))))) or ((Gram_det_2(a+b,RF,c-d,RF) == approx(0)) or (not (c == d).all()))))

    def case_8_symmetry(a,b,c,d):
        return ((dot(a,a) == approx(dot(b,b))) and (dot(c,c) == approx(dot(d,d))) and (Gram_det_2(a,RF,a,RF) == approx(Gram_det_2(b,RF,b,RF))) and (Gram_det_2(c,RF,c,RF) == approx(Gram_det_2(d,RF,d,RF))) and ((((dot(a+b,a+b) == approx(4*dot(a,a)) == approx(4*dot(b,b)))) and ((dot(c+d,c+d) == approx(4*dot(c,c)) == approx(4*dot(d,d))))) or ((Gram_det_2(a+b,RF,c-d,RF) == approx(0)) or (not (c == d).all()))))

    case_9 = case_8_symmetry(a,c,b,d)
    case_10 = case_8_symmetry(a,d,b,c)

    case_19 = ((dot(a,a) == approx(dot(b,b)) == approx(dot(c,c)) == approx(dot(d,d))) and (Gram_det_2(a,RF,a,RF) == approx(Gram_det_2(b,RF,b,RF)) == approx(Gram_det_2(c,RF,c,RF)) == approx(Gram_det_2(d,RF,d,RF))) and approx((Gram_det_2(a - c,RF,b - d,RF) == 0)))

    def case_19_symmetry(a,b,c,d):
        return ((dot(a,a) == approx(dot(b,b)) == approx(dot(c,c)) == approx(dot(d,d))) and (Gram_det_2(a,RF,a,RF) == approx(Gram_det_2(b,RF,b,RF)) == approx(Gram_det_2(c,RF,c,RF)) == approx(Gram_det_2(d,RF,d,RF))) and approx((Gram_det_2(a - c,RF,b - d,RF) == 0)))

    case_20 = case_19_symmetry(a,b,d,c)
    case_21 = case_19_symmetry(a,c,b,d)
    case_22 = case_19_symmetry(a,c,d,b)
    case_23 = case_19_symmetry(a,d,c,b)
    case_24 = case_19_symmetry(a,d,b,c)

    return (case_1 or case_2 or case_3 or case_4 or case_5 or case_6 or case_7 or case_8 or case_9 or case_10
            or case_19 or case_20 or case_21 or case_22 or case_23 or case_24)

def construct_state():

    B = (uniform(-10, 10) * (e1 ^ e2) + uniform(-10, 10) * (e1 ^ e3) + uniform(-10, 10) * (e2 ^ e3)).normal()
    R = e ^ (uniform(0, 2 * pi) * B)

    rdm = randint(0, 0)

    if rdm == 0:
        ma, mb, mc, md = randint(0, 10), randint(0, 10), randint(0, 10), randint(0, 10)
        a = uniform(-10, 10) * e1 + uniform(-10, 10) * e2 + uniform(-10, 10) * e3
        b = uniform(-10, 10) * e1 + uniform(-10, 10) * e2 + uniform(-10, 10) * e3
        c = uniform(-10, 10) * e1 + uniform(-10, 10) * e2 + uniform(-10, 10) * e3
        d = uniform(-10, 10) * e1 + uniform(-10, 10) * e2 + uniform(-10, 10) * e3

    # Type of non-chiral (checked) (ab)
    if rdm == 1:
        ma, mb, mc, md = 1, 1, 2, 3
        a = e1
        b = e2
        c = e3
        d = (-e3)

    # Type of non-chiral (checked) (ab)(cd)
    if rdm == 2:
        ma, mb, mc, md = 1, 1, 2, 2
        n = uniform(-10, 10)*e1 + uniform(-10, 10)*e2 + uniform(-10, 10)*e3
        n = n.normal()
        a = uniform(-10, 10)*e1 + uniform(-10, 10)*e2 + uniform(-10, 10)*e3
        b = -n*a*n
        c = uniform(-10, 10)*e1 + uniform(-10, 10)*e2 + uniform(-10, 10)*e3
        d = -n*c*n

    # Type of non-chiral (checked) (abc)
    if rdm == 3:
        ma, mb, mc, md = 1, 1, 1, 3
        mag = uniform(-10, 10)
        a = mag*e1 + mag*e2
        b = mag*e1 - mag*e2
        c = -mag*e1 + mag*e2
        d = mag*e1 - mag*e2

    # Type of non-chiral (checked) (abcd)
    if rdm == 4:
        ma, mb, mc, md = 1, 1, 1, 1
        a = R*(-e1 + e3)*~R
        b = R*(e1 + e2)*~R
        c = R*(-e1 - e3)*~R
        d = R*(e1 - e2)*~R

    Ea, Eb, Ec, Ed = energy(ma, a), energy(mb, b), energy(mc, c), energy(md, d)

    M = [ma, mb, mc, md]
    E = [Ea, Eb, Ec, Ed]
    S = [a, b, c, d]

    return S, E, M

def chirality_test():

    chiral_states = []
    non_chiral_states = []

    S, E, M = construct_state()
    a, b, c, d = S[0], S[1], S[2], S[3]

    same_mass_with_a = [idx for idx in range(len(M)) if M[idx] == M[0] and idx != 0]
    same_energy_with_a = [idx for idx in range(len(E)) if E[idx] == approx(E[0]) and idx != 0]
    # Holds the indices of particles that can be permuted with a
    permute_with_a = list(set(same_mass_with_a) and set(same_energy_with_a))
    len_a = len(permute_with_a)

    same_mass_with_b = [idx for idx in range(len(M)) if M[idx] == M[1] and idx != 1]
    same_energy_with_b = [idx for idx in range(len(E)) if E[idx] == approx(E[1]) and idx != 1]
    # Holds the indices of particles that can be permuted with b
    permute_with_b = list(set(same_mass_with_b) and set(same_energy_with_b))
    len_b = len(permute_with_b)

    same_mass_with_c = [idx for idx in range(len(M)) if M[idx] == M[2] and idx != 2]
    same_energy_with_c = [idx for idx in range(len(E)) if E[idx] == approx(E[2]) and idx != 2]
    # Holds the indices of particles that can be permuted with c
    permute_with_c = list(set(same_mass_with_c) and set(same_energy_with_c))
    len_c = len(permute_with_c)

    same_mass_with_d = [idx for idx in range(len(M)) if M[idx] == M[3] and idx != 3]
    same_energy_with_d = [idx for idx in range(len(E)) if E[idx] == approx(E[3]) and idx != 3]
    # Holds the indices of particles that can be permuted with d
    permute_with_d = list(set(same_mass_with_d) and set(same_energy_with_d))
    len_d = len(permute_with_d)

    len_list = [len_a, len_b, len_c, len_d]

    len_max = max(len_list)

    flag = False # This is set to true when the state is non-chiral
    if len_max == 0: # If there are no permutations
        S1 = parity(S)
        if a^b != 0:
            R1 = (a^b).normal()
            S2 = rotate(S1, R1)
        elif a^c != 0:
            R1 = (a^c).normal()
            S2 = rotate(S1, R1)
        else:
            flag = True # If b and c are colinear then the state is non-chiral
        if flag or S == S2:
            non_chiral_states.append([S,E])
        else:
            chiral_states.append([S,E])

    else:

        if (np.sum(len_list) == 2): # this is the case when only 2 can be permuted (ab) (for example)

            # map the ones who cant be permuted and then check or swap permutable and check
            idx_perm = [idx for idx in range(len(len_list)) if len_list[idx] != 0] # permutable indices in S
            idx_non_perm = [idx for idx in range(len(len_list)) if len_list[idx] == 0] # non-permutable indices in S

            x = S[idx_non_perm[0]] # These 2 are the particles that cant be permuted and we map them first
            y = S[idx_non_perm[1]]

            z = S[idx_perm[0]] # These 2 are the particles that can be permuted
            q = S[idx_perm[1]]

            S1 = parity(S)
            if x^y != 0: # If x and y are not colinear
                R1 = (x^y).normal()
                S2 = rotate(S1, R1)
                if S == S2 or S == swap(S2, idx_perm[0], idx_perm[1]):
                    non_chiral_states.append([S, E])
                else:
                    chiral_states.append([S, E])
            else: # If x and y are colinear then use x and z for first rotation
                if x^z != 0:
                    R1 = (x^z).normal()
                    S2 = rotate(S1, R1)
                    if dot(x,q) == 0 and dot(x,z) == 0: # If z,q are in the plane perpendicular to colinear mapped x,y
                        n = (q+S2[idx_perm[0]]).normal() # take z_rotated which has index idx_perm[0] to q and swap
                        R2 = q.normal()*n
                        S3 = rotate(S2, R2)
                        S4 = swap(S3, idx_perm[0], idx_perm[1])
                        if S == S4:
                            non_chiral_states.append([S, E])
                        else:
                            chiral_states.append([S, E])
                    else:
                        if S == S2 or S == swap(S2, idx_perm[0], idx_perm[1]):
                            non_chiral_states.append([S, E])
                        else:
                            chiral_states.append([S, E])
                else: # Here I have 3 colinear vectors so state is non-chiral
                    non_chiral_states.append([S, E])

        elif (np.sum(len_list) == 4): # This is the case of permutations of the type (ab)(cd)

            # Start with a and find the particle x it can be permuted with, map them both to their selves. now the only
            # rotation we have available is pi in the plane spanned by (a cross x) = -I(a^x) (2.80 Lasenby) and a - x

            a = S[0]
            x = S[permute_with_a[0]]

            S1 = parity(S)
            R1 = (a^x).normal() # Note this will not work if a, x are explicitly colinear but this is very unlikely
            S2 = rotate(S1, R1)
            R2 = ((-I*(a^x))^(a-x)).normal() # Rotation by pi in the plane spanned by (a cross x) = -I(a^x) and a - x

            S3 = rotate(S2, R2)
            S4 = swap(S3, 0, permute_with_a[0]) # this includes the extra pi rotation I can make because of (ax)

            diff = 6-permute_with_a[0] # 6 = 0 + 1 + 2 + 3 -> the indices of the particles in S
            if diff == 5:
                idx_1, idx_2 = 2, 3 # the other two other than a,x are c (S[2]) and d (S[3])
            elif diff == 4:
                idx_1, idx_2 = 1, 3
            else:
                idx_1, idx_2 = 1, 2

            if S == S2 or S == swap(S2, idx_1, idx_2) or S == S4 or S == swap(S4, idx_1, idx_2):
                non_chiral_states.append([S, E])
            else:
                chiral_states.append([S, E])

        elif (np.sum(len_list) == 6): # this is the case of permutations of the type (abc)

            # find the particle x that cant be permuted and map to itself along with any other of y,z,q then we can
            # check or swap the other 2 and check

            idx_non_perm = [idx for idx in range(len(len_list)) if len_list[idx] == 0]
            idx_list = [0,1,2,3]
            idx_list.remove(idx_non_perm[0])

            x = S[idx_non_perm[0]]
            y = S[idx_list[0]]
            idx_1, idx_2 = idx_list[1], idx_list[2]

            S1 = parity(S)
            R1 = (x^y).normal() # Note this will not work if x, y are explicitly colinear but this is very unlikely

            S2 = rotate(S1, R1)

            if S == S2 or S == swap(S2, idx_1, idx_2):
                non_chiral_states.append([S, E])
            else:
                chiral_states.append([S, E])

        else: # this is the (abcd) case of permutations

            # map a and b then check or swap cd and check or rotate by pi in plane perpendicular to a+b swap ab and
            # check or swap cd and check

            S1 = parity(S)
            # TODO: catch problem of a b being colinear see * below
            R1 = (a^b).normal()
            S2 = rotate(S1, R1) # check

            S3 = swap(S, 2, 3) # check # TODO: this should change if a b are colinear see * above

            R2 = ((-I*(a^b))^(a-b)).normal() # Rotation by pi in the plane spanned by (a cross b) = -I(a^b) and a - b
            S4 = rotate(S3, R2)
            S5 = swap(S4, 0, 1) # check # TODO: this should change if a b are colinear see * above
            S6 = swap(S5, 2, 3) # check

            if S == S2 or S == S3 or S == S5 or S == S6:
                non_chiral_states.append([S, E])
            else:
                chiral_states.append([S, E])

    return non_chiral_states, chiral_states

non_chiral_states_list = []
chiral_states_list = []
for iterations in range(1000):
    S, E, M = construct_state()
    non_chiral_states, chiral_states = chirality_test()
    non_chiral_states_list += non_chiral_states
    chiral_states_list += chiral_states

print(len(non_chiral_states_list), len(chiral_states_list))

non_chiral_evaluation_on_logic_statement = [0, 0]
for non_chiral_state in non_chiral_states_list:
    mp, mq = uniform(1, 10), uniform(1, 10)
    flag = logic_statement_true_for_non_chiral(non_chiral_state[0], non_chiral_state[1], mp, mq)
    if flag:
        non_chiral_evaluation_on_logic_statement[0] += 1
    else:
        non_chiral_evaluation_on_logic_statement[1] += 1

chiral_evaluation_on_logic_statement = [0, 0]
for chiral_state in chiral_states_list:
    mp, mq = uniform(1, 10), uniform(1, 10)
    flag = logic_statement_true_for_non_chiral(chiral_state[0], chiral_state[1], mp, mq)
    if flag:
        chiral_evaluation_on_logic_statement[0] += 1
    else:
        chiral_evaluation_on_logic_statement[1] += 1

x = ['True', 'False']
height_non_chiral = [non_chiral_evaluation_on_logic_statement[0], non_chiral_evaluation_on_logic_statement[1]]
height_chiral = [chiral_evaluation_on_logic_statement[0], chiral_evaluation_on_logic_statement[1]]

plt.bar(x, height_non_chiral, color = 'k', width = 0.1)
plt.title('Non-chiral states evaluated on the logic statement\nwhich is true iff the input is non-chiral')
plt.ylabel('Frequency')
#plt.show()
#plt.savefig('non_chiral_non_collision_logic_statement_test.pdf', bbox_inches='tight')

plt.bar(x, height_chiral, color = 'k', width = 0.1)
plt.title('Chiral states evaluated on the logic statement\nwhich is true iff the input is non-chiral')
plt.ylabel('Frequency')
#plt.show()
plt.savefig('chiral_non_collision_logic_statement_test.pdf', bbox_inches='tight')

















