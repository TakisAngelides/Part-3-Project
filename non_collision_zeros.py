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

def sym_2_Gram_det(a,b):
    return Gram_det_2(a,b,a,b)

def logic_statement_true_for_non_chiral(S, E):

    a = multivec_to_vec(S[0])
    a = np.insert(a, 0, E[0])
    b = multivec_to_vec(S[1])
    b = np.insert(b, 0, E[1])
    c = multivec_to_vec(S[2])
    c = np.insert(c, 0, E[2])
    d = multivec_to_vec(S[3])
    d = np.insert(d, 0, E[3])
    RF = a+b+c+d

    case_1 = (Gram_det_2(a,RF,a,RF) == approx(0))

    case_2 = (dot(a+b+c+d, a+b+c+d) == approx(0))

    case_3 = (epsilon(b,c,d,RF) == approx(0))

    case_4 = ((dot(b,b) == dot(c,c)) and (Gram_det_2(b-c,RF,a,RF) == 0) and (sym_2_Gram_det(b,RF) == sym_2_Gram_det(c,RF)))

    case_5 = ((dot(c,c) == dot(d,d)) and (Gram_det_2(c-d,RF,a,RF) == 0) and (sym_2_Gram_det(c,RF) == sym_2_Gram_det(d,RF)))

    case_6 = ((dot(d,d) == dot(b,b)) and (Gram_det_2(d-b,RF,a,RF) == 0) and (sym_2_Gram_det(d,RF) == sym_2_Gram_det(b,RF)))

    return (case_1 or case_2 or case_3 or case_4 or case_5 or case_6)

def permute_with_idx(M, E, idx_to_permute):

    same_mass_with_idx = [idx for idx in range(len(M)) if M[idx] == M[idx_to_permute] and idx != idx_to_permute and idx != 0 and idx != 1]
    same_energy_with_idx = [idx for idx in range(len(E)) if E[idx] == approx(E[idx_to_permute]) and idx != idx_to_permute]

    return list(set(same_mass_with_idx) and set(same_energy_with_idx))

def permutation_boolean(M, E, idx_1, idx_2):

    if (M[idx_1] == M[idx_2]) and (E[idx_1] == E[idx_2]):
        return True
    else:
        return False

def construct_state():

    rdm = randint(0, 2)

    if rdm == 0:
        ma, mb, mc, md = randint(0, 10), randint(0, 10), randint(0, 10), randint(0, 10)
        a = uniform(-10, 10) * e3
        b = uniform(-10, 10) * e1 + uniform(-10, 10) * e2 + uniform(-10, 10) * e3
        c = uniform(-10, 10) * e1 + uniform(-10, 10) * e2 + uniform(-10, 10) * e3
        d = -a-b-c

    if rdm == 1:
        # This is the no permutation non-chiral case
        ma, mb, mc, md = randint(0, 10), randint(0, 10), randint(0, 10), randint(0, 10)
        a = uniform(-10, 10) * e3
        b = e1 + e2 + uniform(-10, 10)*e3
        c = -e1 - e2 + uniform(-10, 10)*e3
        d = -a-b-c

    if rdm == 2:
        # Permute 2 non chiral case
        ma, mb, mc, md = 1, 1, 1, 1
        a = uniform(-10, 10) * e3
        angle = uniform(0, pi / 5)
        b = sin(angle)*e1 + cos(angle)*e2 + uniform(-10, 10)*e3
        c = cos(angle)*e1 + sin(angle)*e2 + b[3]*e3
        d = -a-b-c

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

    permute_with_b = permute_with_idx(M, E, 1)
    permute_with_c = permute_with_idx(M, E, 2)

    S_parity = parity(S)  # Perform parity on the set of momenta

    flag = False  # The flag is set to true if the state is non-chiral

    R1 = e1 * e3
    S1 = rotate(S_parity, R1)  # Now a is fixed back to their original state

    if (b[1] != 0) or (b[2] != 0): # if b has components in the 1-2 plane

        for idx in permute_with_b + [1]:

            if idx == 0:
                continue

            x = S1[idx]
            x_12 = x - x[3]*e3
            b_12 = b - b[3]*e3

            n = b_12 + x_12

            if n == 0:

                R2 = e1*e2

            else:

                R2 = b_12.normal()*n.normal()

            S2 = rotate(S1, R2)
            S3 = swap(S2, 1, idx) # Index 1 corresponds to b

            if S == S3:
                flag = True

            if permutation_boolean(M, E, 2, 3): # If we can permute (cd)
                S4 = swap(S3, 2, 3)
                if S == S4:
                    flag = True

    elif (c[1] != 0) or (c[2] != 0):

        for idx in permute_with_c + [2]:

            if idx == 0 or idx == 1:
                continue

            x = S1[idx]
            x_12 = x - x[3] * e3
            c_12 = c - c[3] * e3

            n = c_12 + x_12

            if n == 0:

                R2 = e1 * e2

            else:

                R2 = c_12.normal() * n.normal()

            S2 = rotate(S1, R2)
            S3 = swap(S2, 2, idx) # Index 2 corresponds to c

            if S == S3:
                flag = True

    if flag:
        non_chiral_states.append([S, E])
    else:
        chiral_states.append([S, E])

    return non_chiral_states, chiral_states

# non_chiral_states_list = []
# chiral_states_list = []
# for iterations in range(1000):
#     S, E, M = construct_state()
#     non_chiral_states, chiral_states = chirality_test()
#     non_chiral_states_list += non_chiral_states
#     chiral_states_list += chiral_states
#
# print(len(non_chiral_states_list), len(chiral_states_list))
#
# non_chiral_evaluation_on_logic_statement = [0, 0]
# for non_chiral_state in non_chiral_states_list:
#     mp, mq = uniform(1, 10), uniform(1, 10)
#     flag = logic_statement_true_for_non_chiral(non_chiral_state[0], non_chiral_state[1])
#     if flag:
#         non_chiral_evaluation_on_logic_statement[0] += 1
#     else:
#         non_chiral_evaluation_on_logic_statement[1] += 1
#
# chiral_evaluation_on_logic_statement = [0, 0]
# for chiral_state in chiral_states_list:
#     mp, mq = uniform(1, 10), uniform(1, 10)
#     flag = logic_statement_true_for_non_chiral(chiral_state[0], chiral_state[1])
#     if flag:
#         chiral_evaluation_on_logic_statement[0] += 1
#     else:
#         chiral_evaluation_on_logic_statement[1] += 1
#
# x = ['True', 'False']
# height_non_chiral = [non_chiral_evaluation_on_logic_statement[0], non_chiral_evaluation_on_logic_statement[1]]
# height_chiral = [chiral_evaluation_on_logic_statement[0], chiral_evaluation_on_logic_statement[1]]
#
# plt.bar(x, height_non_chiral, color = 'k', width = 0.1)
# plt.title('Non-chiral states evaluated on the logic statement\nwhich is true iff the input is non-chiral')
# plt.ylabel('Frequency')
# #plt.show()
# plt.savefig('non_chiral_non_collision_logic_statement_test.pdf', bbox_inches='tight')
#
# plt.bar(x, height_chiral, color = 'k', width = 0.1)
# plt.title('Chiral states evaluated on the logic statement\nwhich is true iff the input is non-chiral')
# plt.ylabel('Frequency')
# plt.show()
# plt.savefig('chiral_non_collision_logic_statement_test.pdf', bbox_inches='tight')

print(chirality_test())














