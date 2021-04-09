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

    p = multivec_to_vec(S[0])
    p = np.insert(p, 0, E[0])
    q = multivec_to_vec(S[1])
    q = np.insert(q, 0, E[1])
    a = multivec_to_vec(S[2])
    a = np.insert(a, 0, E[2])
    b = multivec_to_vec(S[3])
    b = np.insert(b, 0, E[3])
    c = multivec_to_vec(S[4])
    c = np.insert(c, 0, E[4])
    d = multivec_to_vec(S[5])
    d = np.insert(d, 0, E[5])
    RF = a + b + c + d

    case_1 = (epsilon(a,b,p+q,RF) == epsilon(a,c,p+q,RF) == epsilon(a,d,p+q,RF) == epsilon(b,c,p+q,RF) == epsilon(b,d,p+q,RF) == epsilon(c,d,p+q,RF) == 0)

    case_2 = ((dot(a,a) == approx(dot(b,b))) and (Gram_det_2(a-b,RF,p+q,RF) == approx(0)) and (sym_2_Gram_det(a,RF) == approx(sym_2_Gram_det(b,RF))))

    def case_2_symmetry(a,b,c,d):
        return ((dot(a,a) == dot(b,b)) and (Gram_det_2(a-b,RF,p+q,RF) == 0) and (sym_2_Gram_det(a,RF) == sym_2_Gram_det(b,RF)))

    case_3 = case_2_symmetry(a, c, b, d)
    case_4 = case_2_symmetry(a, d, c, b)
    case_5 = case_2_symmetry(b, c, a, d)
    case_6 = case_2_symmetry(b, d, a, c)
    case_7 = case_2_symmetry(c, d, a, b)

    case_8 = ((dot(a,a) == approx(dot(b,b))) and (Gram_det_2(a-b,RF,p+q,RF) == approx(0)) and (sym_2_Gram_det(a,RF) == approx(sym_2_Gram_det(b,RF))) and (dot(c,c) == approx(dot(d,d))) and (Gram_det_2(c-d,RF,p+q,RF) == approx(0)) and (sym_2_Gram_det(c,RF) == approx(sym_2_Gram_det(d,RF))) and (Gram_det_2(a-b,RF,c+d,RF) == approx(0)))

    def case_8_symmetry(a,b,c,d):
        return ((dot(a,a) == approx(dot(b,b))) and (Gram_det_2(a-b,RF,p+q,RF) == approx(0)) and (sym_2_Gram_det(a,RF) == approx(sym_2_Gram_det(b,RF))) and (dot(c,c) == approx(dot(d,d))) and (Gram_det_2(c-d,RF,p+q,RF) == approx(0)) and (sym_2_Gram_det(c,RF) == approx(sym_2_Gram_det(d,RF))) and (Gram_det_2(a-b,RF,c+d,RF) == approx(0)))

    case_9 = case_8_symmetry(a,c,b,d)
    case_10 = case_8_symmetry(a,d,b,c)

    return (case_1 or case_2 or case_3 or case_4 or case_5 or case_6 or case_7 or case_8 or case_9 or case_10)

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

    rdm = randint(0, 0)

    if rdm == 0:
        mp, mq, ma, mb, mc, md = 0, 0, randint(0, 10), randint(0, 10), randint(0, 10), randint(0, 10)
        p = uniform(-10, 10)*e3
        q = uniform(-10, 10)*e3
        a = uniform(-10, 10) * e1 + uniform(-10, 10) * e2 + uniform(-10, 10) * e3
        b = uniform(-10, 10) * e1 + uniform(-10, 10) * e2 + uniform(-10, 10) * e3
        c = uniform(-10, 10) * e1 + uniform(-10, 10) * e2 + uniform(-10, 10) * e3
        d = - a - b - c

    if rdm == 1:
        # non chiral
        mp, mq, ma, mb, mc, md = 0, 0, randint(0, 10), randint(0, 10), randint(0, 10), randint(0, 10)
        p = uniform(-10, 10)*e3
        q = uniform(-10, 10)*e3
        a = uniform(-10, 10) * e1 + uniform(-10, 10) * e3
        b = uniform(-10, 10) * e1 + uniform(-10, 10) * e3
        c = uniform(-10, 10) * e1 + uniform(-10, 10) * e3
        d = - a - b - c

    if rdm == 2:
        # (ab) non chiral
        mp, mq, ma, mb, mc, md = 0, 0, 1, 1, randint(0, 10), randint(0, 10)
        p = uniform(-10, 10)*e3
        q = uniform(-10, 10)*e3
        a = 4 * e1 + 5 * e2 + 3 * e3
        B = (uniform(-10, 10) * (e1 ^ e2) + uniform(-10, 10) * (e1 ^ e3) + uniform(-10, 10) * (e2 ^ e3)).normal()
        R = e ^ (uniform(0, 2 * pi) * B)
        b = -5 * e1 + 4 * e2 + 3 * e3
        c = a+b
        d = -c

    if rdm == 3:
        # (ab)(cd) non chiral
        mp, mq, ma, mb, mc, md = 0, 0, 1, 1, 2, 2
        p = uniform(-10, 10)*e3
        q = uniform(-10, 10)*e3
        angle_a = uniform(0,pi/5)
        angle_c = uniform(0, pi / 5)
        a = cos(angle_a)*e1 + sin(angle_a)*e2 + uniform(-10, 10)*e3
        b = sin(angle_a)*e1 + cos(angle_a)*e2 + a[3]*e3
        c = -cos(angle_c)*e1 - sin(angle_c)*e2 - uniform(-10, 10)*e3
        d = -sin(angle_c)*e1 - cos(angle_c)*e2 + c[3]*e3

    Ep, Eq, Ea, Eb, Ec, Ed = energy(mp, p), energy(mq, q), energy(ma, a), energy(mb, b), energy(mc, c), energy(md, d)

    M = [mp, mq, ma, mb, mc, md]
    E = [Ep, Eq, Ea, Eb, Ec, Ed]
    S = [p, q, a, b, c, d]

    return S, E, M

def chirality_test():

    chiral_states = []
    non_chiral_states = []

    S, E, M = construct_state()
    p, q, a, b, c, d = S[0], S[1], S[2], S[3], S[4], S[5]

    # These lists hold indices (as they appear in S) of the particles that can be permuted with a,b,c and d respectively
    permute_with_a = permute_with_idx(M, E, 2)
    permute_with_b = permute_with_idx(M, E, 3)
    permute_with_c = permute_with_idx(M, E, 4)
    permute_with_d = permute_with_idx(M, E, 5)

    S_parity = parity(S)  # Perform parity on the set of momenta

    flag = False  # The flag is set to true if the state is non-chiral

    R1 = e1*e3
    S1 = rotate(S_parity, R1) # Now p and q are fixed back to their original state

    if (a[1] != 0) or (a[2] != 0): # If a has components in the 1-2 plane

        for idx in permute_with_a + [2]: # For every x that can be permuted with a, map x to a and perform (ax)

            x = S1[idx]
            x_12 = x - x[3]*e3
            a_12 = a - a[3]*e3

            n = a_12 + x_12

            if n == 0:

                R2 = e1*e2

            else:

                R2 = a_12.normal()*n.normal()

            S2 = rotate(S1, R2)
            S3 = swap(S2, 2, idx)

            if S == S3:
                flag = True

            # The degrees of freedom left now that a is fixed to its original state are permutations between b,c,d

            flag_tmp_1 = permutation_boolean(M, E, 3, 4)  # This checks if (bc) is available

            if flag_tmp_1:
                S4 = swap(S3, 3, 4)
                if S == S4:
                    flag = True

            flag_tmp_2 = permutation_boolean(M, E, 3, 5)  # This checks if (bd) is available

            if flag_tmp_2:
                S5 = swap(S3, 3, 5)
                if S == S5:
                    flag = True

            flag_tmp_3 = permutation_boolean(M, E, 4, 5)  # If we have (cd)

            if flag_tmp_3:
                S6 = swap(S3, 4, 5)
                if S == S6:
                    flag = True

            if flag_tmp_1 and flag_tmp_2:  # If we have (bc) and (bd) then we have (bcd)
                # The following achieves b->c->d->b
                S7 = swap(S3, 3, 4)  # (bc)
                S8 = swap(S7, 3, 5)  # (bd), here in the 3 index lies c but we name it b still, notice we use S6
                if S == S8:
                    flag = True

                # The following achieves b->d->c->b
                S9 = swap(S3, 3, 5)  # (bd)
                S10 = swap(S9, 3, 4)  # (bc), here in the 3 index lies d but we name it b still, notice we use S7
                if S == S10:
                    flag = True

    elif (b[1] != 0) or (b[2] != 0): # To get here we asserted that a is collinear with e3 so fixed by R1

        for idx in permute_with_b + [3]:  # For every x that can be permuted with b, map x to b and perform (bx)

            if idx == 2: # We do not want permutations with a since a is fixed by R1 when collinear with e3
                continue

            x = S1[idx]
            x_12 = x - x[3] * e3
            b_12 = b - b[3] * e3

            n = b_12 + x_12

            if n == 0:

                R2 = e1 * e2

            else:

                R2 = b_12.normal() * n.normal()

            S2 = rotate(S1, R2)
            S3 = swap(S2, 3, idx)

            if S == S3:
                flag = True

            # The degrees of freedom left now that b is fixed to its original state are permutations between c,d

            flag_tmp_1 = permutation_boolean(M, E, 4, 5)  # This checks if (cd) is available

            if flag_tmp_1:
                S4 = swap(S3, 4, 5)
                if S == S4:
                    flag = True

    elif (c[1] != 0) or (c[2] != 0): # To get here we asserted that a,b are collinear with e3 so fixed by R1

        for idx in permute_with_c + [4]:  # For every x that can be permuted with c, map x to c and perform (cx)

            if (idx == 2) or (idx == 3):  # We do not want permutations with a,b since a,b are fixed by R1
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
            S3 = swap(S2, 4, idx)

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
#     flag = logic_statement_true_for_non_chiral(non_chiral_state[0], non_chiral_state[1])
#     if flag:
#         non_chiral_evaluation_on_logic_statement[0] += 1
#     else:
#         non_chiral_evaluation_on_logic_statement[1] += 1
#
# chiral_evaluation_on_logic_statement = [0, 0]
# for chiral_state in chiral_states_list:
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
# plt.savefig('non_chiral_non_collision_massless_logic_statement_test.pdf', bbox_inches='tight')
#
# plt.bar(x, height_chiral, color = 'k', width = 0.1)
# plt.title('Chiral states evaluated on the logic statement\nwhich is true iff the input is non-chiral')
# plt.ylabel('Frequency')
# plt.show()
# plt.savefig('chiral_non_collision_massless_logic_statement_test.pdf', bbox_inches='tight')

print(chirality_test())
