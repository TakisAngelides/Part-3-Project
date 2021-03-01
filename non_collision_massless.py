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

def construct_state():

    rdm = randint(1, 3)

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

    same_mass_with_a = [idx for idx in range(len(M)) if M[idx] == M[2] and idx != 2]
    same_energy_with_a = [idx for idx in range(len(E)) if E[idx] == approx(E[2]) and idx != 2]
    # Holds the indices of particles that can be permuted with a
    permute_with_a = list(set(same_mass_with_a) and set(same_energy_with_a))
    len_a = len(permute_with_a)

    same_mass_with_b = [idx for idx in range(len(M)) if M[idx] == M[3] and idx != 3]
    same_energy_with_b = [idx for idx in range(len(E)) if E[idx] == approx(E[3]) and idx != 3]
    # Holds the indices of particles that can be permuted with b
    permute_with_b = list(set(same_mass_with_b) and set(same_energy_with_b))
    len_b = len(permute_with_b)

    same_mass_with_c = [idx for idx in range(len(M)) if M[idx] == M[4] and idx != 4]
    same_energy_with_c = [idx for idx in range(len(E)) if E[idx] == approx(E[4]) and idx != 4]
    # Holds the indices of particles that can be permuted with c
    permute_with_c = list(set(same_mass_with_c) and set(same_energy_with_c))
    len_c = len(permute_with_c)

    same_mass_with_d = [idx for idx in range(len(M)) if M[idx] == M[5] and idx != 5]
    same_energy_with_d = [idx for idx in range(len(E)) if E[idx] == approx(E[5]) and idx != 5]
    # Holds the indices of particles that can be permuted with d
    permute_with_d = list(set(same_mass_with_d) and set(same_energy_with_d))
    len_d = len(permute_with_d)

    permutation_dictionary = {2:permute_with_a,3:permute_with_b,4:permute_with_c,5:permute_with_d}

    len_list = [len_a, len_b, len_c, len_d]

    len_max = max(len_list)

    S1 = parity(S)
    R1 = e1 * e3  # This is R_y(pi) used to map p and q back so that we avoid boosts and rotations in x-z,y-z planes
    S2 = rotate(S1, R1)
    a_proj = a - a[3] * e3

    flag = False  # This is set to true when the state is non-chiral
    if len_max == 0:  # If there are no permutations (this can also be done just using p^a without e1e3)

        # Project a into 1-2 plane and rotate it into its original
        a_rot = S2[2]
        a_rot_proj = a_rot - a_rot[3]*e3

        if a_proj + a_rot_proj == 0:
            R2 = e1*e2
        else:
            n = (a_rot_proj + a_proj).normal()
            R2 = a_proj.normal()*n
        S3 = rotate(S2, R2)
        if S == S3:
            flag = True

    else:

        # Project a into 1-2 plane and rotate it into its original
        a_rot = S2[2]
        a_rot_proj = a_rot - a_rot[3] * e3

        if a_proj + a_rot_proj == 0:
            R2 = e1 * e2
        else:
            n = (a_rot_proj + a_proj).normal()
            R2 = a_proj.normal() * n
        S3 = rotate(S2, R2)
        if S == S3:
            flag = True

        for idx in permute_with_a:

            x_rot = S2[idx]
            x_rot_proj = x_rot - x_rot[3]*e3
            if a_proj + x_rot_proj == 0:
                R2 = e1 * e2
            else:
                n = (x_rot_proj + a_proj).normal()
                R2 = a_proj.normal() * n
            S3 = rotate(S2, R2)
            S4 = swap(S3, idx, 2) # index 2 corresponds to a
            if S == S4:
                flag = True
            # Check if the other 2 other than a,x can be permuted
            idx_list = [3, 4, 5]
            idx_list.remove(idx)
            if idx_list[0] in permutation_dictionary[idx_list[1]]:
                S5 = swap(S4, idx_list[0], idx_list[1])
                if S == S5:
                    flag = True
    if flag:
        non_chiral_states.append([S, E])  # only possibility is all lie in the same plane
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
    flag = logic_statement_true_for_non_chiral(non_chiral_state[0], non_chiral_state[1])
    if flag:
        non_chiral_evaluation_on_logic_statement[0] += 1
    else:
        non_chiral_evaluation_on_logic_statement[1] += 1

chiral_evaluation_on_logic_statement = [0, 0]
for chiral_state in chiral_states_list:
    flag = logic_statement_true_for_non_chiral(chiral_state[0], chiral_state[1])
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
plt.savefig('non_chiral_non_collision_massless_logic_statement_test.pdf', bbox_inches='tight')

plt.bar(x, height_chiral, color = 'k', width = 0.1)
plt.title('Chiral states evaluated on the logic statement\nwhich is true iff the input is non-chiral')
plt.ylabel('Frequency')
#plt.show()
#plt.savefig('chiral_non_collision_massless_logic_statement_test.pdf', bbox_inches='tight')