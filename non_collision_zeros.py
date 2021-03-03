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

def construct_state():

    rdm = randint(1, 2)

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

    len_list = [len_b, len_c, len_d]

    len_max = max(len_list)

    S1 = parity(S)
    R1 = e1 * e3  # This is R_y(pi) used to map a back so that we avoid boosts and rotations in x-z,y-z planes
    S2 = rotate(S1, R1)

    flag = False # This is set to true when the state is non-chiral
    if len_max == 0: # If there are no permutations

        b_proj = b - b[3]*e3
        b_rot = S2[1]
        b_rot_proj = b_rot - b_rot[3]*e3
        # Rotate b_rot_proj back to b_proj
        n = (b_rot_proj + b_proj).normal()
        if n == 0:
            R2 = e1*e2
        else:
            R2 = b_proj.normal()*n
        S3 = rotate(S2, R2)
        if S == S3:
            flag = True
    else:

        b_proj = b - b[3] * e3
        b_rot = S2[1]
        b_rot_proj = b_rot - b_rot[3] * e3

        # Rotate b_rot_proj back to b_proj
        n = (b_rot_proj + b_proj).normal()
        if n == 0:
            R2 = e1 * e2
        else:
            R2 = b_proj.normal() * n
        S3 = rotate(S2, R2)
        if S == S3:
            flag = True

        if 3 in permute_with_c: # if c and d can be permuted
            if S == swap(S3, 2, 3):
                flag = True

        if permute_with_b: # In summary: a is mapped, map b (by swap) and now only freedom is swap c,d
            for idx in permute_with_b:
                x_rot = S2[idx]
                x_rot_proj = x_rot - x_rot[3]*e3
                n = (x_rot_proj+b_proj).normal()
                if n == 0:
                    R2 = e1*e2
                else:
                    R2 = b_proj.normal()*n
                S3 = rotate(S2, R2)
                S4 = swap(S3, idx, 1) # Swap x with b
                if S == S3:
                    flag = True
                if 5-idx in permute_with_c: # If c and d can be permuted
                    S5 = swap(S4, 2, 3)
                    if S == S5:
                        flag = True
    if flag:
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
    flag = logic_statement_true_for_non_chiral(non_chiral_state[0], non_chiral_state[1])
    if flag:
        non_chiral_evaluation_on_logic_statement[0] += 1
    else:
        non_chiral_evaluation_on_logic_statement[1] += 1

chiral_evaluation_on_logic_statement = [0, 0]
for chiral_state in chiral_states_list:
    mp, mq = uniform(1, 10), uniform(1, 10)
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
plt.savefig('non_chiral_non_collision_logic_statement_test.pdf', bbox_inches='tight')

plt.bar(x, height_chiral, color = 'k', width = 0.1)
plt.title('Chiral states evaluated on the logic statement\nwhich is true iff the input is non-chiral')
plt.ylabel('Frequency')
#plt.show()
#plt.savefig('chiral_non_collision_logic_statement_test.pdf', bbox_inches='tight')

















