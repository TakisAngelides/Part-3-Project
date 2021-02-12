from clifford.g3 import blades
from main import multivec_to_vec, energy, parity, rotate, swap
from numpy.random import uniform, randint

e1, e2, e3 = blades['e1'], blades['e2'], blades['e3'] # Create the basis vectors from the Clifford library

# Masses
mp, mq, ma, mb, mc, md = randint(0, 10), randint(0, 10), randint(0, 10), randint(0, 10), randint(0, 10), randint(0,10)

# 3-Momenta
p = uniform(-10, 10) * e3
q = -p
a = uniform(-10, 10) * e1 + uniform(-10, 10) * e2 + uniform(-10, 10) * e3
b = uniform(-10, 10) * e1 + uniform(-10, 10) * e2 + uniform(-10, 10) * e3
c = uniform(-10, 10) * e1 + uniform(-10, 10) * e2 + uniform(-10, 10) * e3
d = uniform(-10, 10) * e1 + uniform(-10, 10) * e2 + uniform(-10, 10) * e3

# Energies
Ep, Eq, Ea, Eb, Ec, Ed = energy(mp, p), energy(mq, q), energy(ma, a), energy(mb, b), energy(mc, c), energy(md, d)

M = [mp, mq, ma, mb, mc, md]
E = [Ep, Eq, Ea, Eb, Ec, Ed]
S = [p, q, a, b, c, d]

S_parity = parity(S) # Perform parity on the set of momenta

same_mass_with_a = [idx for idx in range(len(M)) if M[idx] == M[2] and idx != 2]
same_energy_with_a = [idx for idx in range(len(E)) if E[idx] == E[2] and idx != 2]
# Holds the indices of particles that can be permuted with a
permute_with_a = list(set(same_mass_with_a) and set(same_energy_with_a))

same_mass_with_b = [idx for idx in range(len(M)) if M[idx] == M[3] and idx != 3]
same_energy_with_b = [idx for idx in range(len(E)) if E[idx] == E[3] and idx != 3]
# Holds the indices of particles that can be permuted with b
permute_with_b = list(set(same_mass_with_b) and set(same_energy_with_b))

# The indices of b,c,d in S are 3,4,5 which sum to 12. If I can permute a with eg b then I need to check if I can
# silmutaneously permute c,d with indices [4,5] so the dictionary for 12-3 points to [4,5] where 3 is b
swap_dictionary = {9: [4, 5], 8: [3,5], 7: [3, 4]}

if not permute_with_a:
    # For no permutations at all and for pq permutations
    # ie pq permutations don't change anything if there are no final state permutations
    a_12 = a - a[3] * e3
    R = a_12.normal() * e3
    S_final = rotate(S_parity, R)
    if S == S_final:
        print('Non-chiral')
    elif permute_with_b and 2 not in permute_with_b: # Check if non-chiral for (bc) or (bd)
        for idx in permute_with_b:
            if S == swap(S_final, 3, idx):
                print('Non-chiral')
    elif mc == md and Ec == Ed: # Final check available (cd)
        if S == swap(S_final, 4, 5):
            print('Non-chiral')
    else:
        print('Chiral')
else:
    flag = False # Flag is set to true when one of the permutations managed to map everything back
    for idx in permute_with_a:
        # For ax permutations but no pq permutations
        Ea, Ex = E[2], E[idx]
        a, x = S[2], S[idx]
        # The other 2 final state particles besides a and x have indices
        idx_1 = swap_dictionary[12 - idx][0]
        idx_2 = swap_dictionary[12 - idx][1]
        if a[3] == x[idx] and mp != mq:
            R1 = e1 * e3 # Forced to do R1 since no pq permutation
            S_1 = rotate(S_parity, R1)
            x_new = S_1[3]
            x_12 = x_new - x_new[3] * e3
            a_initial = S[2]
            a_12 = a_initial - a_initial[3] * e3
            if a_12 == -x_12: # If we need a pi rotation then use this construction
                R2 = e1*e2
            else:
                n = (a_12 + x_12).normal()
                R2 = a_12.normal() * n
            S_2 = rotate(S_1, R2)
            # (ax permutation) index 2 corresponds to a and index idx corresponds to x in the list S
            S_final = swap(S_2, 2, idx)
            if S == S_final:
                print('Non-chiral')
                flag = True # Flag is set to true when one of the permutations managed to map everything back
                break
            # Last check to make here is to swap the other 2 final particles if we can
            elif E[idx_1] == E[idx_2] and M[idx_1] == M[idx_2] and S == swap(S_final, idx_1, idx_2):
                print('Non-chiral')
                flag = True  # Flag is set to true when one of the permutations managed to map everything back
                break
            else:
                continue
        # For ax and pq permutations
        # If i have pq permutation and energies of a, x match and z component of a,x match in magnitude
        else:
            a_initial = S[2]
            a_12 = a_initial - a_initial[3] * e3
            if a[3] == -x[3]:  # if the z comp of a,b is opposite
                x_new = S_parity[idx]
                x_12 = x_new - x_new[3] * e3  # Project x into the 1-2 plane
                # If the following is true it means we need to construct a rotor to do a pi rotation in the 1-2 plane
                if a_12 == -x_12:
                    R2 = e1*e2
                    S_final = rotate(S_parity, R2) # Only use R2 and not R1
                    S_final = swap(S_final, 0, 1) # Swap p and q
                    S_final = swap(S_final, 2, idx)  # Swap a and x
                    if S == S_final:
                        print('Non-chiral')
                        flag = True
                        break
                    elif E[idx_1] == E[idx_2] and M[idx_1] == M[idx_2] and S == swap(S_final, idx_1, idx_2):
                        print('Non-chiral')
                        flag = True  # Flag is set to true when one of the permutations managed to map everything back
                        break
                    else:
                        continue
                else:
                    n = (a_12 + x_12).normal()
                    R2 = a_12.normal()*n
                    S_final = rotate(S_parity, R2)  # only use R2 and not R1
                    S_final = swap(S_final, 0, 1)  # swap p and q
                    S_final = swap(S_final, 2, idx)  # swap a and b
                    if S == S_final:
                        print('Non-chiral')
                        flag = True
                        break
                    elif E[idx_1] == E[idx_2] and M[idx_1] == M[idx_2] and S == swap(S_final, idx_1, idx_2):
                        print('Non-chiral')
                        flag = True  # Flag is set to true when one of the permutations managed to map everything back
                        break
                    else:
                        continue
            else: # If the z component of a,x is equal
                # Forced to do R1 since no pq permutation and so we have to bring the parity b
                # which has opposite sign z back to positive sign
                R1 = e1 * e3
                S_1 = rotate(S_parity, R1)
                x_new = S_1[idx]
                x_12 = x_new - x_new[3] * e3
                a_initial = S[2]
                a_12 = a_initial - a_initial[3] * e3
                if a_12 == -x_12:
                    R2 = e1*e2
                else:
                    n = (a_12 + x_12).normal()
                    R2 = a_12.normal() * n
                S_2 = rotate(S_1, R2)
                # (ax permutation) index 2 corresponds to a and index 3 corresponds to b in the list S
                S_final = swap(S_2, 2, idx)
                if S == S_final:
                    print('Non-chiral')
                    flag = True
                    break
                elif E[idx_1] == E[idx_2] and M[idx_1] == M[idx_2] and S == swap(S_final, idx_1, idx_2):
                    print('Non-chiral')
                    flag = True  # Flag is set to true when one of the permutations managed to map everything back
                    break
                else:
                    continue
    if not flag: # If any available permutation has not managed to map everything back to itself then print chiral
        print('Chiral')









