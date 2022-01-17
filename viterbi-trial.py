import copy
import json

# problem definition
states = ("A", "S", "U", "P", "R", "G")
trans = {"A" : {"A": 0.01, "S": 0.15, "U": 0.25, "P": 0.20, "R": 0.20, "G": 0.19},
         "S" : {"A": 0.1, "S": 0.05, "U": 0.25, "P": 0.35, "R": 0.1, "G": 0.15},
         "U" : {"A": 0.05, "S": 0.1, "U": 0.05, "P": 0.35, "R": 0.40, "G": 0.05},
         "P" : {"A": 0.20, "S": 0.15, "U": 0.1, "P": 0.01, "R": 0.35, "G": 0.19},
         "R" : {"A": 0.1, "S": 0.05, "U": 0.5, "P": 0.1, "R": 0.1, "G": 0.15},
         "G" : {"A": 0.19, "S": 0.15, "U": 0.25, "P": 0.20, "R": 0.20, "G": 0.01},
         }
emission = {"A": {"S_Noisy" : 0.2, "U_Noisy" : 0.2, "P_Noisy" : 0.3, "R_Noisy" : 0.3},
            "S": {"S_Noisy" : 0.25, "U_Noisy" : 0.25, "P_Noisy" : 0.25, "R_Noisy" : 0.25},
            "U": {"S_Noisy" : 0.1, "U_Noisy" : 0.5, "P_Noisy" : 0.2, "R_Noisy" : 0.2},
            "P": {"S_Noisy" : 0.1, "U_Noisy" : 0.1, "P_Noisy" : 0.6, "R_Noisy" : 0.2},
            "R": {"S_Noisy" : 0.1, "U_Noisy" : 0.1, "P_Noisy" : 0.3, "R_Noisy" : 0.5},
            "G": {"S_Noisy" : 0.25, "U_Noisy" : 0.25, "P_Noisy" : 0.25, "R_Noisy" : 0.25},
            }
initial = {"A": 0.1, "S": 0.25, "U": 0.15, "P": 0.15, "R": 0.25, "G": 0.20}
observed = ["S_Noisy", "U_Noisy", "P_Noisy", "R_Noisy"]

print("%40s: %s " % ("Observed sequence", str(observed)))

N=len(observed)

#############################
# Obviously brute force is a big mess, and slow -- each every day requires another nested loop and 2x the computation time.
# so instead, compute using Viterbi!

V_table = {i:[0] * N for i in states}
which_table = {i:[0] * N for i in states}

# Here you'll have a loop to build up the viterbi table, left to right
for s in states:
    V_table[s][0] = initial[s] * emission[s][observed[0]]

for i in range(1, N):
    for s in states:
        (which_table[s][i], V_table[s][i]) =  max( [ (s0, V_table[s0][i-1] * trans[s0][s]) for s0 in states ], key=lambda l:l[1] ) 
        V_table[s][i] *= emission[s][observed[i]]

# Here you'll have a loop that backtracks to find the most likely state sequence
viterbi_seq = [""] * N
temp = [(state, V_table[state][N-1]) for state in states]
viterbi_seq[N-1], _ = max(temp, key=lambda array:array[1])
for i in range(N-2, -1, -1):
    viterbi_seq[i] = which_table[viterbi_seq[i+1]][i+1]
    
print("%40s: %s" % ("Most likely sequence by Viterbi:", str(viterbi_seq)))
