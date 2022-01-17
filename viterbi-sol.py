import copy
import json

# problem definition
states = ("R", "S")
trans = {"R" : {"R": 0.65, "S": 0.35},
         "S" : {"R" : 0.25, "S": 0.75}}
emission = {"R": {"Y" : 0.8, "N" : 0.2},
            "S" : {"Y": 0.2, "N": 0.8}}
initial = {"R": 0.5, "S": 0.5}
observed = ["Y", "N", "Y", "Y", "Y", "Y", "N"]

print("%40s: %s " % ("Observed sequence", str(observed)))

N=len(observed)

###########################
# We can find the most likely state sequence in a brute-force way by just
# trying all of them!

joint = {}
for s0 in states:
    for s1 in states:
        for s2 in states:
            for s3 in states:
                for s4 in states:
                    for s5 in states:
                        for s6 in states:
                            seq = str([s0, s1, s2, s3, s4, s5, s6])
                            joint[seq] = initial[s0] * trans[s0][s1] * trans[s1][s2] * trans[s2][s3]*  trans[s3][s4] * trans[s4][s5] * trans[s5][s6] * \
                                emission[s0][observed[0]] * emission[s1][observed[1]] * emission[s2][observed[2]] * emission[s3][observed[3]] * \
                                emission[s4][observed[4]] * emission[s5][observed[5]] * emission[s6][observed[6]]

print("%40s: %s" % ("Most likely sequence by brute force:", str(max(joint, key=joint.get))))

#############################
# obviously that's a big mess, and slow -- each every day requires another nested loop and 2x the computation time.
# so instead, compute using Viterbi!

# Viterbi table will have two rows and N columns 
V_table = {"R": [0] * N, "S" : [0] * N}
which_table = {"R": [0] * N, "S" : [0] * N}

# Here you'll have a loop to build up the viterbi table, left to right
for s in states:
    V_table[s][0] = initial[s] * emission[s][observed[0]]

json.dump(V_table, open( "V_table.json", 'w' ) )

for i in range(1, N):
    for s in states:
        (which_table[s][i], V_table[s][i]) =  max( [ (s0, V_table[s0][i-1] * trans[s0][s]) for s0 in states ], key=lambda l:l[1] ) 
        V_table[s][i] *= emission[s][observed[i]]

#       Easier to understand but longer version that does the same as the above two lines:
    #    V_table[s][i] = emission[s][observed[i]]
    #    if V_table["R"][i-1] * trans["R"][s] > V_table["S"][i-1] * V_table["S"][i-1] * trans["S"][s]:
    #        V_table[s][i] *= V_table["R"][i-1] * trans["R"][s]
    #        which_table[s][i] = "R"
    #    else:
    #        V_table[s][i] *= V_table["S"][i-1] * trans["S"][s]
    #        which_table[s][i] = "S"

json.dump(V_table, open( "V_table_final.json", 'w' ) )
# Here you'll have a loop that backtracks to find the most likely state sequence
viterbi_seq = [""] * N
viterbi_seq[N-1] = "R" if V_table["R"][i] > V_table["S"][i] else "S"
for i in range(N-2, -1, -1):
    viterbi_seq[i] = which_table[viterbi_seq[i+1]][i+1]
    
print("%40s: %s" % ("Most likely sequence by Viterbi:", str(viterbi_seq)))
