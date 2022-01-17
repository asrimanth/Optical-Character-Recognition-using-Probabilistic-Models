from PIL import Image, ImageDraw, ImageFont
import sys
import re
import math
from copy import deepcopy


CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    print(im.size)
    print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

#This block reads the data from the bc.train file and removes the parts of speech tags and also removes unwanted charecters.
def read_data(fname): 
    exemplars = []
    file = open(fname, 'r')
    for line in file:
        data =[w for w in line.split()]
        exemplars += list(re.sub(r'[&|$|*|;|`|#|@|%|^|~|/|<|>|:|[|\]|{|}|+|=|_]', r'', " ".join(data[0::2])))
    return exemplars

#This block calculates the initial probabilities of all the charecters present in the bc.train file.
def initial_probabilities(data):
    initial_probabilities_dict = dict()
    hero=data
    for i in range(len(hero)):
        if hero[i] not in initial_probabilities_dict:
            initial_probabilities_dict[hero[i]] = 1
        else:
            initial_probabilities_dict[hero[i]]+=1
    for j in initial_probabilities_dict.keys():
        initial_probabilities_dict[j]=math.log(initial_probabilities_dict[j]/len(data))
    return initial_probabilities_dict

#This block calculates the transition probabilities from one charecter to another charecter.
def transition_probabilities(data):
    hero=data
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"'/ "
    transition_probabilities_dict={i:{j:0.0000000001 for j in TRAIN_LETTERS}for i in TRAIN_LETTERS}
    # print(transition_probabilities,'!!!')
    for i in range(0,len(hero)-1):
        transition_probabilities_dict[hero[i]][hero[i+1]]=transition_probabilities_dict[hero[i]][hero[i+1]]+1
    final_transition_table = deepcopy(transition_probabilities_dict)
    for key ,value in transition_probabilities_dict.items():
        sum=0
        for i in TRAIN_LETTERS:
            if i in value.keys():
                sum += value[i]                 
        for i in TRAIN_LETTERS:
            if i in value.keys():
                final_transition_table[key][i] = math.log(value[i] / sum)
    return(final_transition_table)
    
#This block returns the simple output for the test images based on the calculated emission probabilities.
def simple(train,test):
    given_train_letters="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    str_to_list=[char for char in given_train_letters]
    a=[]
    final=[]
    for i in range (len(test)):
        a=[]
        for j in given_train_letters:
            x=calculate(test[i],train[j])
            value=(x[0]*0.85)+(x[1]*0.10)+(x[2]*0.05)
            a.append(value)
        sum_a = sum(a)
        for m in range(len(a)):
            a[m] = a[m]/sum_a
        ind=a.index(max(a))
        final.append(str_to_list[ind])
        final_string=''.join(final)
    return final_string

#This block calculates the emission probabilities by comparing the number of matched stars, matched spaces and mismatches between train image charecter and train charecters.
def emission_probabilities(train,test):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    emission_probabilities_dict={i:{j:0.0000000001 for j in range(len(test))}for i in TRAIN_LETTERS}
    for i in emission_probabilities_dict:
        a=[]
        for j in range(len(test)):
            x=calculate(train[i],test[j])
            value=(x[0]*0.85)+(x[1]*0.10)+(x[2]*0.05)
            a.append(value)
        sum_emission_values=sum(a)
        for k in range(len(a)):
            a[k]=500*(math.log(a[k]/sum_emission_values))
            emission_probabilities_dict[i][k]=a[k]
    return emission_probabilities_dict

#This block calculates the number of matched stars, matched spaces and mismatches.
def calculate(train,test):
    star_count=0
    space_count=0
    mismatch_count=0
    for i in range(len(train)):
        for j in range(len(train[i])):
            if train[i][j]==test[i][j]:
                if train[i][j]=='*':
                    star_count+=1
                elif train[i][j]==' ':
                    space_count+=1
            else:
                mismatch_count+=1
    return [star_count,space_count,mismatch_count]

#### The below code has been referred from the solution of viterbi in-class activity.
def viterbi(data,train,test):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    states = tuple(TRAIN_LETTERS)
    data = read_data(data)
    trans=transition_probabilities(data)
    emission=emission_probabilities(train,test)
    initial=initial_probabilities(data)
    observed = [i for i in range(len(test))]
    N=len(observed)

    V_table = {i:[0] * N for i in states}
    which_table = {i:[0] * N for i in states}

    for s in states:
        if s not in initial.keys():
            initial[s] = math.log(0.0000000000000000001)
        V_table[s][0] = initial[s] + emission[s][observed[0]]

    for i in range(1, N):
        for s in states:
            (which_table[s][i], V_table[s][i]) =  max( [ (s0, V_table[s0][i-1] + trans[s0][s]) for s0 in states ], key=lambda l:l[1] ) 
            V_table[s][i] += emission[s][observed[i]]

    viterbi_seq = [""] * N
    temp = [(state, V_table[state][N-1]) for state in states]
    viterbi_seq[N-1], _ = max(temp, key=lambda array:array[1])
    for i in range(N-2, -1, -1):
        viterbi_seq[i] = which_table[viterbi_seq[i+1]][i+1]
    final=''.join(viterbi_seq)
    return final  
    # print("Viterbi:", ' '.join(viterbi_seq))
#### The referred code ends here.


#####
# main program
if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

## Below is just some sample code to show you how the functions above work. 
# You can delete this and put your own code here!

# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,
#  here's what "a" looks like:
# print("\n".join([ r for r in train_letters['G'] ]))

# Same with test letters. Here's what the third letter of the test data
#  looks like:
# print("\n".join([ r for r in test_letters[2] ]))



# The final two lines of your output should look something like this:
print("Simple: " + simple(train_letters,test_letters))
print("   HMM: " + viterbi(train_txt_fname,train_letters,test_letters))