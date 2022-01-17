#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: Srimanth Agastyaraju, sragas
# (based on skeleton code by D. Crandall, Oct 2020)
#

from PIL import Image, ImageDraw, ImageFont
import sys

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
# print("\n".join([ r for r in train_letters['g'] ]))

# Same with test letters. Here's what the third letter of the test data
#  looks like:
# print("\n".join([ r for r in test_letters[2] ]))


# My imports
from itertools import chain
import math


class SimpleModel():
    def __init__(self) -> None:
        pass


    @staticmethod
    def convert_to_sparse_matrix(character_matrix):
        sparse_matrix = []
        for row in character_matrix:
            sparse_row = [1 if row[i]=="*" else 0 for i in range(len(row))]
            sparse_matrix.append(sparse_row)
        return sparse_matrix


    @staticmethod
    def flatten_to_1D(sparse_matrix):
        return list(chain.from_iterable(sparse_matrix))


    @staticmethod
    def union_and_intersection(train_1D_sparse, test_1D_sparse):
        union_count = 0
        intersection_count = 0
        for i in range(len(train_1D_sparse)):
            union_count += train_1D_sparse[i] or test_1D_sparse[i]
            intersection_count += train_1D_sparse[i] and test_1D_sparse[i]
        return union_count, intersection_count


    def probability(self, training_matrix, test_matrix):
        train_1D_sparse = self.flatten_to_1D(self.convert_to_sparse_matrix(training_matrix))
        test_1D_sparse = self.flatten_to_1D(self.convert_to_sparse_matrix(test_matrix))
        union_count, intersection_count = self.union_and_intersection(train_1D_sparse, test_1D_sparse)
        noisy_pixels = union_count - intersection_count
        fraction_of_noisy_pixels = (noisy_pixels + 1) / (CHARACTER_HEIGHT * CHARACTER_WIDTH)
        return (1-fraction_of_noisy_pixels)


    def probability_2(self, training_matrix, test_matrix):
        # Intersection over Union (IoU)
        train_1D_sparse = self.flatten_to_1D(self.convert_to_sparse_matrix(training_matrix))
        test_1D_sparse = self.flatten_to_1D(self.convert_to_sparse_matrix(test_matrix))
        union_count, intersection_count = self.union_and_intersection(train_1D_sparse, test_1D_sparse)
        probability = ((intersection_count + 1) / (union_count + 1))
        return probability


    def probability_3(self, training_matrix, test_matrix):
        # Reference: Dice coefficient
        # https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
        train_1D_sparse = self.flatten_to_1D(self.convert_to_sparse_matrix(training_matrix))
        test_1D_sparse = self.flatten_to_1D(self.convert_to_sparse_matrix(test_matrix))
        _, intersection_count = self.union_and_intersection(train_1D_sparse, test_1D_sparse)
        sum_of_train = sum(train_1D_sparse)
        sum_of_test = sum(test_1D_sparse)
        dice_score = (intersection_count + 1) / (sum_of_train + sum_of_test + 1)
        return dice_score


    def pixel_wise_probability(self, train_char_flat, test_char_flat):
        probability = 1
        penalty = 0.2 # m % of noisy data, given in the hints of the problem.
        reward = 1 - penalty
        for i in range(len(test_char_flat)):
            # True positives and true negatives
            pixel_result = (train_char_flat[i] and test_char_flat[i]) or (not train_char_flat[i] and not test_char_flat[i])
            if(pixel_result):
                probability *= reward
            else:
                probability *= penalty
        return probability


    def simple_model(self, training_matrices, test_matrices):
        # Taken from the starter code.
        states = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
        final_sentence = ""
        for test_matrix in test_matrices:
            max_probability = 0
            argmax_char = ''
            for state in states:
                training_matrix = training_matrices[state]
                probability = self.probability_2(training_matrix, test_matrix)
                if(max_probability < probability):
                    max_probability = probability
                    argmax_char = state
            final_sentence += argmax_char
        return final_sentence



class HMM(SimpleModel):
    def __init__(self) -> None:
        super().__init__()
    
    def convert_to_sparse_matrix(self, character_matrix):
        return SimpleModel.convert_to_sparse_matrix(character_matrix)
    
    def flatten_to_1D(self, sparse_matrix):
        return SimpleModel.flatten_to_1D(sparse_matrix)
    
    def probability_2(self, training_matrix, test_matrix):
        return super().probability_2(training_matrix, test_matrix)
    
    def probability_3(self, training_matrix, test_matrix):
        return super().probability_3(training_matrix, test_matrix)
    
    def pixel_wise_probability(self, train_char_flat, test_char_flat):
        return super().pixel_wise_probability(train_char_flat, test_char_flat)

    
    def emission_probabilities(self, train_matrices, test_matrices):
        N = len(test_matrices)
        observed_sequence = ['observed_'+str(i) for i in range(N)]
        # Taken from the starter code.
        states = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
        emission_table = {state:{observed:0 for observed in observed_sequence} for state in states}

        for i in range(N):
            for state in states:
                emission_table[state][observed_sequence[i]] = \
                    math.log(self.probability_3(train_matrices[state], test_matrices[i])) * 50
        
        return emission_table
    
    
    def initial_transition_train(self):
        # Taken from label.py in part 1.
        raw_word_data = []
        training_file_path = train_txt_fname
        training_file = open(training_file_path, 'r')
        for line in training_file:
            word_and_pos = tuple([word for word in line.split()])
            # For bc.train where there are word and POS in alternative occurences, use the below linw.
            raw_word_data += [(word_and_pos[0::2]), ]
            # For any text file in general, uncomment the below line.
            # raw_word_data += [word_and_pos]

        cleaned_data = ""
        # Taken from the starter code.
        states = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
        for line1 in raw_word_data:
            new_sentence = ""
            for line in line1:
                new_sentence += " " + ''.join(char for char in line if char in states)
            new_sentence = new_sentence.replace(' ,', ',')
            new_sentence = new_sentence.replace(' ,', ',')
            new_sentence = new_sentence.replace(' \'\'','\"')
            new_sentence = new_sentence.replace('`` ','\"')
            new_sentence = new_sentence.replace('``','\"')
            new_sentence = new_sentence.replace("  ", " ")
            new_sentence = new_sentence.replace(" .", ".").strip()
            cleaned_data += new_sentence + "\n"
        
        cleaned_data = cleaned_data.strip()
        # https://www.govinfo.gov/content/pkg/CDOC-110hdoc50/html/CDOC-110hdoc50.htm
        # cleaned_data = open('train.txt', 'r').read()

        transition_frequencies = {i:{j:0.1 for j in states} for i in states}
        for i in range(len(cleaned_data) - 1):
            current_char = cleaned_data[i]
            next_char = cleaned_data[i+1]
            if(current_char in transition_frequencies):
                if(next_char in transition_frequencies[current_char]):
                    transition_frequencies[current_char][next_char] += 1

        initial_frequencies = {i: 0.0000000000001 for i in states}
        for character in cleaned_data:
            if(character in initial_frequencies):
                initial_frequencies[character] += 1
        # Too many spaces and quotes in the dataset. Also, in reality, initial character being a space or quote is quite low.
        initial_frequencies[' '] = initial_frequencies[' '] / 1000
        initial_frequencies["'"] = initial_frequencies["'"] / 10
        initial_probabilities = {item: math.log(value/sum(initial_frequencies.values())) for item, value in initial_frequencies.items()}
        
        # Summing up probabilities in a dictionary comprehension
        # https://stackoverflow.com/questions/30964577/divide-each-python-dictionary-value-by-total-value/30964739
        transition_probabilities = {i:{k: math.log(v/total) for total in (sum(transition_frequencies[i].values()),) for k, v in transition_frequencies[i].items()} for i in states}

        return initial_probabilities, transition_probabilities
    

    def hidden_markov_model(self, train_matrices, test_matrices):
        initial_table, transition_table = self.initial_transition_train()
        emission_table = self.emission_probabilities(train_matrices, test_matrices)
        # Taken from the starter code.
        states = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
        N_observed = len(test_matrices)
        observed_sequence = ['observed_' + str(i) for i in range(N_observed)]

        return self.viterbi(observed_sequence, states, initial_table, emission_table, transition_table, N_observed)


    def viterbi(self, observed, states, initial, emission, transition, N_observed):
        # The viterbi algorithm code, taken from the solution provided for the in-class activity 2 on Oct 20, 2021
        # Here you'll have a loop to build up the viterbi table, left to right
        viterbi_table = {state:[0] * N_observed for state in states}
        which_table = {state:[0] * N_observed for state in states}
        # Here you'll have a loop to build up the viterbi table, left to right
        for state in states:
            viterbi_table[state][0] =  initial[state] + emission[state][observed[0]]
        for i in range(1, N_observed):
            for state in states:
                (which_table[state][i], viterbi_table[state][i]) =  max( [ (s0, viterbi_table[s0][i-1] + transition[s0][state]) for s0 in states ], key=lambda l:l[1] ) 
                viterbi_table[state][i] += emission[state][observed[i]]
        
        # Here you'll have a loop that backtracks to find the most likely state sequence
        viterbi_seq = [""] * N_observed
        temp = [(state, viterbi_table[state][N_observed-1]) for state in states]
        viterbi_seq[N_observed-1], _ = max(temp, key=lambda array:array[1])
        for i in range(N_observed-2, -1, -1):
            viterbi_seq[i] = which_table[viterbi_seq[i+1]][i+1]
        
        return viterbi_seq


if(__name__ == '__main__'):

    sm = SimpleModel()
    simple_result = sm.simple_model(train_letters, test_letters)

    hmm = HMM()
    # print(hmm.initial_transition_train())
    hmm_result = hmm.hidden_markov_model(train_letters, test_letters)

    # The final two lines of your output should look something like this:
    print("Simple: " + simple_result)
    print("   HMM: " + ''.join(hmm_result)) 
