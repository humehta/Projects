#!/usr/bin/python
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: (Kushal, Harsh, Aditya)
# (based on skeleton code by D. Crandall, Oct 2017)
#
####REPORT####
#In this implementation of OCR we made use of the algorithms and training set of the part1 "Parts of Speech Tagging" problem.
#The emissions calculated for each letter image was done by comparing the pixels of each letter to the hidden state pixels and calculating the probability with respect to the matching pixels.
#For taking into account an empty image we put a threshold on the number of pixels to distinguish a space from a letter.


from PIL import Image, ImageDraw, ImageFont
import sys
from probability import *
import operator
import copy
import sys

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25

def read_data(fname):
    exemplars = []
    file = open(fname, 'r');
    return [[letter for letter in line] for line in file]

def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    #print im.size
    #print int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

def train(data):
    obs=[[alphabet for letter in line for alphabet in letter] for line in data]
    obs=[filter(lambda a:a in states,line) for line in obs]
    prior,end_prior=cal_prior(obs)  #Prior probability
    transition=cal_transition(obs)  #Transition probability
    temp_emission=cal_emission(train_letters)
    return prior,end_prior,transition,temp_emission

def simplified():
    emission=cal_final_emission(temp_emission,train_letters,test_letters)
    #emission=find_emission(train_letters,test_letters)
    prediction=[]
    for i in range(len(test_letters)):
        tot=[1 if test_letters[i][k][j]=='*' else 0 for k in \
               range(len(test_letters[i])) for j in range(len(test_letters[i][k]))]
        if sum(tot)<=10:
            prediction+=states[-1]
        else:    
            pos={s:emission[i][s]*prior[s] for s in states}
            prediction+=[max(pos.iteritems(), key=operator.itemgetter(1))[0]]
    return emission,prediction
#####

def hmm_ve():
         prediction = []
         max_val = 0
         pos_newdict_class = {}
         pos_prevdict_class = {}
         alpha = 0
         count = 0
         for i in range(0,len(test_letters)):
             #print test_word
             pos_prevdict_class = copy.deepcopy(pos_newdict_class)
             tot=[1 if test_letters[i][k][j]=='*' else 0 for k in \
                       range(len(test_letters[i])) for j in range(len(test_letters[i][k]))]
             if count == 0:
                 #print "YESSSSSSS"
                 #print sum(tot)
                 for pos in states:
                     
                    if sum(tot) <= 10:
                            if pos != " ":
                                pos_newdict_class[pos] = 0.0001
                                continue
                            else:
                                pos_newdict_class[pos] = 0.9
                                continue
                    else: 
                        alpha = prior[pos]*emission[i][pos]
                        pos_newdict_class[pos] = alpha    
                 count +=1        
             else:
                #print sum(tot)    
                for pos2 in states:
                        if sum(tot) <= 10:
                            if pos2 != " ":
                                pos_newdict_class[pos2] = 0.0001
                                continue
                            else:
                                pos_newdict_class[pos2] = 0.9
                                continue
                        else:
                            for pos1 in states:
                                transition_prob = transition[pos1].get(pos2)    
                                alpha = alpha + (emission[i][pos2]*transition_prob*pos_prevdict_class[pos1])
                            pos_newdict_class[pos2] = alpha*math.pow(10,23)
                            alpha = 0
             #print pos_newdict_class
             max_val = max(pos_newdict_class, key=pos_newdict_class.get)
             if pos_newdict_class[max_val] < math.pow(10,-220):
                 #print pos_newdict_class[max_val]
                 count = 0
             prediction.append(max(pos_newdict_class, key=pos_newdict_class.get))
         return prediction

def hmm_viterbi(sentence):
	#print(sentence)
	speeches = states
	number_of_words = len (sentence)
	number_of_speech = len (speeches)
	

	# create matrix for both backtracking and probability
	probability, backtrack = [ [ 0 ] * (number_of_words + 1) for x in range (number_of_speech) ], \
							 [ [ 0 ] * (number_of_words + 1) for x in range (number_of_speech) ]

	for i in range (0, number_of_speech):
		probability[ i ][ 0 ] = prior[ speeches[ i ] ] * emission[ 0 ][ speeches[ i ] ]
		backtrack[ i ][ 0 ] = ""

	initial_alpha = 0
	for word_subscript in range (1, number_of_words):
		for tag_subscript in range (0, number_of_speech):
			arg_max = -sys.maxsize
			max = -sys.maxsize
			arg_bt = ""
			final_alpha = 0
			tot = [ 1 if test_letters[ word_subscript ][ k ][ j ] == '*' else 0 for k in range (len (test_letters[ word_subscript ])) for j in range (len (test_letters[ word_subscript ][ k ])) ]
			curr_tag = speeches[ tag_subscript ]
			for all_previous_tags in range (0, number_of_speech):
				
				prev_tag = speeches[ all_previous_tags ]
				if sum(tot) <= 10:
					initial_alpha = probability[ all_previous_tags ][ word_subscript - 1 ]
					if tag_subscript == number_of_speech-1:
						max = 0.9
					else:
						max = 0.00001
					if initial_alpha > arg_max:
						arg_max = initial_alpha
						arg_bt = prev_tag
						continue

				initial_alpha = transition[ curr_tag ][ prev_tag ] * probability[ all_previous_tags ][
					word_subscript - 1 ]

				if initial_alpha < math.pow(10,-300):
					initial_alpha = emission[ word_subscript ][ speeches[tag_subscript] ] * prior[ speeches[all_previous_tags] ]
				
				if initial_alpha > arg_max:
					arg_max = initial_alpha
					arg_bt = prev_tag
				else:
					final_alpha = initial_alpha * emission[ word_subscript  ][ curr_tag ] * math.pow(10,23)
				if final_alpha > max:
					max = final_alpha
			probability[ tag_subscript ][ word_subscript ] = max
			backtrack[ tag_subscript ][ word_subscript ] = arg_bt

			# terminal step, calculate speech for last word
	end_value_probability = -1
	end_value_backtrack = ""

	for i in range (0, number_of_speech):
		tag = speeches[ i ]
		last_prob = probability[ i ][ number_of_words - 1 ] * end_prior[ tag ]

		if end_value_probability < last_prob:
			end_value_probability = last_prob
			end_value_backtrack = tag
	
			# backtrack, get best path
	end_tag = end_value_backtrack

	solution = [ end_tag ]
	for word_index in range (number_of_words - 1, 0, -1):
		prev_tag_index = speeches.index (end_tag)
		end_tag = backtrack[ prev_tag_index ][ word_index ]
		solution.append (end_tag)
	solution.reverse ()
	return solution
	

states=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R',\
        'S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j',\
        'k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1',\
        '2','3','4','5','6','7','8','9','(',')',',','.','-','!','?','"','\'',' ']
# main program

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
prior,end_prior,transition,temp_emission=train(read_data(train_txt_fname))
test_letters = load_letters(test_img_fname)
emission,prediction=simplified()
print 'Simplified: '+''.join(prediction)
prediction=hmm_ve()
print 'HMM VE: '+''.join(prediction)
print 'HMM MAP: '+''.join(hmm_viterbi(test_letters))
#prediction=hmm_ve()
#print ''.join(prediction)

#print ''.join(prediction)
## Below is just some sample code to show you how the functions above work. 
# You can delete them and put your own code here!


# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,
#  here's what "a" looks like:
#print "\n".join([ r for r in train_letters['a'] ])

# Same with test letters. Here's what the third letter of the test data
#  looks like:
#print "\n".join([ r for r in test_letters[2] ])



