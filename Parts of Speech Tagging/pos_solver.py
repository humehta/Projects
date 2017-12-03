###################################
# CS B551 Fall 2017, Assignment #3
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#
#
####

###### REPORT #####
#A seperate probability.py file is created where functions like emmission, transition and rest of the functions are calculated. 

#To find more accuracy for the words that never occured, we are using the technique of morphology.
#In this we take all the word and checks whether it is a digit, or eding with -like, ed, ly. If yes, we allocate the higgest probability for that perticular tag
#and the smallest to the rest of the speeches. This way it accurately predict the best possible speech for the unknown words.
#Also, we are taking the sufix of all the words and checking if that word has come for more then 70% times in the training data.
#If yes, then we assign the perticular probability or else we move ahead by giving it the noun probability.

#For code explination, we have commented each section of the algorithm for easy understanding.


####

import random
import math
from probability import *
import operator
import copy
import re
# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#

class Solver:

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        return sum(math.log(self.prob[l]) for l in range(len(label)))

    # Do the training!
    #
    def train(self, data):
        obs=[line for line,label in data]
        states=[label for line,label in data]
        #self.end_probablity : Computes the probability of end word in the sentence 
        self.prior,self.end_probability=cal_prior(states)  #Prior probability
        self.transition=cal_transition(states)  #Transition probability
        self.emission,self.suffix_emission,self.suff_constant=cal_emission(obs,states)  #Emission probability
        # Previous word emission & next word emission
        self.prev_emission,self.next_emission=cal_prev_next_emission(obs,states)
        #print self.prev_emission
        #print self.next_emission
        
    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        prediction=[]
        self.prob=[]
        for word in sentence:
            if word not in self.emission:
                prediction+=[max(self.prior.iteritems(), key=operator.itemgetter(1))[0]]
                self.prob+=[max(self.prior.iteritems(), key=operator.itemgetter(1))[1]]
            else:
                self.pos={s:self.emission[word][s]*self.prior[s] for s in self.prior}
                prediction+=[max(self.pos.iteritems(), key=operator.itemgetter(1))[0]]
                #Computes the posterior probability [P(POS|Word)]
                self.prob+=[max(self.pos.iteritems(), key=operator.itemgetter(1))[1]]
        return prediction

    def hmm_ve(self, sentence):
         prediction = []
         self.prob = []
         max_val = 0
         pos_newdict_class = {}
         pos_prevdict_class = {}
         emission_prob = 0
         alpha = 0
         count = 0
         for test_word in sentence:
             #print test_word
             pos_prevdict_class = copy.deepcopy(pos_newdict_class)
             if count == 0:
                 for pos in states:
                     if test_word not in self.emission:
                         alpha = self.prior[pos]
                     else:
                         alpha = 0
                         #print pos
                         if self.emission[test_word].has_key(pos):
                             emission_prob = self.emission[test_word].get(pos)
                             #print emission_prob
                         else:
                             emission_prob = 0.00001
                         alpha = emission_prob*self.prior[pos]
                         #print alpha
                     pos_newdict_class[pos] = alpha
                 count +=1        
             else:
                 for pos2 in states:
                     if test_word not in self.emission:
                         #print 'hello'
                         for pos1 in states:
                             transition_prob = self.transition[pos1].get(pos2)
                             alpha = alpha + (transition_prob*pos_prevdict_class[pos1])
                     else:
                         if self.emission[test_word].has_key(pos2):
                             emission_prob = self.emission[test_word].get(pos2)
                         else:
                             emission_prob = 0.00001       
                         for pos1 in states:
                             transition_prob = self.transition[pos1].get(pos2)
                             alpha = alpha + (emission_prob*transition_prob*pos_prevdict_class[pos1])
                     pos_newdict_class[pos2] = alpha
                     alpha = 0
             max_val = max(pos_newdict_class, key=pos_newdict_class.get)
             self.prob += [pos_newdict_class[max_val]]
             prediction.append(max(pos_newdict_class, key=pos_newdict_class.get))
         return prediction



        
        


    def hmm_viterbi(self, sentence):	
		speeches = states
		number_of_words = len (sentence)
		number_of_speech = len (speeches)

		# create matrix for both backtracking and probability
		probability, backtrack = [ [ 0 ] * (number_of_words) for x in range (number_of_speech) ], \
								 [ [ 0 ] * (number_of_words) for x in range (number_of_speech) ]

		for i in range (0, number_of_speech):
			#Checks is the first word is a digit
			if bool(re.compile ("\d").search (sentence[ 0 ])):
				probability[ i ][ 0 ] = self.prior[ speeches[ i ] ] * self.suff_constant[ "number" ][ speeches[ i ] ]
			#Checks is the first word ends with ly, provide the high adv probability
			elif sentence[ 0 ].endswith("ly"):
				probability[ i ][ 0 ] = self.prior[ speeches[ i ] ] * self.suff_constant[ "ly" ][ speeches[ i ] ]
			#Checks is the first word ends with ly, provide the high adjective probability
			elif sentence[ 0 ].endswith("-like"):
				probability[ i ][ 0 ] = self.prior[ speeches[ i ] ] * self.suff_constant[ "-like" ][ speeches[ i ] ]
			#Checks is the first word ends with ly, provide the high verb probability
			elif sentence[ 0 ].endswith("ed"):
				probability[ i ][ 0 ] = self.prior[ speeches[ i ] ] * self.suff_constant[ "ed" ][ speeches[ i ] ]

			elif sentence[ 0 ] not in self.emission:

				maximum_speech_count = 0
				total_count = 0
				for s in speeches:
					if sentence[0][-3:] not in self.suff_constant:
						#Give the default probability of noum
						count = self.suff_constant["default_noun"][s]
					else:
						count = self.suff_constant[sentence[0][-3:]][s]

					total_count += count
				
					if maximum_speech_count < count:
						maximum_speech_count = count

					# if speech comes more then given number of percentage
					if maximum_speech_count / total_count > 0.9:
						if sentence[0][-3:] not in self.suff_constant:
							#Provides the highest probability of noun to an unkown word
							probability[ i ][ 0 ] = self.prior[ speeches[ i ] ] * self.suff_constant[ "default_noun" ][ speeches[ i ] ]
						else:
							probability[ i ][ 0 ] = self.prior[ speeches[ i ] ] * self.suff_constant[ sentence[0][-3:] ][ speeches[ i ] ]
					else:
						#Provides the highest probability of noun to an unkown word
						probability[ i ][ 0 ] = self.prior[ speeches[ i ] ] * self.suff_constant[ "default_noun" ][ speeches[ i ] ]

			else:
				#This calculates the final alpha by multiplying the emission probability to the word with the 
				probability[ i ][ 0 ] = self.prior[ speeches[ i ] ] * self.emission[ sentence[ 0 ] ][ speeches[ i ] ]
			backtrack[ i ][ 0 ] = ""

		#This outer loop takes every observation word
		for word_subscript in range (1, number_of_words):
			#This loop takes every current hidden state which corresponds to its respective observation state
			for tag_subscript in range (0, number_of_speech):
				maximum_argument = -100000
				max = -10000
				backtrack_argument = ""
				final_alpha = 0
				initial_alpha = 0
				current_tag = speeches[ tag_subscript ]
				#This loops keeps the track of all the previous states
				for all_previous_tags in range (0, number_of_speech):
					prev_tag = speeches[ all_previous_tags ]
					#Initial_alpha checks for all the transition value (Eg: NN-NN, NN-VB etc) and the previous alpha value
					initial_alpha = self.transition[ current_tag ][ prev_tag ] * probability[ all_previous_tags ][
						word_subscript - 1 ]

					if initial_alpha > maximum_argument:
						maximum_argument = initial_alpha
						backtrack_argument = prev_tag
					
					#Checks is the first word is a digit
					elif bool(re.compile ("\d").search (sentence[ word_subscript ])):
						final_alpha = initial_alpha * self.suff_constant[ "number" ][ current_tag ]
					#Checks is the first word ends with ly, provide the high adv probability
					elif sentence[ word_subscript ].endswith("ly"):
						final_alpha = initial_alpha * self.suff_constant[ "ly" ][ current_tag ]
					#Checks is the first word ends with -like, provide the high adverb probability
					elif sentence[ word_subscript ].endswith("-like"):
						final_alpha = initial_alpha * self.suff_constant[ "-like" ][ current_tag ]
					#Checks is the first word ends with ed, provide the high verb probability
					elif sentence[ word_subscript ].endswith("ed"):
						final_alpha = initial_alpha * self.suff_constant[ "ed" ][ current_tag ]
					#Checks is the first word ends with ly, provide the high adv probability
					elif sentence[ word_subscript ] not in self.emission:
						maximum_speech_count = 0
						total_count = 0
						for s in speeches:
							if sentence[word_subscript][-3:] not in self.suff_constant:
								#Give the default probability of noun
								count = self.suff_constant["default_noun"][s]
							else:
								count = self.suff_constant[sentence[word_subscript][-3:]][s]
							total_count += count
						
							if maximum_speech_count < count:
								maximum_speech_count = count
							#This checks for the words last three character that are stored in the suff_constant dictionary
							#If the word came more then 90%, we give it probability or else we give the default dictionary and move forward
							# if speech comes more then given number of percentage
							if maximum_speech_count / total_count > 0.9:
								if sentence[word_subscript][-3:] not in self.suff_constant:
									final_alpha = initial_alpha * self.suff_constant[ "default_noun" ][ current_tag ]
								else:
									final_alpha = initial_alpha * self.suff_constant[ sentence[word_subscript][-3:] ][ current_tag ]
							else:
								final_alpha = initial_alpha * self.suff_constant[ "default_noun" ][ current_tag ]

					else:
						final_alpha = initial_alpha * self.emission[ sentence[ word_subscript ] ][ current_tag ]
					

					if final_alpha > max:
						max = final_alpha
				probability[ tag_subscript ][ word_subscript ] = max
				backtrack[ tag_subscript ][ word_subscript ] = backtrack_argument


		end_value_probability = -1
		end_value_backtrack = ""
		# terminal step, calculate speech for last word		
		for i in range (0, number_of_speech):
			tag = speeches[ i ]
			last_element_probability = end_value_probability * self.end_probability[ tag ]

			if end_value_probability < last_element_probability:
				end_value_probability = last_element_probability
				end_value_backtrack = tag

		# backtrack, get best path
		last_word_tag = end_value_backtrack

		solution = [ last_word_tag ]
		#Get every suitable tag while backtracking
		for word_index in range (number_of_words - 1, 0, -1):
			prev_tag_index = speeches.index (last_word_tag)
			last_word_tag = backtrack[ prev_tag_index ][ word_index ]
			solution.append (last_word_tag)
		solution.reverse ()
		return solution
	

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, algo, sentence):
        if algo == "Simplified":
            return self.simplified(sentence)
        elif algo == "HMM VE":
            return self.hmm_ve(sentence)
        elif algo == "HMM MAP":
            return self.hmm_viterbi(sentence)
        else:
            print "Unknown algo!"

    



