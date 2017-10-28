#The first design decision made was to remove all the non-ascii chaaracters from the tweets read from the text file.
#The next decision was to transformeach tweet into the same format by removing all punctuations and converting each tweet into lower case so that the same word written in different cases is not considered different.
#The next design decision that I made was to avoid stopwords by using the list of stopwords in the "stopwords.txt" file which increased the accuracy by some extent.
#While calculating likelihood probabilities for words in test file, if we encounter a new word for a particular location which was not seen in the training file, instaed of ignoring it, it was given a probability of occuring atleast once in the number of tweets for that particular location.


#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import sys
import heapq
import string
import math
import re

tweet_count = 0
correct = 0
count = 0

def top_word():
    for key,val in top_dict.iteritems():
            top_5 = {}
            top_list = sorted(val.values(),reverse=True)[0:5]
            print "Top 5 Associated words for "+str(key)+":"
            for i in top_list:
                for k,v in val.iteritems():
                    if i == v:
                        if top_5.has_key(k):
                            continue
                        else:
                            top_5[k] = 1
                            print k,v
            print " "
        


def classification(tweet,original_tweet,og_tweet,start):
    global count
    global file
    count += 1
    global correct
    h = []
    likelihood = 1
    for key,val in location_dict.iteritems():
        likelihood = 1
        if start == 0:
            top_dict[key] = {}
        probability_location_given_word = 0        
        for i in range(0,len(tweet)):
            
            if tweet[i] in stopwords_list:
                continue
            else:
                if val.has_key(tweet[i]):
                    prob = val.get(tweet[i])
                    top_dict[key][tweet[i]] = prob * location_occurence_newdict[key]
                    likelihood *= val[tweet[i]]
                else:
                    likelihood *= 1/float(location_occurence_dict[key])
        probability_location_given_word = math.log(likelihood) + math.log(location_occurence_newdict[key])            
        heapq.heappush(h, (probability_location_given_word,key))
    if original_tweet[0] == (heapq.nlargest(len(h), h)[0][-1]):
        correct += 1
    classify =  list(heapq.nlargest(len(h), h)[0][-1])
    classify.insert(-2,'_')
    classify = "".join(classify).upper()
    answer = classify+" "+str(og_tweet)
    file.write(answer+'\n')
    file.write(' ')



def likelihood_prob():
    for key,val in location_dict.iteritems():
        top_dict[key] = {}
        for word in val:
            count = val[word]
            prob =  round(count/float(location_occurence_dict[key]),5)
            location_dict[key][word] = prob         
    
    
def location_prior():
    for key in location_occurence_dict:
        count=location_occurence_dict[key]
        location_occurence_newdict[key]= round(count/float(tweet_count),3)



def word_occurences(tweet):
    length = len(tweet)
    location = tweet[0]
    if not location_dict.has_key(tweet[0]):
        location_dict[tweet[0]]={}
        location_occurence_dict[tweet[0]]= 1
    else:
        count=location_occurence_dict.get(tweet[0])
        location_occurence_dict[tweet[0]]= count+1
    for i in range(1,length):
        word = tweet[i]
        if word in stopwords_list:
            continue
        else:
            if location_dict[location].has_key(word):
                    count=location_dict[location].get(word)
                    location_dict[location][word] = count+1
            else:
                    location_dict[location][word] = 1




def tweet_retrive(filename,filetype,start):
    global tweet_count
    with open(filename) as f:
        ogt = f.next()
        previous=[s.translate(None, string.punctuation).lower() for s in ogt.split()]
        previous=' '.join(previous).split()
        for line in f:
            if not bool(line.strip()):
                continue
            original_tweet = line
            tweet = line.split()
            tweet = [re.sub(r'[^\x00-\x7F]+','', item) for item in tweet]
            if ',_' in tweet[0]:
                if filetype == 'input':
                    tweet_count += 1
                    word_occurences(previous)
                else:
                    if start == 0:
                        classification(previous[1:],previous,ogt,start)
                        start += 1
                    else:
                        classification(previous[1:],previous,ogt,start)
                ogt = original_tweet        
                new_tweet=[s.translate(None, string.punctuation).lower() for s in tweet]
                previous=' '.join(new_tweet).split()
            else:
                ogt = ogt + original_tweet
                new_tweet=[s.translate(None, string.punctuation).lower() for s in tweet]
                previous.extend(' '.join(new_tweet).split())
        if filetype == 'output':
                classification(previous[1:],previous,ogt,start)
        else:
                word_occurences(previous)
                tweet_count +=1
                print 'Calculating Priors!!!!'
                location_prior()
                print 'Calculating Likelihood!!!!'
                likelihood_prob()
                       


tweet_count = 0
location_dict = {}
top_dict = {}
location_occurence_dict={}
location_occurence_newdict={}
stopwords_list = []
start=0

trainfile = sys.argv[1]
testfile = sys.argv[2]
outputfile = sys.argv[3]


with open('stopwords.txt') as f: 
    for word in f:
        stopwords_list.append(word.lower().strip())
tweet_retrive(trainfile,'input',start)
file = open(outputfile,'a')
tweet_retrive(testfile,'output',start)
file.close()
top_word()

print 'Accuracy= ',str((correct/float(count))*100)+"%"


