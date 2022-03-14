#Step1. import libraries

import numpy as np
import pandas as pd
import json
import string
import nltk
import time
import os
import re
import random
import argparse
import spacy
import neuralcoref
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import tensorflow as tf
from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertForSequenceClassification
import faiss
from functools import partial
from sklearn.model_selection import train_test_split

import re
from collections import Counter
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)

# answer type table
with open('./TriviaQuestion2NQ_Transform_Dataset/qanta_id_to_the_answer_type_most_freq_phrase_based_on_page_dict.json') as json_file:
    answer_type_dict_before_parse_tree_nq_like_test_v_3 = json.load(json_file)
answer_type_dict = answer_type_dict_before_parse_tree_nq_like_test_v_3

#Step6. Heuristics for NQlike quality checking

# Heuristic 1 remove punctuation patterns at the beginning and the end of the question [" ' ( ) , .]
def clean_marker(q):
  to_clean = "\"|\'|\(|\)|,|\."
  has_heuristic = False
  q_array = q.split()
  array_leng = len(q_array)
  while re.match(to_clean, q_array[array_leng-1]):
    q_array = q_array[:array_leng-1]
    array_leng = array_leng - 1
    has_heuristic = True

  while re.match(to_clean, q_array[0]):
    q_array = q_array[1:]
    array_leng = array_leng - 1
    has_heuristic = True
  if has_heuristic:
    q = ' '.join(q_array)
  return q

# Heuristic 2 -- name this answer type correction
def clean_answer_type(q):
  to_clean = "-- name this"
  if re.search(to_clean, q):
    start_with = "^-- name this"
    # if start with -- name this converts to which
    if re.search(start_with, q):
      q = re.sub(start_with, 'which', q)
    else:
       q = re.sub(to_clean, 'the', q)
  return q

# Heuristic 3 semicolon
def drop_after_semicolon(q):
  to_clean = ";.*"
  if re.search(to_clean, q):
    q = re.sub(to_clean, '', q)
  return q 

# Heuristic 4 remove pattern issues
def remove_pattern(q):
  to_clean = "Ã¢|Ã¢|â€‹|â€¦|â€•|â€˜ |â–º|Ã£|\(\s?\*\s?\)|\(\s?\+\s?\)|\[\s?\*\s?\]|ftp,|for 10 points|for 10 points ,|for ten points ,|for 10points ,|ftp|--for 10 points--"
  if re.search(to_clean, q):
    q = re.sub(to_clean, '', q)
  return q

# Heuristic 5 remove repetition of the subject “is this” 
def count_num_of_verbs(text, strictly = False):
  verb_tags = []
  if strictly:
    verb_tags = ['VB','VBD','VBN','VBP','VBZ']
  else:
    verb_tags = ['VB','VBD','VBG','VBN','VBP','VBZ']
  tokens = nltk.word_tokenize(text.lower())
  text = nltk.Text(tokens)
  tagged = nltk.pos_tag(text)
  counted = Counter(tag for word,tag in tagged)
  num_of_verb = 0
  for v in verb_tags:
    num_of_verb = num_of_verb + counted[v]
  return num_of_verb

def remove_rep_subject(q):
  to_clean = " is this [a-zA-Z]*\s"
  if re.search(to_clean, q):
    # the sentence has to have 1 verb at least otherwise this will not be done
    if (count_num_of_verbs(q) > 1):
      q = re.sub(to_clean, ' ', q)
  return q

# Heuristic 6 change be determiner to s possession
def remove_bd(q):
  to_clean = "( is his )|( is her )|( is its )"
  if re.search(to_clean, q):
    q = re.sub(to_clean, '\'s ', q)
  return q

# Heuristic 7 add be verb to questions without verb
def add_verb(text):
  tokens = nltk.word_tokenize(text.lower())
  text = nltk.Text(tokens)
  tagged = nltk.pos_tag(text)
  ind = 0
  for tk,tg in tagged:
    if tg == 'NN' or tg == 'NNP':
      tokens.insert(ind+1,'is')
      break
    elif tg == 'NNS' or tg == 'NNPS':
      tokens.insert(ind+1,'are')
      break
    ind = ind + 1
  return ' '.join(tokens)

def fix_no_verb(q):
  if (count_num_of_verbs(q, True) == 0):
    q = add_verb(q)
  return q

# Heuristic 8 remove repetitive be verb when there's more verbs
def remove_rbv(q):
  to_clean = "( is he )|( is she )|( is it )"
  if re.search(to_clean, q):
    if (count_num_of_verbs(q) > 1):
      q = re.sub(to_clean, ' ', q)
  return q

# Heuristic 9 First verb after which in continuous sense
def convert_fbawics(q):
  verb_tags = ['VB','VBD','VBG','VBN','VBP','VBZ']
  text = q
  tokens = nltk.word_tokenize(text.lower())
  text = nltk.Text(tokens)
  tagged = nltk.pos_tag(text)
  ind = 0
  for tk,tg in tagged:
    if tg in verb_tags:
      if tg == 'VBG':
        try:
          old_tk, old_tg = tagged[ind-1]
          if old_tg == 'NN' or old_tg == 'NNP':
            tokens[ind] = re.sub('ing','s',tokens[ind])
            q = ' '.join(tokens)
          else:
            tokens[ind] = re.sub('ing','',tokens[ind])
            q = ' '.join(tokens)
        except:
          break
        break
    else:
      break
    ind = ind + 1
  return q

# Heuristic 10 fix "name which" "identify which"
def remove_niw(q):
  to_clean = "identify which|name which"
  if re.search(to_clean, q):
    q = re.sub(to_clean, 'which', q)
  return q
  
# function counts the number of of questions with 1,2,3 words
def count_word_freq(q_lst):
  count_1 = 0
  count_2 = 0
  count_3 = 0
  for q in q_lst:
    q_array = q.split()
    if len(q_array) == 1:
      count_1 = count_1 + 1
    if len(q_array) == 2:
      count_2 = count_2 + 1
    if len(q_array) == 3:
      count_3 = count_3 + 1
  return (count_1,count_2,count_3)

# Heuristic11: convert 'this' to 'which' when no 'which' is present inside the question
def convert_this_to_which(q):
  x = q
  index = x.find('which')
  if index==-1:
    result = re.sub('this', 'which', x, 1)
    q = result
  return q

# Heuristic12: replace 'this' to 'which'+answer_type within 'this is' pattern
def deal_with_this_is_pattern(qb_id, q):
  x = q
  index = x.find('this is')
  if index!=-1:
    # adding answer type
    qb_id = str(qb_id)
    if qb_id in answer_type_dict.keys():
      answer_type = answer_type_dict[qb_id] # get the answer type from qb_id
      replacement = 'which '+answer_type
      result = re.sub('this is', replacement+' is', x, 1)
      q = result
    else:
      # answer type is not in the dict
      result = re.sub('this', 'which', x, 1)
      q = result
  return q

# Heuristic13: 'is/are' at the end of questions (after cleaning the wrong punc at the end of the sample)
def deal_with_end_be_verbs(q):
  x = q
  x = x.strip()
  if x[-3:] == ' is':
    result = x[:-3]
    q = result
  elif x[-4:] == ' are':
    result = x[:-4]
    q = result
  return q

# Heuristic14: double be/AUX(pos) verbs
def deal_with_double_AUX(q):
  x = q
  doc_dep = nlp(x)
  lemma_lst = []
  tokem_text_lst = []
  for k in range(len(doc_dep)):
    lemma_lst.append(doc_dep[k].lemma_)
    tokem_text_lst.append(doc_dep[k].text)
  if lemma_lst.count('be') == 2:
    index = lemma_lst.index('be')
    if lemma_lst[index+1] == '-PRON-' and lemma_lst[index+2] == 'be':
      # two non-conjunctional be verbs with pronoun in between
      del tokem_text_lst[index+1]
      del tokem_text_lst[index+1]
      result = " ".join(tokem_text_lst)
      q = result
    else:
      # two conjunction BE verbs or two non-conjunctional be verbs without pronoun in between
      del tokem_text_lst[index]
      result = " ".join(tokem_text_lst)
      q = result
  return q

# Heuristic15: 'which is where/why' pattern, convert 'which' to 'that' and check if no 'which' present anymore
# if so, convert 'this' to 'which'
def deal_with_WDT_BE_pattern(q):
  x = q
  index1 = x.find('which is where')
  index2 = x.find('which is why')
  if index1 != -1:
    result = re.sub('which is where', 'that is where', x)
    q = result
  elif index2 != -1:
    result = re.sub('which is why', 'that is why', x)
    q = result
  else:
    result = x
    # check if no 'which' present anymore
  index = result.find('which')
  if index==-1:
    result = re.sub('this', 'which', result, 1)
    q = result
  return q

# Heuristic16: adding 'which+answer_type' at the beginning when no WDT/WRB present
# AFTER Heuristic1: 'which' checking
# WDT tag: which/what
# WRB tag: where/why/when
def deal_with_no_WDT(qb_id, q):
  x = q
  doc_dep = nlp(x)
  tag_lst = []
  tokem_text_lst = []
  for k in range(len(doc_dep)):
    tag_lst.append(doc_dep[k].tag_)
    tokem_text_lst.append(doc_dep[k].text)
  if ('WRB' in tag_lst)!=True and ('WDT' in tag_lst)!=True:
    # adding answer type at the beginning
    qb_id = str(qb_id)
    if qb_id in answer_type_dict.keys():
      answer_type = answer_type_dict[qb_id] # get the answer type from qb_id
      result = 'which '+answer_type+' is '+x
      q = result
    else:
      print(qb_id+'is not in the frequency table!')
  return q

# Heuristic17: VERB/AUX (pos) at the beginning of the sample
def deal_with_VERB_AUX_at_beginning(qb_id, q):
  x = q
  doc_dep = nlp(x)
  pos_lst = []
  tokem_text_lst = []
  for k in range(len(doc_dep)):
    pos_lst.append(doc_dep[k].pos_)
    tokem_text_lst.append(doc_dep[k].text)
  if pos_lst[0]=='AUX' or pos_lst[0]=='VERB':
    # adding answer type at the beginning
    qb_id = str(qb_id)
    if qb_id in answer_type_dict.keys():
      answer_type = answer_type_dict[qb_id] # get the answer type from qb_id
      result = 'which '+answer_type+' '+x
      q = result
    else:
      print(qb_id+'is not in the frequency table!')
  return q

# Heuristic18: convert 'which none is' to 'what is'
# AFTER Heuristic1: 'which' checking
def deal_which_none_is(qb_id, q):
  x = q
  index = x.find('which none is')
  if index != -1:
    qb_id = str(qb_id)
    if qb_id in answer_type_dict.keys():
      answer_type = answer_type_dict[qb_id] # get the answer type from qb_id
      result = re.sub('which none is', 'which '+answer_type+' is', x)
      q = result
    else:
      print(qb_id+'is not in the frequency table!')
  return q

# Heuristic19: 'what is which' pattern
def deal_what_is_which(q):
  x = q
  index = x.find('what is which')
  if index != -1:
    result = re.sub('what is which', 'which', x)
    q = result
  return q

# quality checking for each NQlike question
def quality_check(qb_id, q):
  remove_pattern(q)
  remove_niw(q)
  clean_marker(q)
  clean_answer_type(q)
  drop_after_semicolon(q)
  remove_rep_subject(q)
  remove_bd(q)
  remove_rbv(q)
  fix_no_verb(q)
  convert_fbawics(q)
  convert_this_to_which(q)
  deal_with_this_is_pattern(qb_id, q)
  deal_with_end_be_verbs(q)
  deal_with_double_AUX(q)
  deal_with_WDT_BE_pattern(q)
  deal_with_no_WDT(qb_id, q)
  deal_with_VERB_AUX_at_beginning(qb_id, q)
  deal_which_none_is(qb_id, q)
  deal_what_is_which(q)
  return 
 
#Step7. heuristics for transforming QB to NQlike

def clean_chunk(chunk):
  # might have trailing 'and', 'but', etc
  prefixes = ['and', 'but', 'when', 'while', ',']
  punc = ',.'
  chunk = chunk.strip()
  chunk = chunk.strip(punc)
  chunk = chunk.strip()
  chunk = chunk.strip(punc)
  chunk = chunk.strip()
  
  if chunk.endswith(' '):
    chunk = chunk[:-1]
  
  for prefix in prefixes:
    if chunk.startswith(prefix+' '):
      chunk =  chunk[len(prefix)+1:]
    if chunk.endswith(' '+prefix):
      chunk = chunk[:-len(prefix)-1]
  chunk = chunk.strip()

  return chunk 

def uniques( your_string ):    
    words = your_string.split()

    seen = set()
    seen_add = seen.add

    def add(x):
        seen_add(x)  
        return x
    
    output = ' '.join( add(i) for i in words if i not in seen )
    return output

def capitalization(q):
  # capitalize each sentences after parse tree/junk/answer_type extraction and before the transformation
  q = q[0].upper()+q[1:]
  return

def remove_duplicates(q):
  words = q.split()
  for i, w in enumerate(words):
    if i >= (len(words)-1):
      continue
    w2 = words[i+1]
    w2 = re.sub('\'s', '', w2)
    if w == w2:
      words = words[:i]+words[i+1:]
  q = " ".join(words)
  return q

# BERT answer type classification
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
loaded_model = TFDistilBertForSequenceClassification.from_pretrained("./TriviaQuestion2NQ_Transform_Dataset/BERT_Classification/Aug19_answer_type_classification_model/")
def get_answer_type_group(test_sentence):
  predict_input = tokenizer.encode(test_sentence,
                                 truncation=True,
                                 padding=True,
                                 return_tensors="tf")
  tf_output = loaded_model.predict(predict_input)[0]
  tf_prediction = tf.nn.softmax(tf_output, axis=1)
  labels = ['NON_PERSON','PERSON']
  label = tf.argmax(tf_prediction, axis=1)
  label = label.numpy()
  return labels[label[0]]

def junk_last_sentence(q):
  # to make the last sentence start from the content after 'FTP's (name this/what) [Aug23: do not junk the content]
  # the content before 'FTP's merge it into previous sentence
  q_chunks = ''
  for k,v in remove_dict.items():
    index = q.find(k)
    if index!=-1:
      q_chunks = q[:index] # should merge to previous setence
      q = q[index:]
      break
  for k,v in remove_dict.items():
    q = re.sub(k, v, q)
  return q, q_chunks

def last_sent_transform(q_with_the_chunks):
  q, q_chunks = junk_last_sentence(q_with_the_chunks)
  if q.split(' ')[:2] == ['name', 'this'] or q.split(' ')[:2] == ['identify', 'this'] or q.split(' ')[:2] == ['give', 'this'] or q.split(' ')[:2] == ['name', 'the'] \
  or q.split(' ')[:2] == ['Name', 'this'] or q.split(' ')[:2] == ['Identify', 'this'] or q.split(' ')[:2] == ['Give', 'this'] or q.split(' ')[:2] == ['Name', 'the'] \
  or q.split(' ')[:2] == ['Give', 'the'] or q.split(' ')[:2] == ['give', 'the']:
    doc = nlp(q)
    tok = []
    flag=0
    for i,token in enumerate(doc[2:6]):
      if token.pos_ == 'NOUN':
        #print('Noun Token = ', token)
        tok.append(str(token))
        flag=1
      else:
        if flag:
          break
    word  = (' ').join(tok)
    
    replacement = 'which is the'
    for k,v in last_sent_word_transform_30000.items():
      if k == 'unk':
        continue
      if word in v:
        replacement = k
        break
    
    transformed_q = q.split(' ')
    transformed_q = transformed_q[2:]
    transformed_q = (' ').join(transformed_q)
    transformed_q = replacement + ' ' + transformed_q   
  elif q.split(' ')[:2] == ['name', 'these'] or q.split(' ')[:2] == ['identify', 'these'] or q.split(' ')[:2] == ['give', 'these'] \
  or q.split(' ')[:2] == ['Name', 'these'] or q.split(' ')[:2] == ['Identify', 'these'] or q.split(' ')[:2] == ['Give', 'these'] \
  or q.split(' ')[:2] == ['Give', 'the'] or q.split(' ')[:2] == ['give', 'the']:
    doc = nlp(q)
    tok = []
    flag=0
    for i,token in enumerate(doc[2:6]):
      if token.pos_ == 'NOUN':
        #print('Noun Token = ', token)
        tok.append(str(token))
        flag=1
      else:
        if flag:
          break
    word  = (' ').join(tok)
    
    replacement = 'which are the'
    for k,v in last_sent_word_transform_30000.items():
      if not k == 'unk':
        continue
      if word in v:
        replacement = k
        break
    transformed_q = q.split(' ')
    transformed_q = transformed_q[2:]
    transformed_q = (' ').join(transformed_q)
    transformed_q = replacement + ' ' + transformed_q   
  else:
    transformed_q = q
  transformed_q = q_chunks+' '+transformed_q
  # remove adjancent duplicates
  q = remove_duplicates(q)
  q = q[0].lower()+q[1:]
  return transformed_q.strip()

non_last_sent_transform_dict = {'this ':' which ', 'This ':'Which ',
 'his ':'whose ', 'His ':'Whose ',
'these ':'which ', ''
 'it ':' what ', 'its ': ' what\'s ',
 'It ':'What ', 'Its ':'What\'s ',
    'After ':''
 }

with open('./TriviaQuestion2NQ_Transform_Dataset/word_transform_dict.json', 'r') as f:
  last_sent_word_transform_30000 = json.load(f)

remove_dict = {
    'For 10 points,  ':'', 'for 10 points,  ':'',
    'For ten points,  ':'', 'for ten points,  ':'',
    'FTP,  ':'', 'ftp,  ':'',
    'For 20 points,  ':'', 'for 20 points,  ':'',
    'For 5 points,  ':'',
    'For 10 points, ':'', 'for 10 points, ':'',
    'For ten points, ':'', 'for ten points, ':'',
    'FTP, ':'', 'ftp, ':'',
    'For 20 points, ':'', 'for 20 points, ':'',
    'For 5 points,':'', 'For 10 points — ':'',
    'For 10 points , ':'', 'for 10 points , ':'',
    'For ten points , ':'', 'for ten points , ':'',
    'FTP , ':'', 'ftp , ':'',
    'For 20 points , ':'', 'for 20 points , ':'',
    'For 5 points , ':'', 
    'For 10 points ':'', 'for 10 points ':'',
    'For ten points ':'', 'for ten points ':'',
    'FTP ':'', 'ftp ':'',
    'For 20 points ':'', 'for 20 points ':'',
    'For 5 points ':''
}

def transformation_intermediate_sent(qb_id, q):
  qb_id = str(qb_id)
  answer_type_dict = answer_type_dict_before_parse_tree_nq_like_test_v_3
  # capitalize the sentences after the answer_type extraction [Aug23: and deal with no pronous cases]
  capitalization(q)

  qb_id = str(qb_id) # match the answer type from answer_type_dict
  q_orig = q
  FLAG = 0
  if qb_id in answer_type_dict.keys():
    answer_type = answer_type_dict[qb_id] # get the answer type from qb_id
    # detect if the answer_type (noun) is a person or a thing
    if answer_type in last_sent_word_transform_30000['who is the']:
      # answer_type is PERSON
      replacement_prefix = 'which'
      replacement = replacement_prefix+' '+answer_type
      # he/He/he's/He's/his/His/who/Who/whose/Whose

      for k in ['He ', 'Who ', 'She ']:
        q = re.sub(k, replacement+' ', q, 1)
        if not q_orig == q:
          FLAG = 1
          break
      if FLAG:
        return steps_before_return(q)
      for k in ['This ']:
        q = re.sub(k, 'Which ', q, 1)
        if not q_orig == q:
          FLAG = 1
          break
      if FLAG:
        return steps_before_return(q)
      for k in [' he ', ' who ', ' she ', ' him ']:
        q = re.sub(k, ' '+replacement+' ', q, 1)
        if not q_orig == q:
          FLAG = 1
          break
      if FLAG:
        return steps_before_return(q)
      for k in [' this ']:
        q = re.sub(k, ' '+' which ', q, 1)
        if not q_orig == q:
          FLAG = 1
          break
      if FLAG:
        return steps_before_return(q)          
      for k in ['He\'s ', 'His ', 'Whose ', 'She\'s ', 'Her ']:
        q = re.sub(k, replacement+'\'s'+' ', q, 1)   
        if not q_orig == q:
          FLAG = 1
          break
      if FLAG:
        return steps_before_return(q)     
      for k in [' he\'s ', ' his ', ' whose ', ' she\'s ', ' her ']:
        q = re.sub(k, ' '+replacement+'\'s'+' ', q, 1)
        if not q_orig == q:
          FLAG = 1
          break
      if FLAG:
        return steps_before_return(q)
    # answer type is not in the last_sent_word_transform_30000 dictionary
    else:
      # classified as PERSON by BERT
      classification_output = get_answer_type_group(answer_type)
      if classification_output == 'PERSON':
        # answer_type is PERSON
        replacement_prefix = 'which'
        replacement = replacement_prefix+' '+answer_type
        # he/He/he's/He's/his/His/who/Who/whose/Whose
        for k in ['He ', 'Who ', 'She ']:
          q = re.sub(k, replacement+' ', q, 1)
          if not q_orig == q:
            FLAG = 1
            break
        if FLAG:
          return steps_before_return(q) 
        for k in ['This ']:
          q = re.sub(k, 'Which ', q, 1)      
          if not q_orig == q:
            FLAG = 1
            break
        if FLAG:
          return steps_before_return(q) 
        for k in [' he ', ' who ', ' she ', ' him ']:
          q = re.sub(k, ' '+replacement+' ', q, 1)
          if not q_orig == q:
            FLAG = 1
            break
        if FLAG:
          return steps_before_return(q) 
        for k in [' this ']:
          q = re.sub(k, ' which ', q, 1)
          if not q_orig == q:
            FLAG = 1
            break
        if FLAG:
          return steps_before_return(q) 
        for k in ['He\'s ', 'His ', 'Whose ', 'She\'s ', 'Her ']:
          q = re.sub(k, replacement+'\'s'+' ', q, 1)
          if not q_orig == q:
            FLAG = 1
            break
        if FLAG:
          return steps_before_return(q) 
        for k in [' he\'s ', ' his ', ' whose ', ' she\'s ', ' her ']:
          q = re.sub(k, ' '+replacement+'\'s'+' ', q, 1)
          if not q_orig == q:
            FLAG = 1
            break
        if FLAG:
          return steps_before_return(q) 
      else:
        # answer_type is a thing 
        replacement_prefix = 'which'
        replacement = replacement_prefix+' '+answer_type
        # swap in with the replacement
        # what/What/what's/What's/it/It/it's/It's/its/Its -> what/What+replacement
        for k in ['What ', 'It ']:
          q = re.sub(k, replacement+' ', q, 1)
          if not q_orig == q:
            FLAG = 1
            break
        if FLAG:
          return steps_before_return(q) 
        for k in ['This ']:
          q = re.sub(k, 'Which ', q, 1)
          if not q_orig == q:
            FLAG = 1
            break
        if FLAG:
          return steps_before_return(q) 
        for k in [' what ', ' it ']:
          q = re.sub(k, ' '+replacement+' ', q, 1)
          if not q_orig == q:
            FLAG = 1
            break
        if FLAG:
          return steps_before_return(q) 
        for k in [' this ']:
          q = re.sub(k, ' which ', q, 1)         
          if not q_orig == q:
            FLAG = 1
            break
        if FLAG:
          return steps_before_return(q) 
        for k in ['What\'s ', 'Its ', 'It\'s ']:
          q = re.sub(k, replacement+'\'s'+' ', q, 1)
          if not q_orig == q:
            FLAG = 1
            break
        if FLAG:
          return steps_before_return(q) 
        for k in [' what\'s ', ' its ', ' it\'s ']:
          q = re.sub(k, ' '+replacement+'\'s'+' ', q, 1)
          if not q_orig == q:
            FLAG = 1
            break
        if FLAG:
          return steps_before_return(q) 
  else:
      for k,v in non_last_sent_transform_dict.items():
        q = re.sub(' '+k, ' '+v, q, 1)
        if q.startswith(k):
          q = v + q[len(k):]
  return steps_before_return(q)

def steps_before_return(q):
  # remove adjancent duplicates
  q = remove_duplicates(q)
  q = q[0].lower()+q[1:]
  return q.strip()

def deal_with_no_pronouns_cases(qb_id, q):
  qb_id = str(qb_id)
  # input: questions after the parse tree steps and before transformation
  q = q[0].lower()+q[1:]

  question_test = nlp(q)
  pronouns_tags = {"PRON", "WDT", "WP", "WP$", "WRB", "VEZ"}
  # check whether there are any pronouns or not in the sentence q
  flag = True
  for token in question_test:
    if token.tag_ in pronouns_tags:
      flag = False
      break
  
  if flag == True:
    # no pronouns in the question

    # check wether answer type is singular or plural
    answer_type_dict = answer_type_dict_before_parse_tree_nq_like_test_v_3
    answer_type = answer_type_dict[qb_id]
    processed_text = nlp(answer_type)
    lemma_tags = {"NNS", "NNPS"}

    sigular_plural_flags = True # singular
    for token in processed_text:
      if token.tag_ == 'NNPS':
        sigular_plural_flags = False # plural
        break
    
    # check if the first toke is VERB
    if question_test[0].pos_ == 'VERB' and question_test[1].pos_ != 'PART' and question_test[2].pos_ != 'AUX':
      replacement = 'which '+answer_type+' '
      q = replacement+q
    else:
      if sigular_plural_flags == False:
        # plural
        replacement = 'which '+answer_type+' are '
        q = replacement+q  
      else:
        # singular
        replacement = 'which '+answer_type+' is '
        q = replacement+q
  # capitalize the first letter of each sentence
  q = q[0].upper()+q[1:]
  return

# transformation from one QB question to a list of NQlike
def qb_nq_transformation(qb_id, q):
  # parse tree
  qb_id = str(qb_id)
  nq_like_questions = []

  sample = q.strip()
  sample = sample.strip('.')
  doc = nlp(sample)
  seen = set() # keep track of covered words
  # Find coref clusters
  clusters = doc._.coref_clusters
  # Breakdown sentences using Parse Trees
  chunks = []
  for sent in doc.sents:
      conj_heads = [cc for cc in sent.root.children if cc.dep_ == 'conj']
      advcl_heads = [cc for cc in sent.root.children if cc.dep_ == 'advcl']
      #print('Conjuction Heads found :', conj_heads)
      #print('Advcl Heads found :', advcl_heads)
    
      heads = conj_heads + advcl_heads
      for head in heads:
          words = [ww for ww in head.subtree]
          for word in words:
              seen.add(word)

          chunk = (' '.join([ww.text for ww in words]))
          chunks.append( (head.i, chunk) )

      unseen = [ww for ww in sent if ww not in seen]
      chunk = ' '.join([ww.text for ww in unseen])
      chunks.append( (sent.root.i, chunk) )
  
  # Sort the chunks based on word index to ensure first sentences formed come first
  chunks = sorted(chunks, key=lambda x: x[0])
  
  # Ensure no sentences aren't too small
  if len(chunks)>1:
    for idx in range(1, len(chunks)):
      try:
        curr_i, curr_chunk = chunks[idx]
      except:
        #print('idx=',idx)
        #print('chunk len = ', len(chunks))
        raise NotImplementedError
      if len(curr_chunk.split()) < 8 or (curr_chunk.split()[0] in ['after']):
        #print('\nFound a small sent!\n')
        last_i, last_chunk = chunks[idx-1]
        last_chunk = last_chunk + ' ' + curr_chunk
        chunks[idx-1] = (last_i, last_chunk)
        del chunks[idx]
      if (idx+1)>=len(chunks):
        break
    curr_i, curr_chunk = chunks[0]
    if len(curr_chunk.split()) < 8 and len(chunks)>1:
      #print('\nFound a small pre-sent!\n')
      last_i, next_chunk = chunks[1]
      curr_chunk = curr_chunk + ' ' + next_chunk
      chunks[0] = (last_i, curr_chunk)
      del chunks[1]    
  
  # Clean each sentence of trailing and, comma etc
  for i in range(len(chunks)):
    id,chunk = chunks[i]
    chunk = clean_chunk(chunk)
    chunks[i] = (id, chunk)
    
  
  # Coreference subsitution
  pronoun_list = ['he', 'she', 'his', 'her', 'its']
  if len(chunks)>1:
    for i in range(1, len(chunks)):
      curr_i, curr_chunk = chunks[i]
      chunk_doc = nlp(curr_chunk)
      for id, w in enumerate(chunk_doc[:3]):
        #print('Word in chunk doc ', w, '->',w.tag_)
        if w.tag_ in ['NN', 'NNP', 'NNS', 'NNPS']:
          continue
        rep = w.text
        for cluster in clusters:
          #print('Noun chunks: ', cluster[0], '->', [x for x in cluster[0].noun_chunks])
          if (len([x for x in cluster[0].noun_chunks]) > 0) and (str(cluster[0]).lower() not in pronoun_list):
            match_cluster = [str(cc) for cc in cluster]
            #print(match_cluster)
            if w.text in match_cluster:
              rep = match_cluster[0]
              if w.text.lower() in ['his', 'her', 'its', 'it\'s']:
                rep += '\'s'
              #print(f'Found {w} in cluster!!!')
              #print('Replaceing with ', match_cluster[0])
              break
        if not w.text == rep:
          replacement_list = [str(c) for c in chunk_doc] 
          replacement_list[id] = rep
          curr_chunk = (' ').join(replacement_list)
          chunks[i] = (curr_i, curr_chunk)
        else:
          curr_chunk = '' + curr_chunk


  #print('\033[1m'+'Different nq like statements: (after 2nd breakdown):')
  for ii, chunk in chunks:
    # with the same qid
    nq_like_questions.append(chunk)
  for i in range(len(nq_like_questions)):
    # check if no pronouns in the question
    deal_with_no_pronouns_cases(qb_id, nq_like_questions[i])
    if i == len(nq_like_questions)-1:
      # last sent transformation
      nq_like_questions[i] = last_sent_transform(nq_like_questions[i])
      quality_check(qb_id, nq_like_questions[i])
    else:
      # intermediate sent transformation
      nq_like_questions[i] = transformation_intermediate_sent(qb_id, nq_like_questions[i])
      quality_check(qb_id, nq_like_questions[i])
  # return a NQlike list from one qb question
  return nq_like_questions
  
def answer_type_classifier_training():
    #No need to rerun the answer type classifier to replicate results as we are providing checkpoints for the same
    #the checkpoints are already provided in the corresponding folder
    
    #A PERSON: 
    #     replace 'he/she/who/him' and 'He/She/Who/Him' with 'which + answer_type + is/are'
    #     replace 'his/whose/she's/he's' and 'His/Whose/She's/He's' with 'which + answer_type's'
    
    #A THING: 
    #     replace 'it/this/these' and 'It/This/These' with 'which + answer_type + is/are'
    #     replace 'it's' and 'It's' with 'which + answer_type's'
    
    
    #manually annotated
    with open('./word_transform_dict.json', 'r') as f:
      last_sent_word_transform_30000 = json.load(f)
    
    person_list = []
    label_list = []
    for v in last_sent_word_transform_30000['who is the']:
      person_list.append(v)
      label_list.append('PERSON')
    
    non_person_list = []
    for v in last_sent_word_transform_30000['which is the']:
      non_person_list.append(v)
      label_list.append('NON-PERSON')
    
    for v in last_sent_word_transform_30000['what is the']:
      non_person_list.append(v)
      label_list.append('NON-PERSON')
    
    my_answer_type_list = person_list+non_person_list
    label_list = label_list[:len(my_answer_type_list)]
    # convert lists to dataframe
    zippedList =  list(zip(label_list, my_answer_type_list))
    classification_df = pd.DataFrame(zippedList, columns=['label','answer_type'])
    
    LE = LabelEncoder()
    classification_df['label'] = LE.fit_transform(classification_df['label'])
    classification_df.head()
    
    groups = classification_df['answer_type'].values.tolist()
    labels = classification_df['label'].tolist()
    
    training_sentences, validation_sentences, training_labels, validation_labels = train_test_split(groups, labels, test_size=.2)
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    tokenizer([training_sentences[0]], truncation=True,
                                padding=True, max_length=128)
    train_encodings = tokenizer(training_sentences,
                                truncation=True,
                                padding=True)
    val_encodings = tokenizer(validation_sentences,
                                truncation=True,
                                padding=True)
    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        training_labels
    ))
    
    val_dataset = tf.data.Dataset.from_tensor_slices((
        dict(val_encodings),
        validation_labels
    ))
    
    model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',num_labels=2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5, epsilon=1e-08)
    model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])
    model.fit(train_dataset.shuffle(100).batch(16),
              epochs=3,
              batch_size=16,
              validation_data=val_dataset.shuffle(100).batch(16))
    
    #save the checkpoint
    model.save_pretrained("./TriviaQuestion2NQ_Transform_Dataset/BERT_Classification/Aug19_answer_type_classification_model/")
  
  
#Step5. Main

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Apply heuristic functions")
  parser.add_argument('--limit', type=int, default=20,help="Limit of number of QB questions input")
  parser.add_argument('--qb_path', type=str, default='./TriviaQuestion2NQ_Transform_Dataset/qb_train_with_contexts_lower_nopunc_debug_Feb24.json',
                      help="path of the qb dataset")
  parser.add_argument('--save_result', type=bool, default=True, help="Save NQlike questions with corresponding contexts")
  parser.add_argument('--save_only_NQlike_questions', type=bool, default=False, help="Only Save the NQlike outputs")
  parser.add_argument('--answer_type_classifier', type=bool, default=False, help="Retrain the answer type classifier from scratch")
  args = parser.parse_args()
  # Load dataset
  qb_path = args.qb_path
  limit = args.limit
  qb_df = None
  if limit > 0:
    qb_df = pd.read_json(qb_path, lines=True, orient='records',nrows=limit)
  else:
    qb_df = pd.read_json(qb_path, lines=True, orient='records')

  qb_questions_input = qb_df['question'].values
  qb_id_input = qb_df['qanta_id'].values
  # transformation
  nq_like_questions_transformation_results = []
  for i in range(len(qb_questions_input)):
    # transform single QB
    q = qb_questions_input[i]
    qb_id = qb_id_input[i]
    nq_like_questions_lst = qb_nq_transformation(qb_id, q)
    nq_like_questions_transformation_results.append(nq_like_questions_lst)
  print(nq_like_questions_transformation_results)
  
  # save to json as a dataframe
  if args.save_result:
    nq_like_df = {
      'qanta_id':[],
      'question':[],
      'answer':[],
      'char_spans':[],
      'context':[]
    }
    for i in range(len(nq_like_questions_transformation_results)):
      assert len(nq_like_questions_transformation_results)==len(qb_df)
      nqlist = nq_like_questions_transformation_results[i]
      for j in range(len(nqlist)):
        nq_like_df['qanta_id'].append(qb_df.iloc[i]['qanta_id'])
        nq_like_df['question'].append(nqlist[j])
        nq_like_df['answer'].append(qb_df.iloc[i]['answer'])
        nq_like_df['char_spans'].append(qb_df.iloc[i]['char_spans'])
        nq_like_df['context'].append(qb_df.iloc[i]['context'])
    new_nqlike = pd.DataFrame(nq_like_df)
    new_nqlike.to_json('./TriviaQuestion2NQ_Transform_Dataset/nq_like_questions_train_with_contexts.json', lines=True, orient='records')
  # only save the NQLike questions
  if args.save_only_NQlike_questions:
    with open('./TriviaQuestion2NQ_Transform_Dataset/nqlike_questions_outputs.txt', 'w') as f:
      for nqlike_lists in nq_like_questions_transformation_results:
        for nqlike in nqlike_lists:
          f.write(nqlike + "\n")
          
  # retraining the answer type classifier
  if args.answer_type_classifier:
      print("retraining the answer type classifier from scratch......")
      answer_type_classifier_training()
