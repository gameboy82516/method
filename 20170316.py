# -*- coding:utf-8 -*-

print("------------Ex1---------------")
s = set()
s.add(1) # s is now { 1 }
s.add(2) # s is now { 1, 2 }
s.add(2) # s is still { 1, 2 }
x = len(s) # equals 2
y = 2 in s # equals True
z = 3 in s # equals False
print("len(x) = "+str(x))
print("2 in s"+str(y))
print("3 in s"+str(z))

print("------------Ex2---------------")

stopwords_list = ["a","an","at"] + ["I","he","Me"] + ["yet", "you"]
print("zip in stopwords_list = " + str("zip" in stopwords_list)) # False, but have to check every element
stopwords_set = set(stopwords_list)
print("zip in stopwords_set = " + str("zip" in stopwords_set)) # very fast to check

item_list = [1, 2, 3, 1, 2, 3]
num_items = len(item_list) # 6
item_set = set(item_list)
num_distinct_items = len(item_set)
distinct_item_list = list(item_set)

print("item_set = " + str(item_set)) # {1, 2, 3}
print("num_distinct_items = " + str(num_distinct_items)) # 3
print("distinct_item_list = " + str(distinct_item_list)) # [1, 2, 3]

print("------------Ex3---------------")

x=3
parity = "even" if x % 2 == 0 else "odd"
print("x = " + str(x) + ", and its parity is " + parity)

print("------------Ex4---------------")

x = 0
print("while test:")
while x < 10:
    print x, "is less than 10"
    x += 1

print("while test:")
for x in range(10):
    print x, "is less than 10"

x = [4,1,2,3]
y = sorted(x) # is [1,2,3,4], x is unchanged
print(x)
x.sort() # now x is [1,2,3,4]
print(x,y)

print("------------Ex5---------------")

even_numbers = [x for x in range(5) if x % 2 == 0] # [0, 2, 4]
squares = [x * x for x in range(5)] # [0, 1, 4, 9, 16]
even_squares = [x * x for x in even_numbers] # [0, 4, 16]
print(even_numbers,squares,even_squares)

square_dict = { x : x * x for x in range(5) } # { 0:0, 1:1, 2:4, 3:9, 4:16 }
square_set = { x * x for x in [1, -1] } # { 1 }
print(square_dict,square_set)

print("------------Ex6---------------")

pairs = [(x, y)
for x in range(10)
for y in range(10)] # 100 pairs (0,0) (0,1) ... (9,8), (9,9)
print(pairs)

print("------------Ex7---------------")
import time
def lazy_range(n):
    """a lazy version of range"""
    i = 0
    while i < n:
        #time.sleep(i/2)
        yield i
        i += 1
print("lazy_range_test:")
for i in lazy_range(10):
    print(i)


print("------------Ex8---------------")

import random
import datetime
four_uniform_randoms = [random.random() for _ in range(4)]
print(four_uniform_randoms)

random.seed(datetime) # set the seed to 10
print random.random() # 0.57140259469

up_to_ten = range(10)
random.shuffle(up_to_ten)
print up_to_ten
# [2, 5, 1, 9, 7, 3, 8, 6, 4, 0] (your results will probably be different)

my_best_friend = random.choice(["Alice", "Bob", "Charlie"]) # "Bob" for me

lottery_numbers = range(60)
winning_numbers = random.sample(lottery_numbers, 6) # [16, 36, 10, 6, 25, 9]

#從0~9中，選4次
four_with_replacement = [random.choice(range(10))
for _ in range(4)]
print four_with_replacement

print("------------Ex9---------------")

import re
print all([ # all of these are true, because
    not re.match("a", "cat"), # * 'cat' doesn't start with 'a'
    re.search("a", "cat"), # * 'cat' has an 'a' in it
    not re.search("c", "dog"), # * 'dog' doesn't have a 'c' in it
    3 == len(re.split("[ab]", "carbs")), # * split on a or b to ['c','r','s']
    "R-D-" == re.sub("[0-9]", "-", "R2D2") # * replace digits with dashes
    ]) # prints True
print(re.split("[br]","cards"))
