# -*- coding:utf-8 -*-

from collections import defaultdict
#儲存不同資料的函式庫

salaries_and_tenures = [(83000, 8.7), (88000, 8.1),
(48000, 0.7), (76000, 6),
(69000, 6.5), (76000, 7.5),
(60000, 2.5), (83000, 10),
(48000, 1.9), (63000, 4.2)]

# keys are years, values are lists of the salaries for each tenure
# defaultdict(list)：裡面放的預設值是list，也就是[]，用法和dict是一樣的
# salary_by_tenure字典先存放tenure值，再附加salary值 (對應Key:value = tenure:salary)
salary_by_tenure = defaultdict(list)
for salary, tenure in salaries_and_tenures:
    salary_by_tenure[tenure].append(salary)
print ("salary_by_tenure："+str(salary_by_tenure))
print("-------------")

# keys are years, each value is average salary for that tenure
# 再建立average_salary_by_tenure字典，存放 key:value = tenure:薪水總和/(薪水長度)
# salary_by_tenure.items()代表取出salary_by_tenure逐一項字典內容
average_salary_by_tenure = {
tenure : sum(salaries) / len(salaries) for tenure, salaries in salary_by_tenure.items()
}
#print (salary_by_tenure.items())

print ("average_salary_by_tenure："+str(average_salary_by_tenure))
print("-------------")

def tenure_bucket(tenure):
    if tenure < 2:
        return "less than two"
    elif tenure < 5:
        return "between two and five"
    else:
        return "more than five"
print(tenure_bucket(5.5))
print("-------------")

# keys are tenure buckets, values are lists of salaries for that bucket
salary_by_tenure_bucket = defaultdict(list)

# 把資料做些分群，舉例來說：像病歷資料感冒人很多，但如果把年齡層給分群：幼年壯年老年，可以發現幼年老年感冒人比較多
# 資料經過處理可以看出一些資訊，但比較難的是臨界值怎麼定，幾歲以上算幼年?壯年?老年? 就要去做測試
# 寫一個很大的迴圈，去調整條件值，去找出哪一個是有效益的 (Ex:if tenure<1~100 elif tenure<2100)
for salary, tenure in salaries_and_tenures:
    bucket = tenure_bucket(tenure)
    salary_by_tenure_bucket[bucket].append(salary)

print(salary_by_tenure_bucket)
print("-------------")

# keys are tenure buckets, values are average salary for that bucket
average_salary_by_bucket = {
tenure_bucket : sum(salaries) / len(salaries)
for tenure_bucket, salaries in salary_by_tenure_bucket.iteritems()
}
print("average_salary_by_bucket：\n"+str(average_salary_by_bucket))
print("-------------")

interests = [
(0, "Hadoop"), (0, "Big Data"), (0, "HBase"), (0, "Java"),
(0, "Spark"), (0, "Storm"), (0, "Cassandra"),
(1, "NoSQL"), (1, "MongoDB"), (1, "Cassandra"), (1, "HBase"),
(1, "Postgres"), (2, "Python"), (2, "scikit-learn"), (2, "scipy"),
(2, "numpy"), (2, "statsmodels"), (2, "pandas"), (3, "R"), (3, "Python"),
(3, "statistics"), (3, "regression"), (3, "probability"),
(4, "machine learning"), (4, "regression"), (4, "decision trees"),
(4, "libsvm"), (5, "Python"), (5, "R"), (5, "Java"), (5, "C++"),
(5, "Haskell"), (5, "programming languages"), (6, "statistics"),
(6, "probability"), (6, "mathematics"), (6, "theory"),
(7, "machine learning"), (7, "scikit-learn"), (7, "Mahout"),
(7, "neural networks"), (8, "neural networks"), (8, "deep learning"),
(8, "Big Data"), (8, "artificial intelligence"), (9, "Hadoop"),
(9, "Java"), (9, "MapReduce"), (9, "Big Data")
]

from collections import Counter

words_and_counts = Counter(word
for user, interest in interests
for word in interest.lower().split())

for word, count in words_and_counts.most_common():
    if count > 1:
        print ("word,count=(" +str(word)+","+ str(count)+")")
print("-------------")

for i in [1, 2, 3, 4, 5]:
    print i # first line in "for i" block
    for j in [1, 2, 3, 4, 5]:
        print j # first line in "for j" block
        print i + j # last line in "for j" block
    print i # last line in "for i" block
print "done looping"

print("-------------")

for i in [1, 2, 3, 4, 5]:
    cmd=str(i)
    for j in [1, 2, 3, 4, 5]:
        add = cmd + "+" + str(j) + "=" +str(i + j) # first line in "for j" block
        print add # last line in "for i" block
print "done looping"

print("-------------")

def my_print(message="my default message"):
    print message

my_print("hello Kao")
my_print()
print("-------------")

def subtract(a=0, b=0):
    print(a - b)

subtract(10, 5) # returns 5
subtract(0, 5) # returns -5
subtract(b=5) # same as previous
print("-------------")

x1=10
x2=0

try:
    a=x1/x2
    print str(x1)+"/"+str(x2)+"="+str(a)
    print(a)
except ZeroDivisionError:
    print "cannot divide by zero"
    print "被除數不得為0"
print("-------------")

integer_list = [1, 2, 3]
heterogeneous_list = ["string", 0.1, True]
list_of_lists = [ integer_list, heterogeneous_list, [] ]
list_length = len(integer_list) # equals 3
list_sum = sum(integer_list) # equals 6
print list_length,list_sum
print("-------------")

x = range(10) # is the list [0, 1, ..., 9]
zero = x[0] # equals 0, lists are 0-indexed
one = x[1] # equals 1
nine = x[-1] # equals 9, 'Pythonic' for last element
eight = x[-2] # equals 8, 'Pythonic' for next-to-last element
x[0] = -1 # now x is [-1, 1, 2, 3, ..., 9]
print zero,one,nine,eight
print 'x='+str(x)
print("-------------")

first_three = x[:3] # [-1, 1, 2]
three_to_end = x[3:] # [3, 4, ..., 9]
one_to_four = x[1:5] # [1, 2, 3, 4]
last_three = x[-3:] # [7, 8, 9]
without_first_and_last = x[1:-1] # [1, 2, ..., 8]
copy_of_x = x[:] # [-1, 1, 2, ..., 9]
print first_three, three_to_end, one_to_four, last_three, without_first_and_last, copy_of_x
print("-------------")

x = [1, 2, 3]
x.extend([4, 5, 6]) # x is now [1,2,3,4,5,6]

print("extend："+str(x))

x = [1, 2, 3]
x.append(0) # x is now [1, 2, 3, 0]

print("append："+str(x))
print("-------------")

def sum_and_product(x, y):
    return (x + y),(x * y)
sp = sum_and_product(2, 3) # equals (5, 6)
s, p = sum_and_product(5, 10) # s is 15, p is 50
print(sp)
print(s,p)

empty_dict = {} # Pythonic
empty_dict2 = dict() # less Pythonic
grades = { "Joel" : 80, "Tim" : 95 } # dictionary literal
joels_grade = grades["Joel"] # equals 80
print (grades)

joels_grade = grades.get("Joel", 2) # equals 80
kates_grade = grades.get("Kate", 0) # equals 0
no_ones_grade = grades.get("No One") # default default is None
print joels_grade,kates_grade, no_ones_grade
