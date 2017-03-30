# -*- coding:utf-8 -*-

from __future__ import division  # want 3 / 2 == 1.5
import re, math, random  # regexes, math functions, random numbers
import matplotlib.pyplot as plt  # pyplot
from collections import defaultdict, Counter
from functools import partial


#
# functions for working with vectors
#

print("----------------Ch.3---------------------")

def vector_add(v, w):
    """adds two vectors componentwise"""
    return [v_i + w_i for v_i, w_i in zip(v, w)]


def vector_subtract(v, w):
    """subtracts two vectors componentwise"""
    return [v_i - w_i for v_i, w_i in zip(v, w)]


def vector_sum(vectors):
    return reduce(vector_add, vectors)


def scalar_multiply(c, v):
    return [c * v_i for v_i in v]


# this isn't right if you don't from __future__ import division
def vector_mean(vectors):
    """compute the vector whose i-th element is the mean of the
    i-th elements of the input vectors"""
    n = len(vectors)
    return scalar_multiply(1 / n, vector_sum(vectors))


def dot(v, w):
    """v_1 * w_1 + ... + v_n * w_n"""
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def sum_of_squares(v):
    """v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)


def magnitude(v):
    return math.sqrt(sum_of_squares(v))


def squared_distance(v, w):
    return sum_of_squares(vector_subtract(v, w))


def distance(v, w):
    return math.sqrt(squared_distance(v, w))


#
# functions for working with matrices
#

def shape(A):
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols


def get_row(A, i):
    return A[i]


def get_column(A, j):
    return [A_i[j] for A_i in A]


def make_matrix(num_rows, num_cols, entry_fn):
    """returns a num_rows x num_cols matrix
    whose (i,j)-th entry is entry_fn(i, j)"""
    return [[entry_fn(i, j) for j in range(num_cols)]
            for i in range(num_rows)]


def is_diagonal(i, j):
    """1's on the 'diagonal', 0's everywhere else"""
    return 1 if i == j else 0


identity_matrix = make_matrix(5, 5, is_diagonal)

#          user 0  1  2  3  4  5  6  7  8  9
#
friendships = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # user 0
               [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # user 1
               [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # user 2
               [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],  # user 3
               [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # user 4
               [0, 0, 0, 0, 1, 0, 1, 1, 0, 0],  # user 5
               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # user 6
               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # user 7
               [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],  # user 8
               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]  # user 9


#####
# DELETE DOWN
#


def matrix_add(A, B):
    if shape(A) != shape(B):
        raise ArithmeticError("cannot add matrices with different shapes")

    num_rows, num_cols = shape(A)

    def entry_fn(i, j): return A[i][j] + B[i][j]

    return make_matrix(num_rows, num_cols, entry_fn)


def make_graph_dot_product_as_vector_projection(plt):
    v = [2, 1]
    w = [math.sqrt(.25), math.sqrt(.75)]
    c = dot(v, w)
    vonw = scalar_multiply(c, w)
    o = [0, 0]

    plt.arrow(0, 0, v[0], v[1],
              width=0.002, head_width=.1, length_includes_head=True)
    plt.annotate("v", v, xytext=[v[0] + 0.1, v[1]])
    plt.arrow(0, 0, w[0], w[1],
              width=0.002, head_width=.1, length_includes_head=True)
    plt.annotate("w", w, xytext=[w[0] - 0.1, w[1]])
    plt.arrow(0, 0, vonw[0], vonw[1], length_includes_head=True)
    plt.annotate(u"(vâ¢w)w", vonw, xytext=[vonw[0] - 0.1, vonw[1] + 0.1])
    plt.arrow(v[0], v[1], vonw[0] - v[0], vonw[1] - v[1],
              linestyle='dotted', length_includes_head=True)
    plt.scatter(*zip(v, w, o), marker='.')
    plt.axis('equal')
    plt.show()

A = [[1,2,3],[4,5,6]]
B = [[1,2],[3,4],[5,6]]
print(shape(A))
print(get_row(A,0))
print(get_column(A,1))

def myfun(i,j):
    return 1

print(make_matrix(3,5,myfun))

print(make_graph_dot_product_as_vector_projection(plt))

print("----------------Ch.3-學生補充舉例---------------------")

#第一個範例-向量的基本運算

x = [1,2,3]
y = [4,5,6]
print (x+y)

import numpy as np
a = np.array([1, 2, 3])
b = np.array([2, 4, 6])
print (a+b)

#第二個範例-計算向量夾角

import numpy as np
a = np.array([1,3,2])
b = np.array([-2,1,-1])

la =np.sqrt(a.dot(a))
lb =np.sqrt(b.dot(b))
print("----計算向量長度----")
print(la,lb)

cos_angle = a.dot(b)/(la*lb)
print("----計算cos----")
print(cos_angle)

angle = np.arccos(cos_angle)
print("----計算夾角(單位為π)----")
print(angle)

angle2=angle*360/2/np.pi
print("----轉換單位為角度----")
print(angle2)

# 第三個範例-比較array與matrix的差別

import numpy as np

a = np.array([[3, 4], [2, 3]])
b = np.array([[1, 2], [3, 4]])
c = np.mat([[3, 4], [2, 3]])
d = np.mat([[1, 2], [3, 4]])
e = np.dot(a, b)
f = np.dot(c, d)
print("----乘法運算----")
print(a * b)
print(c * d)
print("----矩陣相乘----")
print(e)
print(f)

# 第四個範例-利用亂數建立矩陣

import numpy as np

a = np.random.randint(1, 10, (3, 5))

print(a)

# 第五個範例-計算矩陣行列式(Determinant)

from numpy import *

a = mat([[1, 2, -1], [3, 0, 1], [4, 2, 1]])

print
linalg.det(a)

# 第六個範例-結合matplotlib製作sin的圖形

import numpy as np
from matplotlib import pyplot

x = np.arange(0, 10, 0.1)
y = np.sin(x)
pyplot.plot(x, y)
pyplot.show()

print("----------------Ch.4---------------------")

#from __future__ import division
from collections import Counter
from linear_algebra import sum_of_squares, dot
import math

num_friends = [100, 49, 41, 40, 25, 21, 21, 19, 19, 18, 18, 16, 15, 15, 15, 15, 14, 14, 13, 13, 13, 13, 12, 12, 11, 10,
               10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
               9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6,
               6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4,
               4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
               3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1, 1]


def make_friend_counts_histogram(plt):
    friend_counts = Counter(num_friends)
    xs = range(101)
    ys = [friend_counts[x] for x in xs]
    plt.bar(xs, ys)
    plt.axis([0, 101, 0, 25])
    plt.title("Histogram of Friend Counts")
    plt.xlabel("# of friends")
    plt.ylabel("# of people")
    plt.show()


num_points = len(num_friends)  # 204

largest_value = max(num_friends)  # 100
smallest_value = min(num_friends)  # 1

sorted_values = sorted(num_friends)
smallest_value = sorted_values[0]  # 1
second_smallest_value = sorted_values[1]  # 1
second_largest_value = sorted_values[-2]  # 49


# this isn't right if you don't from __future__ import division
def mean(x):
    return sum(x) / len(x)


def median(v):
    """finds the 'middle-most' value of v"""
    n = len(v)
    sorted_v = sorted(v)
    midpoint = n // 2

    if n % 2 == 1:
        # if odd, return the middle value
        return sorted_v[midpoint]
    else:
        # if even, return the average of the middle values
        lo = midpoint - 1
        hi = midpoint
        return (sorted_v[lo] + sorted_v[hi]) / 2

# 取出百分位數的值
def quantile(x, p):
    """returns the pth-percentile value in x"""
    p_index = int(p * len(x))
    return sorted(x)[p_index]

# 取出眾數的值
def mode(x):
    """returns a list, might be more than one mode"""
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.iteritems()
            if count == max_count]


# "range" already means something in Python, so we'll use a different name
def data_range(x):
    return max(x) - min(x)


def de_mean(x):
    """translate x by subtracting its mean (so the result has mean 0)"""
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]

# 取出變異數的值
def variance(x):
    """assumes x has at least two elements"""
    n = len(x)
    deviations = de_mean(x)
    return sum_of_squares(deviations) / (n - 1)

# 取出標準差的值：變異數開根號
def standard_deviation(x):
    return math.sqrt(variance(x))


def interquartile_range(x):
    return quantile(x, 0.75) - quantile(x, 0.25)


####
#
# CORRELATION
#
#####

daily_minutes = [1, 68.77, 51.25, 52.08, 38.36, 44.54, 57.13, 51.4, 41.42, 31.22, 34.76, 54.01, 38.79, 47.59, 49.1,
                 27.66, 41.03, 36.73, 48.65, 28.12, 46.62, 35.57, 32.98, 35, 26.07, 23.77, 39.73, 40.57, 31.65, 31.21,
                 36.32, 20.45, 21.93, 26.02, 27.34, 23.49, 46.94, 30.5, 33.8, 24.23, 21.4, 27.94, 32.24, 40.57, 25.07,
                 19.42, 22.39, 18.42, 46.96, 23.72, 26.41, 26.97, 36.76, 40.32, 35.02, 29.47, 30.2, 31, 38.11, 38.18,
                 36.31, 21.03, 30.86, 36.07, 28.66, 29.08, 37.28, 15.28, 24.17, 22.31, 30.17, 25.53, 19.85, 35.37, 44.6,
                 17.23, 13.47, 26.33, 35.02, 32.09, 24.81, 19.33, 28.77, 24.26, 31.98, 25.73, 24.86, 16.28, 34.51,
                 15.23, 39.72, 40.8, 26.06, 35.76, 34.76, 16.13, 44.04, 18.03, 19.65, 32.62, 35.59, 39.43, 14.18, 35.24,
                 40.13, 41.82, 35.45, 36.07, 43.67, 24.61, 20.9, 21.9, 18.79, 27.61, 27.21, 26.61, 29.77, 20.59, 27.53,
                 13.82, 33.2, 25, 33.1, 36.65, 18.63, 14.87, 22.2, 36.81, 25.53, 24.62, 26.25, 18.21, 28.08, 19.42,
                 29.79, 32.8, 35.99, 28.32, 27.79, 35.88, 29.06, 36.28, 14.1, 36.63, 37.49, 26.9, 18.58, 38.48, 24.48,
                 18.95, 33.55, 14.24, 29.04, 32.51, 25.63, 22.22, 19, 32.73, 15.16, 13.9, 27.2, 32.01, 29.27, 33, 13.74,
                 20.42, 27.32, 18.23, 35.35, 28.48, 9.08, 24.62, 20.12, 35.26, 19.92, 31.02, 16.49, 12.16, 30.7, 31.22,
                 34.65, 13.13, 27.51, 33.2, 31.57, 14.1, 33.42, 17.44, 10.12, 24.42, 9.82, 23.39, 30.93, 15.03, 21.67,
                 31.09, 33.29, 22.61, 26.89, 23.48, 8.38, 27.81, 32.35, 23.84]

# 取出共變異數
def covariance(x, y):
    n = len(x)
    return dot(de_mean(x), de_mean(y)) / (n - 1)

# 判別正向關或負相關的相關程度
def correlation(x, y):
    stdev_x = standard_deviation(x)
    stdev_y = standard_deviation(y)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(x, y) / stdev_x / stdev_y
    else:
        return 0  # if no variation, correlation is zero


outlier = num_friends.index(100)  # index of outlier

num_friends_good = [x
                    for i, x in enumerate(num_friends)
                    if i != outlier]

daily_minutes_good = [x
                      for i, x in enumerate(daily_minutes)
                      if i != outlier]

if __name__ == "__main__":
    print "num_points", len(num_friends)
    print "largest value", max(num_friends)
    print "smallest value", min(num_friends)
    print "second_smallest_value", sorted_values[1]
    print "second_largest_value", sorted_values[-2]
    print "mean(num_friends)", mean(num_friends)
    print "median(num_friends)", median(num_friends)
    print "quantile(num_friends, 0.10)", quantile(num_friends, 0.10)
    print "quantile(num_friends, 0.25)", quantile(num_friends, 0.25)
    print "quantile(num_friends, 0.75)", quantile(num_friends, 0.75)
    print "quantile(num_friends, 0.90)", quantile(num_friends, 0.90)
    print "mode(num_friends)", mode(num_friends)
    print "data_range(num_friends)", data_range(num_friends)
    print "variance(num_friends)", variance(num_friends)
    print "standard_deviation(num_friends)", standard_deviation(num_friends)
    print "interquartile_range(num_friends)", interquartile_range(num_friends)

    print "covariance(num_friends, daily_minutes)", covariance(num_friends, daily_minutes)
    print "correlation(num_friends, daily_minutes)", correlation(num_friends, daily_minutes)
    print "correlation(num_friends_good, daily_minutes_good)", correlation(num_friends_good, daily_minutes_good)

print("----------------Ch.4-學生補充舉例---------------------")

import pandas as pd
import numpy as np

num_friends = pd.Series([100,49,41,40,25,21,21,19,19,18,18,16,15,15,15,15,14,14,
               13,13,13,13,12,12,11,10,10,10,10,10,10,10,10,10,10,10,
               10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,
               8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,
               6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,
               5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
               4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2
                ,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

num_newFriends =  pd.Series(np.sort(np.random.binomial(203,0.06,204))[::-1]) #用A series去建立B series
df_friendsGroup = pd.DataFrame({"A":num_friends,"B":num_newFriends}) #將兩張series合成為一個DataFrame

