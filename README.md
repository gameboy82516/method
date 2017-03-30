*Kao Notes*
=========================

```python
# 顯示矩陣m*n
def shape(A):
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols

# 取得某矩陣的列
def get_row(A, i):
    return A[i]

# 取得某矩陣的欄
def get_column(A, j):
    return [A_i[j] for A_i in A]

# 舉個例子
A = [[1,2,3],[4,5,6]]
B = [[1,2],[3,4],[5,6]]
print(shape(A))
print(get_row(A,0))
print(get_column(A,1))
```


>#### 第二組學生報告
>>#####第一個範例-向量的基本運算

```
#第一個範例-向量的基本運算

x = [1,2,3]
y = [4,5,6]
print x+y

import numpy as np
a = np.array([1, 2, 3])
b = np.array([2, 4, 6])
print a+b
```


>>#####第二個範例-計算向量夾角
```
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
```

>>#####第三個範例-比較array與matrix的差別
```
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
```

>>#####第四個範例-利用亂數建立矩陣
```
# 第四個範例-利用亂數建立矩陣

import numpy as np

a = np.random.randint(1, 10, (3, 5))

print(a)
```

>>#####第五個範例-計算矩陣行列式(Determinant)
```
# 第五個範例-計算矩陣行列式(Determinant) 

from numpy import *

a = mat([[1, 2, -1], [3, 0, 1], [4, 2, 1]])

print
linalg.det(a)
```

>>#####第六個範例-結合matplotlib製作sin的圖形
```
# 第六個範例-結合matplotlib製作sin的圖形

import numpy as np
from matplotlib import pyplot

x = np.arange(0, 10, 0.1)
y = np.sin(x)
pyplot.plot(x, y)
pyplot.show()
```

Or can be run from the command line to get a demo of what it does (and to execute the examples from the book):

```bat
python recommender_systems.py
```  

Additionally, I've collected all the [links](https://github.com/joelgrus/data-science-from-scratch/blob/master/links.md) from the book.

And, by popular demand, I made an index of functions defined in the book, by chapter and page number. 
The data is in a [spreadsheet](https://docs.google.com/spreadsheets/d/1mjGp94ehfxWOEaAFJsPiHqIeOioPH1vN1PdOE6v1az8/edit?usp=sharing), or I also made a toy (experimental) [searchable webapp](http://joelgrus.com/experiments/function-index/).

## Table of Contents

1. Introduction
2. A Crash Course in Python
3. [Visualizing Data](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/visualizing_data.py)
4. [Linear Algebra](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/linear_algebra.py)
5. [Statistics](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/statistics.py)
6. [Probability](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/probability.py)
7. [Hypothesis and Inference](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/hypothesis_and_inference.py)
8. [Gradient Descent](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/gradient_descent.py)
9. [Getting Data](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/getting_data.py)
10. [Working With Data](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/working_with_data.py)
11. [Machine Learning](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/machine_learning.py)
12. [k-Nearest Neighbors](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/nearest_neighbors.py)
13. [Naive Bayes](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/naive_bayes.py)
14. [Simple Linear Regression](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/simple_linear_regression.py)
15. [Multiple Regression](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/multiple_regression.py)
16. [Logistic Regression](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/logistic_regression.py)
17. [Decision Trees](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/decision_trees.py)
18. [Neural Networks](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/neural_networks.py)
19. [Clustering](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/clustering.py)
20. [Natural Language Processing](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/natural_language_processing.py)
21. [Network Analysis](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/network_analysis.py)
22. [Recommender Systems](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/recommender_systems.py)
23. [Databases and SQL](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/databases.py)
24. [MapReduce](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/mapreduce.py)
25. Go Forth And Do Data Science
