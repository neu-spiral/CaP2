import numpy as np
from munkres import Munkres, print_matrix
import sys

def add_virtualmachines(c,b=None):
    '''
    c is an array of shape (num of tasks, num of machines)
    b is the capacity of the machines
    creates dulicates of each machine to exress capacity
    '''
    (n,m) = c.shape
    if n<=m:
        return c,b
    
    if (b is None):
        b=n//m
        if b*m<n:
            b=b+1
    c= np.repeat(c, b, axis=1)
    return c,b
        
def computeassignment(c,b=None):
    """
    computes the assignment problem 
    returns list of elements (task, machine, cost)
    """
    (cv,b)=add_virtualmachines(c,b)
    m = Munkres()
    indexes = m.compute(cv)
    output=[]
    for row, column in indexes:
        rc=column // b
        #print("vm= ", column, "m= ",rc)
        value = c[row][rc]
        output.append((row,rc,value))
    return output
        
    

if __name__ == "__main__":
    """
    testing assignment solutions
    """
    matrix = [[5, 9, 1,10,10,10,10],
              [5, 9, 1,10,10,10,10],
              [5, 9, 1,10,10,10,10],
              [10, 3, 2,10,10,10,10],
              [10, 3, 2,10,10,10,10],
              [10, 3, 2,10,10,10,10],
              [8, 7, 4,10,10,10,10]]
    
    matrix2= [[2,1,0],
              [0,3,2],
              [1,1,0],
              [2,0,5],
              [0,2,2],
              [1,0,1],
              [2,6,0],
              [0,1,2],
              [5,1,0],
              [2,0,2]]
    
    c=np.array(matrix2)
    print(np.repeat(c, 5, axis=1))
    (cv,b)=add_virtualmachines(c)
    print(c.shape,b)
    print(cv.shape,b)
    
    m = Munkres()
    indexes = m.compute(cv)
    print_matrix(c, msg='Lowest cost through this matrix:')
    total = 0
    print(indexes)
    for row, column in indexes:
        rc=column // b
        #print("vm= ", column, "m= ",rc)
        value = c[row][rc]
        total += value
        print(f'({row}, {rc}) -> {value}')
    print(f'total cost: {total}')
    print("test")
    print(computeassignment(c))
    
        