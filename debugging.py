# import numpy as np

# def check_inside(boolean_array1, boolean_array2):
#     # Initialize overlap and inside with boolean_array1 instead of True
#     overlap = np.logical_and(boolean_array1, boolean_array2)
#     inside = np.logical_and(boolean_array1, np.logical_not(overlap))
#     return np.all(inside)

# # test with boolean_array1 and boolean_array2
# boolean_array1 = [False, False,False, True,True, False,False, False,
#                  False, False,False, False,False, False,False, False,
#                  False, False,False, False,False, False,False, False,
#                  False, False,False, False,False, False,False, False,
#                  False, False,False, False,False, False,False, False,]

# boolean_array2 = [False, False,True, True,True, True,False, False,
#                  False, False,False, False,False, False,False, False,
#                  False, False,False, False,False, False,False, False,
#                  False, False,False, False,False, False,False, False,
#                  False, False,False, False,False, False,False, False,]

# print(check_inside(boolean_array1, boolean_array2))




# # Python program explaining 
# # logical_and() function 
# import numpy as np 
  
# # input 
# arr1 = [True, True, True, False] 
# arr2 = [True, True, True, True] 

# compare = [True, True, True, False]
# # output 
# out_arr = np.logical_and(arr1, arr2) 

# result = (compare == out_arr).all()
# print ("Output Array : ", result) 





# maskMout = [[240, 0, 0], [0, 0, 0],[240, 0, 0],[0, 0, 0], [0, 0, 0],
#             [240, 0, 0], [0, 0, 0],[240, 0, 0],[0, 0, 0], [0, 0, 0],
#             [0, 0, 0], [0, 0, 0],[0, 0, 0],[0, 0, 0], [0, 0, 0],
#             ]



maskMout = [[240, 0, 0], [0, 0, 0],[240, 0, 0],[0, 0, 0], [0, 0, 0],
            [240, 0, 0], [0, 0, 0],[240, 0, 0],[0, 0, 0], [0, 0, 0],
            [0, 0, 0], [0, 0, 0],[0, 0, 0],[0, 0, 0], [0, 0, 0],
            ]

maskMout = [item for sublist in maskMout for item in sublist]

print(maskMout)