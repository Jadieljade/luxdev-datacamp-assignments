'''Given an array of elements find the sum of its elements'''

# wr can use the inbuilt sum() function in python
# print(sum(array)) or return sum(array) if you are defining a function

# or iterate over every item in the array


def sum_of_array(array):
    answer = 0
    for number in array:
        answer += number
    return answer


# test
print(sum_of_array([10, 16, 23, 45, 556, 6]))
