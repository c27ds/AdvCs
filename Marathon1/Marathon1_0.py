import numpy as np
from random import randint

# Initialize variables
n = randint(10, 40)
c = randint(1, 10)
b = int(input("Enter the number of random elements to place in tb: "))
k = 0
j = 0

# Create empty arrays for tb and sb
tb = np.zeros((n, n), dtype=int)
sb = np.zeros((n, n), dtype=int)

# Define a function to generate values based on a rule
def generate_value(x, y):
    for i in range (1,c):
        return i  # Example function


# Populate tb using random positions and the defined rule
for _ in range(b):
    x, y = randint(0, n-1), randint(0, n-1)
    tb[x, y] = generate_value(x, y)

# Update sb independently based on the same rule and count matches
non_zero_indices = np.where(tb != 0)
for i, j in zip(*non_zero_indices):
    if sb[i, j] == 0:
        calculated_value = generate_value(i, j)
        if calculated_value == tb[i, j]:  # Ensure it matches the value in tb
            sb[i, j] = calculated_value
            k += 1 
    k +=1 # Increment for each match

print("K is " + str(k))

# Verify if sb matches tb at non-zero positions in tb
if np.array_equal(sb[non_zero_indices], tb[non_zero_indices]):
    print("Yes")
else:
    print("No")
