import numpy as np


def gauss_seidel(a, x, b):
	"""
	Solves given Ax = b linear system where A is coefficients matrix
	is multiplied by solution vector x to obtain vector b.
	Input matrix and vectors must be consistent.
	"""
	# Get matrix size
	n = len(a)
	
	# Arrange matrix by diagonal dominance
	a = a[:, np.argmax(abs(a), axis=1)]
	# Loop for every row of a and b simultaneously
	for i in range(n):
		# Save i'th row of b temporarily for every equation
		temp = b[i]
		# Loop for elements in A's and x's rows
		for j in range(n):
			# Skip if j = i (the x itself)
			if i != j:
				# Operation for obtaining numerator for every x
				temp -= a[i][j] * x[j]
		# Find xi with dividing temp value by xi's coefficient
		x[i] = temp / a[i][i]
	# Return new vector of x's
	return x


# Main program

# Get matrix size from user
n = int(input("Enter n (n x n): "))

# Initialize empty arrays
a = np.zeros((n, n), dtype=float)
x = []
b = []

print("Matrix A")
# Get elements of Matrix A from user
for i in range(n):
	for j in range(n):
		a[i][j] = float(input(f"Enter A[{i+1}][{j+1}]: "))
	
print("Vector b")
# Get elements of Vector b from user
for i in range(n):
	b.append(float(input(f"Enter b[{i+1}]: ")))

print("Initial Vector x")
# Get initial x values from user
for i in range(n):
	x.append(float(input(f"Enter initial x{i+1} : ")))
# Get stopping error value from user
e0 = float(input("Enter e0 error value: "))

print()
print("Coefficients matrix A = \n", a)
print("Initial vector x = \n", x)
print("Vector b = \n", b)
print()

# Initial error value and iteration number
err = 1
ino = 1
# Loop until error value is smaller than e0
while err > e0:
	errors = []
	# Save current x for iteration
	old = x.copy()
	# Obtain new x vector
	x = gauss_seidel(a, x, b)

	# Calculate error
	for j in range(n):
		if x[j] != 0:
			errors.append(abs((x[j]-old[j])/x[j]))
	err = max(errors)
	# Print x and error values for desired iteration
	print(f"iteration {ino}")
	print("------")
	print("x = ", ["{:.6f}".format(item) for item in x])
	print("Relative error: {:.6f} ({:.4%})".format(err, err))
	print()
	ino += 1

# Print final values of solution and error
for i, xi in enumerate(x):
	print(f"x{i+1} = {xi:.6f}")
print(f"Error = {err:.6f} < {e0}")
	