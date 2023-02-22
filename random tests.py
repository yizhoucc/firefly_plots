
a,n,x=[-4,3,-5,9,-1,0,2],7,3
for i in range(n):
    a[i] = abs(a[i])

# Sort the array
a = sorted(a)
print('sorted',a)
# Assign K = N - K
x = n - x

# Count number of zeros
z = a.count(0)
print('n zeros',z)

# If number of zeros if greater
if (x > n - z):
    print("-1")
    
for i in range(0, n, 2):
    if x <= 0:
        break

    # Using 2nd operation convert
    # it into one negative
    a[i] = -a[i]
    x -= 1
for i in range(n - 1, -1, -1):
    if x <= 0:
        break

    # Using 2nd operation convert
    # it into one negative
    if (a[i] > 0):
        a[i] = -a[i]
        x -= 1

# Print array
for i in range(n):
    print(a[i], end = " ")


