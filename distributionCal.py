import os
import collections
import matplotlib.pyplot as plt
import numpy as np

X_train_orig = []
# change directory
os.chdir('C:/Users/Cheryl Zhang/Desktop/utkface-new/crop_part1')
onlyfiles = os.listdir()
Y_age =[]
for file in onlyfiles:
    Y_age.append(int(file.split("_")[0]))
print(len(Y_age))
plt.hist(Y_age, bins=100)
plt.xlabel('Age')
plt.ylabel('Counts')
plt.show()

# Calculate distribution in specific age range
a = range(15,46)
Y_age = [x for x in Y_age if x in a]
print(len(Y_age))
plt.hist(Y_age, bins=100)
plt.xlabel('Age')
plt.ylabel('Counts')
plt.show()
counter=collections.Counter(Y_age)
print(counter)
total = sum(counter.values())
print(total)
probability_mass = {k:v/total for k,v in counter.items()}
print(probability_mass)
a= 0
for key,prob in probability_mass.items():
    a = a + float(key) * float(prob)
print(a)
'''
test_path = "C:/Users/Cheryl Zhang/Desktop/c123_4"
testimages = os.listdir(test_path)
print(len(testimages))
'''