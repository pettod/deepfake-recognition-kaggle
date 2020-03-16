import os
import csv
import pandas as pd


file_old = open("/home/peter/Downloads/submission.csv")
file_new = open("/home/peter/Documents/codes/deepfake-recognition-kaggle/submission.csv")
old = csv.reader(file_old, delimiter=',')
new = csv.reader(file_new, delimiter=',')

a = []
b = []
for row in old:
    a.append(row)
for row in new:
    b.append(row)
for i in range(len(a)):
    if(a[i] != b[i]):
        print(a[i], b[i])


our_submission = pd.read_csv("submission.csv")
kaggle_submission = pd.read_csv("/home/peter/Downloads/submission.csv")
print("yes")
#sub.to_csv('submission.csv', index=False)
