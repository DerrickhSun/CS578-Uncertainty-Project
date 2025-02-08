import csv
import os

# csv_name is the name of the file you want to read
def csv_read(csv_name):
    result = []
    with open(csv_name, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            result.append(row)
    return result
            

# file_name is the name you want for the file
# arr is an 2d array of rows, index 0 is headers
def csv_write(file_name, arr):
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(arr)

# for getting the current path, if needed
def get_path():
    print(os.getcwd())

# example method usage
'''get_path()
test = [["name", "id"],["adam","1"],["eve","2"]]
csv_write("test.csv",test)
print(csv_read("test.csv"))'''