from random import seed
from random import randrange
from csv import reader
from math import sqrt

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r', newline='') as file:
		csv_reader = reader(file, delimiter=',')
		next(csv_reader)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

def str_column_to_float(dataset, column):
    for row in dataset:
        try:
            row[column] = float(row[column].strip())
        except ValueError:
            # Empty string
            if row[column].strip('" ') == '':
                row[column] = 0.0
                # or
                pass  # ignore
            else:
                raise
                

def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup
filename = 'TRAININGDATAEDITED.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
            str_column_to_float(dataset, i)
str_column_to_int(dataset, len(dataset[0])-1)
