#!/bin/python3

import csv
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
	from main import DATASET_FILENAME, VALID_VALUES

def read_labeled_csv(filename: str, delim: str=" ") -> tuple[list[str], list[list[str]]]:
	data = csv.reader(open(filename, mode="r"), delimiter=delim)
	return next(data), [*data]

def process_data(labels: list[str], data: list[list[str]], validation_dict: dict[str, list[str]]) -> tuple[list[list[float]], list[list[str]]]:
	valid_samples = []
	invalid_samples = []
	for sample in data:
		try:
			processed_sample = []
			for i in range(len(sample)):
				if labels[i] in validation_dict:
					processed_sample.append(validation_dict[labels[i]].index(sample[i]))
				else:
					processed_sample.append(float(sample[i]))
			valid_samples.append(processed_sample)
		except:
			invalid_samples.append(sample)
	return valid_samples, invalid_samples

def process_csv(filename: str, validation_dict: dict[str, list[str]], delim: str=" "):
	labels, rawdata = read_labeled_csv(filename, delim=delim)
	valid_samples, invalid_samples = process_data(labels, rawdata, validation_dict)
	return labels, valid_samples, invalid_samples

def get_unique_values(labels: list[str], data: list[list[str]], label: str) -> tuple[list[str], list[int]]:
	return np.unique(np.array(data).T[labels.index(label),:], return_counts=True)

if __name__ == "__main__":
	labels, valid_samples, invalid_samples = process_csv(DATASET_FILENAME, VALID_VALUES, ",")
	print(f"{len(valid_samples)} valid samples loaded ({len(invalid_samples)} invalidated)")
