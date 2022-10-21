#!/bin/python3
# Author: Timothy Walker <james.and.tisa@gmail.com>
# Usage: python3 main.py

import numpy as np
from random import shuffle
from sklearn.svm import SVC
from sklearn import metrics
from data_validation import process_csv

DATASET_FILENAME = "healthcare-dataset-stroke-data.csv"
VALID_VALUES = {
	"gender": ["Male","Female"],
	"hypertension": ["0","1"],
	"heart_disease": ["0","1"],
	"ever_married": ["Yes","No"],
	"work_type": ["Govt_job","Private","Self-employed","children","Never worked"],
	"Residence_type": ["Urban","Rural"],
	"smoking_status": ["smokes","formerly smoked","never smoked","Unknown"],
	"stroke": ["0","1"]
}

def train_and_test(samples, training_size, kernel="rbf"):
	shuffle(samples)
	cutoff = max(training_size, 1 + [data[-1] for data in samples].index(1))
	samples = np.array(samples)
	training_features = samples[:cutoff,1:-1]
	training_results = samples[:cutoff,-1]
	test_features = samples[cutoff:,1:-1]
	test_results = samples[cutoff:,-1]
	
	sample_weight = [(1 + data * 19) for data in samples[:cutoff,-1]]

	SVC_model = SVC(kernel=kernel, probability=True)
	SVC_model.fit(training_features, training_results, sample_weight=sample_weight)
	predicted_results = SVC_model.predict(test_features)

	return (
		SVC_model,
		metrics.classification_report(test_results, predicted_results, output_dict=False, zero_division=0),
		samples[:cutoff,:]
	)

def f1_score(recall: float, precision: float) -> float:
	return 2 * (recall * precision) / max(1, recall + precision)

def _make_stats_display(data: list[float], format: str="6.1%") -> str:
	data = np.sort(data)
	return f"{data[int(len(data)/2)]:<{format}} [{data[int(len(data)*0.05)]:>{format}} ~ {data[int(len(data)*0.95)]:<{format}}]"

def model_summary(trials, samples, training_size, pos_weight=10, kernel="rbf", advanced=False) -> dict:
	pass

def main():
	labels, valid_samples, invalid_samples = process_csv(DATASET_FILENAME, VALID_VALUES, ",")
	print(f"{len(valid_samples)} valid samples loaded ({len(invalid_samples)} invalidated)")
	model, report, _ = train_and_test(valid_samples, 2000)
	print(report)
	pass

if __name__ == "__main__":
	main()
