import numpy as np
import random



thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
preds = []
for i in range(400):
    pred = random.choices(population=thresholds, k=24)
    print(pred)
    preds.append(pred)
print(preds)
print(np.asarray([1, 1]) > np.asarray([0, 0]))