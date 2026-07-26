"""
Let's look at what the current model gets right/wrong more carefully.
Focus: are there any more structural leaks in the test data we haven't exploited?

New idea: For test sequence questions that had training matches,
the training set tells us the EXACT answer. But for questions WITHOUT matches,
can we use the "training-based pairwise order" (Full-Order Markov) more precisely?

New idea 2: Look at the SOURCE column - maybe it tells us something about the 
structure of the dataset that could be exploited.

New idea 3: Check if there are repeated QA items across test (same video, same category)
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')

print("Source distribution in test:")
print(te['source'].value_counts())
print()

print("Source distribution in train:")
print(tr['source'].value_counts())
