import csv

test_data = {}
for row in csv.DictReader(open('test_qa.csv')):
    test_data[row['qa_id']] = row

sample_data = {}
for row in csv.DictReader(open('sample_submission.csv')):
    sample_data[row['qa_id']] = row['prediction']

cats = {}
for k, v in test_data.items():
    cat = v['category']
    ans_len = len(sample_data.get(k, ''))
    if cat not in cats:
        cats[cat] = set()
    cats[cat].add(ans_len)

print(cats)
