import csv

def analyze():
    # Read training data
    train_data = []
    with open('training_qa.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_data.append(row)
            
    # Check if there is a mapping from options to answer
    # Example: are some actions much more likely to be the correct answer?
    action_counts = {}
    for row in train_data:
        ans_letter = row.get('answer', '')
        if ans_letter in ['A', 'B', 'C', 'D']:
            action = row.get(ans_letter, '')
            action_counts[action] = action_counts.get(action, 0) + 1
            
    sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("Most common correct actions:")
    for a, c in sorted_actions[:20]:
        print("%s: %s" % (a, c))

if __name__ == '__main__':
    analyze()
