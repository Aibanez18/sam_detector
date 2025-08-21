from sklearn.metrics import average_precision_score

def calculate_average_precision(labels):
    precisions = []
    for idx, label in enumerate(labels):
        if label == 1:
            precision = labels[:idx + 1].count(label) / (idx + 1)
            precisions.append(precision)

    if precisions:
        return sum(precisions) / len(precisions)
    return 0.0

truth = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] # 1, 2/3, 3/5, 4/7, 5/9 -> 0.67873015874
truth2 = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] # 1/2, 2/4, 3/6, 4/8, 5/10 -> 0.5
predictions = [0.9, 0.1, 0.8, 0.4, 0.7, 0.2, 0.6, 0.3, 0.5, 0.0]
average_precision = calculate_average_precision(truth2)

print(f"Average Precision: {average_precision}")