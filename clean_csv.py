import csv

def clean_similarity_csv(input_file, output_file):
    seen_pairs = set()
    cleaned_rows = []

    with open(input_file, mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            qid = row['query_id']
            tid = row['region_id']
            sim = row['similarity']

            # Skip self-similarity
            if qid == tid:
                continue

            # Create an unordered pair key
            pair_key = tuple(sorted((qid, tid, sim)))

            # Skip if this pair (in either order) is already seen
            if pair_key in seen_pairs:
                continue

            seen_pairs.add(pair_key)
            cleaned_rows.append(row)

    # Write the cleaned data to a new file
    with open(output_file, mode='w', newline='') as csvfile:
        fieldnames = ['query_id', 'region_id', 'label', 'similarity', 'distance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',')
        writer.writeheader()
        writer.writerows(cleaned_rows)

# Example usage
clean_similarity_csv('combined_output.csv', 'cleaned_output.csv')
