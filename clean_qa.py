import json

# Read the input file
with open('evaluation/prediction_train_llama3.3/final.json', 'r') as f:
    data = json.load(f)

# Remove 'id' field from each QA entry
for entry in data:
    if 'qa' in entry and 'id' in entry['qa']:
        del entry['qa']['id']

# Write the modified data to a new file
with open('evaluation/prediction_train_llama3.3/final_cleaned.json', 'w') as f:
    json.dump(data, f, indent=2)

print("Successfully removed 'id' field from all QA entries.") 