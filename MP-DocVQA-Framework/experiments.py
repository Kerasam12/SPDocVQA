import json

def count_characters(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
        total_characters = 0
        item_count = 0 
        for item in data:
            context = item.get("context", "")
            total_characters += len(context)
            item_count += 1
        avg_context_len = total_characters/item_count
        return avg_context_len


json_file_correct = 'answer_noans_correct.json' 
json_file_wrong = 'answer_noans_wrong.json'
avg_len_corrects = count_characters(json_file_correct)
avg_len_fail = count_characters(json_file_wrong)
print("CORRECT AVG LEN:", avg_len_corrects, "WRONG AVG LEN:", avg_len_fail)

#print("Total characters in 'context' pa# Example usageameter:", total_characters)