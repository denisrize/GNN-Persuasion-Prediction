def clean_selftext(text):
    """
    Remove the CMV rules section from the selftext field.
    """
    return text.split('*Hello, users of CMV!')[0]

# Step 2: Preprocessing - Extract Relevant Features
def preprocess_data(data):
    """Extract and clean relevant data."""
    processed_data = []
    for entry in data:
        submission = {
            "id": entry["id"],
            "name": entry["name"],
            "author": entry["author"],
            "author_flair_text": entry["author_flair_text"], # General number of deltas got in the CMV subreddit
            "ups": entry["ups"],
            "downs": entry["downs"],
            "num_comments": entry["num_comments"],
            "body": clean_selftext(entry["selftext"]),
            "title": entry["title"],
            "comments": entry['comments'],
        }
        processed_data.append(submission)
    return processed_data