import json
from typing import List, Dict


def extract_reddit_data(reddit_json: List[Dict]) -> Dict:
    """
    Extract post and comments from Reddit API JSON

    Returns:
        {
            'id': post_id,
            'title': post_title,
            'content': post_selftext,
            'body': combined_comments_text
        }
    """

    # Extract post data (first element in array)
    post_listing = reddit_json[0]['data']['children'][0]['data']

    post_id = post_listing['id']
    title = post_listing['title']
    content = post_listing['selftext']

    # Extract all comments (second element in array)
    comments_listing = reddit_json[1]['data']['children']
    all_comments = []

    def extract_comments_recursive(comment_list):
        """Recursively extract comment text from nested structure"""
        for item in comment_list:
            # Skip if not a comment (could be 'more' type)
            if item['kind'] != 't1':
                continue

            comment_data = item['data']

            # Skip deleted/removed comments
            if comment_data.get('body') in ['[deleted]', '[removed]']:
                continue

            # Add comment text
            all_comments.append(comment_data['body'])

            # Recursively get replies if they exist
            replies = comment_data.get('replies')
            if replies and isinstance(replies, dict):
                reply_children = replies['data']['children']
                extract_comments_recursive(reply_children)

    # Start recursive extraction
    extract_comments_recursive(comments_listing)

    # Combine all comments into single body text
    body = '\n\n'.join(all_comments)

    return {
        'id': post_id,
        'title': title,
        'content': content,
        'body': body
    }


# Usage Example
if __name__ == "__main__":
    # Load your JSON file
    with open('/Users/santhiyatheanraj/PycharmProjects/HCI/reddit_jsons/1c9xitq.json', 'r', encoding='utf-8') as f:
        reddit_data = json.load(f)
    output_file = '/Users/santhiyatheanraj/PycharmProjects/HCI/extracted_json/philly.json'
    # Extract data
    extracted = extract_reddit_data(reddit_data)
    print(f"Saving to {'/Users/santhiyatheanraj/PycharmProjects/HCI/extracted_json/'}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(extracted, f, indent=2, ensure_ascii=False)

    # Print results
    print(f"ID: {extracted['id']}")
    print(f"Title: {extracted['title']}")
    print(f"Content: {extracted['content'][:100]}...")
    print(f"Body (first 200 chars): {extracted['body'][:200]}...")
    print(f"\nTotal comments combined: {len(extracted['body'])} characters")