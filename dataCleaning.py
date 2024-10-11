import pandas as pd
import re

# Load the CSV file
df = pd.read_csv('Multimodal-Data-Tweets.csv')

# Define the folder containing images
image_folder = 'Image_Dataset/All'

# Add a new column 'Image Path' to the DataFrame
def get_image_path(file_name):
    match = re.match(r'(\d+)', file_name)  # Extract number from the file name
    if match:
        image_number = match.group(1)
        image_file = f"{image_number}.jpg"  # Construct image file name
        return f"{image_folder}/{image_file}"
    return None

df['Image Path'] = df['File Name'].apply(get_image_path)

# Save the updated DataFrame to a new CSV file
df.to_csv('Multimodal_Tweets_With_Images.csv', index=False)

print("Updated CSV with Image Path saved successfully.")