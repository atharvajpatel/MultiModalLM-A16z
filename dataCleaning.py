#To altter multimodal_tweets.csv and add a column with the path to its corresponding image in all

import pandas as pd

# Load the CSV file
df = pd.read_csv('/mnt/data/Multimodal_Tweets.csv')

# Define the folder containing images
image_folder = 'Image_Dataset/All'

# Add a new column 'Image Path' to the DataFrame
def get_image_path(file_name):
    base_name = file_name.replace('.txt', '')  # Remove .txt from file name
    image_file = f"{base_name}.jpg"  # Assuming images are .jpg as per the screenshot
    return f"{image_folder}/{image_file}"

df['Image Path'] = df['File Name'].apply(get_image_path)

# Save the updated DataFrame to a new CSV file
df.to_csv('/mnt/data/Multimodal_Tweets_With_Images.csv', index=False)

print("Updated CSV with Image Path saved successfully.")