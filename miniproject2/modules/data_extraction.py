import os
import pandas as pd

def create_dataset(base_directory, categories):
    data = []
    
    for category in categories:
        directory = os.path.join(base_directory, category, "images")
        if os.path.exists(directory):
            for image_name in os.listdir(directory):
                image_path = os.path.join("data","archive", category, "images", image_name)
                if os.path.isfile(os.path.join(directory, image_name)):  # Ensures it's a file
                    data.append([category, image_name, image_path])
    
    df = pd.DataFrame(data, columns=['Label', 'Image_Name', 'Image_Path'])
    return df

# Example usage
base_directory = "/Users/claralouisebrodt/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/4. semester/02463 Active machine learnig/Assignments/Assignment 2/Active-Machine-Learning-Assignments/miniproject2/data/archive"
categories = ["COVID", "Lung_opacity", "Normal", "Viral Pneumonia"]
df = create_dataset(base_directory, categories)
df.to_csv("extracted_dataset.csv", index=False)
print(df.head())  