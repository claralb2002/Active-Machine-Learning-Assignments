import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder

class DatasetClassifier(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

        self.label_encoder = LabelEncoder()
        self.df["Encoded_Label"] = self.label_encoder.fit_transform(self.df["Label"])  

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx]['Image_Path']
        label = self.df.iloc[idx]['Encoded_Label']
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
