import torch
from torch.utils.data import Dataset

class KvasirHFDataset(Dataset):
    def __init__(self, hf_dataset, answer_map, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        self.answer_map = answer_map

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # 1. Resim İşleme
        image = item['image'].convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 2. Soru ve Cevap İşleme
        question_text = item['question']
        answer_text = str(item['answer']).lower()
        
        # Cevabı ID'ye çevir (Bulamazsa 0 ver)
        label = self.answer_map.get(answer_text, 0)

        return {
            'image': image, 
            'question': question_text, 
            'answer': torch.tensor(label, dtype=torch.long)
        }