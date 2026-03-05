import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import DistilBertTokenizer
import os

# Proje içindeki yardımcı dosyalar
from data.data_loading import get_kvasir_data, get_train_val_split
from local_datasets.dataset import KvasirHFDataset
from models.model import ResNet_BERT_CoAttention_VQA

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- ResNet + BERT Co-Attention Modeli Başlatılıyor ({device}) ---")

    # 1. HAZIRLIK
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    raw_data = get_kvasir_data()
    train_data, val_data = get_train_val_split(raw_data)
    
    # Cevap haritası
    all_answers = sorted(list(set(str(ans).lower() for ans in train_data['answer'])))
    answer_map = {ans: i for i, ans in enumerate(all_answers)}

    # 2. DATASET VE LOADER
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = KvasirHFDataset(train_data, answer_map, transform=transform)
    
    def collate_fn(batch):
        imgs = torch.stack([item['image'] for item in batch])
        texts = [item['question'] for item in batch]
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=32, return_tensors="pt")
        lbls = torch.tensor([item['answer'] for item in batch])
        return imgs, encoded['input_ids'], encoded['attention_mask'], lbls

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # 3. MODEL, OPTIMIZER, LOSS
    model = ResNet_BERT_CoAttention_VQA(num_classes=len(answer_map)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    # 4. EĞİTİM (Örnek: 3 Epoch)
    print("Eğitim başlıyor...")
    for epoch in range(3):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for imgs, ids, masks, lbs in loop:
            imgs, ids, masks, lbs = imgs.to(device), ids.to(device), masks.to(device), lbs.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs, ids, masks)
            loss = criterion(outputs, lbs)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), "resnet_bert_coattn.pth")
    print("🎉 Eğitim tamamlandı ve model kaydedildi!")

if __name__ == "__main__":
    main()