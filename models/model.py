import torch
import torch.nn as nn
import torchvision.models as models

class SimpleVQA(nn.Module):
    def __init__(self, num_classes, vocab_size=10000, embed_dim=256):
        super(SimpleVQA, self).__init__()
        
        # 1. GÖRÜNTÜ İŞLEME (CNN - ResNet18)
        # Pretrained ağırlıkları kullanıyoruz
        self.resnet = models.resnet18(pretrained=True)
        # Son katmanı (sınıflandırıcıyı) kaldır, sadece özellikleri al
        self.resnet.fc = nn.Identity() 
        
        # 2. SORU İŞLEME (Embedding + LSTM)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, 256, batch_first=True)
        
        # 3. BİRLEŞTİRME VE SINIFLANDIRMA
        # ResNet çıktısı (512) + LSTM çıktısı (256) = 768 giriş
        self.classifier = nn.Sequential(
            nn.Linear(512 + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, images, question_indices):
        # Resim Özellikleri
        img_features = self.resnet(images) # [Batch, 512]
        
        # Soru Özellikleri
        embeds = self.embedding(question_indices)
        _, (hidden, _) = self.lstm(embeds)
        text_features = hidden[-1] # [Batch, 256] (Son durum)
        
        # Birleştir
        combined = torch.cat((img_features, text_features), dim=1)
        output = self.classifier(combined)
        return output