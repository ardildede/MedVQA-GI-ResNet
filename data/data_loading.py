from datasets import load_dataset

def get_kvasir_data():
    print("--- Veri Seti Yükleniyor (Hugging Face) ---")
    # Veriyi indir (Cache varsa oradan okur)
    dataset = load_dataset("SimulaMet-HOST/Kvasir-VQA")
    
    # Dataset yapısını kontrol et
    if 'train' in dataset:
        return dataset['train']
    elif 'raw' in dataset:
        return dataset['raw']
    else:
        # Garanti çözüm: ilk key'i al
        return dataset[list(dataset.keys())[0]]

def get_train_val_split(dataset, split_ratio=0.8):
    # Veriyi eğitim ve doğrulama olarak ayır
    split_data = dataset.train_test_split(test_size=(1 - split_ratio))
    return split_data['train'], split_data['test']