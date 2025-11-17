import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from data_loader import VCTKDataset, mel_transform
from models import SpeakerEncoderCNN
from torch.utils.tensorboard import SummaryWriter

# Гиперпараметры
CONFIG = {
    "data_path": "./data/VCTK-Corpus",
    "epochs": 5,
    "batch_size": 128,
    "learning_rate": 0.001,
    "embedding_dim": 128,
    "triplet_margin": 0.5,
    "train_split_ratio": 0.8,
    "experiment_name": "speaker_verification_cnn_v1" # Добавили имя для логов
}

def train():
    # Инициализация TensorBoard SummaryWriter
    # Логи будут сохраняться в папку runs/speaker_verification_cnn_v1
    writer = SummaryWriter(f"runs/{CONFIG['experiment_name']}")
    
    # Используем config, чтобы не менять код ниже
    config = CONFIG

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Данные
    full_dataset = VCTKDataset(data_path=config["data_path"], transform=mel_transform)
    
    train_size = int(config["train_split_ratio"] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=2)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Модель, лосс и оптимизатор
    sample_spec, _, _ = train_dataset[0]
    input_shape = sample_spec.shape[1:]
    
    model = SpeakerEncoderCNN(input_shape=input_shape, embedding_dim=config["embedding_dim"]).to(device)
    criterion = torch.nn.TripletMarginLoss(margin=config["triplet_margin"])
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    # Можно опционально добавить граф модели в TensorBoard
    # writer.add_graph(model, sample_spec.unsqueeze(0).to(device))

    # Цикл обучения
    for epoch in range(config["epochs"]):
        model.train()
        total_train_loss = 0
        for i, (anchor, positive, negative) in enumerate(train_loader):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            optimizer.zero_grad()
            
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)
            
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            if i % 20 == 0:
                print(f"Epoch {epoch+1}/{config['epochs']}, Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)

        # Валидация
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                
                anchor_emb = model(anchor)
                positive_emb = model(positive)
                negative_emb = model(negative)
                
                loss = criterion(anchor_emb, positive_emb, negative_emb)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} summary: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Логирование в TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)

    # Сохранение модели
    torch.save(model.state_dict(), "speaker_encoder.pth")
    
    writer.close() # Закрываем writer
    print("Training finished.")

if __name__ == "__main__":
    train()
