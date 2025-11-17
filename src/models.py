import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeakerEncoderCNN(nn.Module):
    def __init__(self, input_shape=(128, 282), embedding_dim=128):
        super(SpeakerEncoderCNN, self).__init__()
        # Input shape: (batch, 1, n_mels, time_steps)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Рассчитываем размер для Flatten слоя
        self.flattened_size = self._get_flattened_size(input_shape)

        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, embedding_dim)

    def _get_flattened_size(self, shape):
        # Вспомогательная функция для автоматического расчета размера
        x = torch.zeros(1, 1, shape[0], shape[1])
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        return x.flatten().shape[0]

    def forward(self, x):
        # Добавляем канал (batch, n_mels, time) -> (batch, 1, n_mels, time)
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1) # Flatten
        
        x = F.relu(self.fc1(x))
        embedding = self.fc2(x)
        
        # Нормализуем вектор (важно для косинусного расстояния)
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding
