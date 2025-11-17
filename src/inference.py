import torch
import torchaudio
from models import SpeakerEncoderCNN
from data_loader import mel_transform
import torch.nn.functional as F

class Verifier:
    def __init__(self, model_path, embedding_dim=128, input_shape=(128, 282)):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SpeakerEncoderCNN(input_shape=input_shape, embedding_dim=embedding_dim)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def _preprocess(self, audio_path):
        wav, sr = torchaudio.load(audio_path)
        # Приводим к той же длине, что и при обучении
        target_len = 48000 * 3
        if wav.shape[1] > target_len:
            wav = wav[:, :target_len]
        else:
            padding = target_len - wav.shape[1]
            wav = torch.nn.functional.pad(wav, (0, padding))
            
        spec = mel_transform(wav)
        return spec.unsqueeze(0).to(self.device) # Добавляем batch dimension

    def get_embedding(self, audio_path):
        spec = self._preprocess(audio_path)
        with torch.no_grad():
            embedding = self.model(spec)
        return embedding

    def verify(self, file1, file2, threshold=0.7):
        emb1 = self.get_embedding(file1)
        emb2 = self.get_embedding(file2)
        
        # Косинусное расстояние = 1 - косинусное сходство
        cos_sim = F.cosine_similarity(emb1, emb2)
        cos_dist = 1 - cos_sim
        
        print(f"Cosine distance: {cos_dist.item():.4f}")
        print(f"Threshold: {threshold}")
        
        if cos_dist.item() < threshold:
            print("Result: SAME speaker")
            return True, cos_dist.item()
        else:
            print("Result: DIFFERENT speakers")
            return False, cos_dist.item()

if __name__ == '__main__':
    # Пример использования
    # Вам нужно указать корректные пути к файлам
    # file1 и file2 - от одного диктора
    # file3 - от другого диктора
    
    # ПРИМЕР: нужно найти реальные файлы в data/VCTK-Corpus/wav48/
    speaker1_file1 = "data/VCTK-Corpus/wav48/p225/p225_001.wav"
    speaker1_file2 = "data/VCTK-Corpus/wav48/p225/p225_002.wav"
    speaker2_file1 = "data/VCTK-Corpus/wav48/p226/p226_001.wav"
    
    verifier = Verifier(model_path="speaker_encoder.pth")

    print("\nComparing two files from the SAME speaker")
    verifier.verify(speaker1_file1, speaker1_file2)
    
    print("\nComparing two files from DIFFERENT speakers")
    verifier.verify(speaker1_file1, speaker2_file1)
