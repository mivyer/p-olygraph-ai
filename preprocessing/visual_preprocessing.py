# preprocessing/visual_preprocessing.py

import os
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from torch.utils.data import Dataset

class FaceDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        # Load pre-trained model
        self.model = InceptionResnetV1(pretrained='vggface2').eval()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_files = self.image_paths[idx]
        label = self.labels[idx]
        # Process multiple images per video
        embeddings = []
        for img_path in image_files:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            with torch.no_grad():
                embedding = self.model(img.unsqueeze(0))
            embeddings.append(embedding.squeeze(0))
        # Average embeddings
        embeddings = torch.stack(embeddings)
        avg_embedding = embeddings.mean(dim=0)
        return {
            'embeddings': avg_embedding,
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    def extract_frames_from_videos(videos_dir, frames_dir, labels_df, frame_rate=1):
        os.makedirs(frames_dir, exist_ok=True)
        for idx, row in labels_df.iterrows():
            video_id = row['video_id']
            video_path = os.path.join(videos_dir, f"{video_id}.mp4")
            frame_output_dir = os.path.join(frames_dir, video_id)
            os.makedirs(frame_output_dir, exist_ok=True)
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps / frame_rate)
            count = 0
            success, frame = cap.read()
            while success:
                if count % frame_interval == 0:
                    frame_filename = os.path.join(frame_output_dir, f"frame_{count}.jpg")
                    cv2.imwrite(frame_filename, frame)
                success, frame = cap.read()
                count += 1
            cap.release()


def create_face_data_loader(image_paths, labels, batch_size):
    ds = FaceDataset(
        image_paths=image_paths,
        labels=labels
    )
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=4)
