# main.py

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel

from preprocessing.audio_preprocessing import process_videos
from preprocessing.text_preprocessing import create_text_data_loader
from preprocessing.visual_preprocessing import create_face_data_loader
from preprocessing.visual_preprocessing import extract_frames_from_videos

from models.multimodal_model import MultimodalModel

def main():
    # Paths
    data_dir = 'datasets/bag_of_lies'
    videos_dir = os.path.join(data_dir, 'videos')
    audios_dir = os.path.join(data_dir, 'audios')
    transcripts_dir = os.path.join(data_dir, 'transcripts')
    frames_dir = os.path.join(data_dir, 'frames')
    labels_file = os.path.join(data_dir, 'labels.csv')

    # Parameters
    batch_size = 8
    max_length = 512
    num_epochs = 10
    learning_rate = 1e-4
    num_classes = 2  # Deceptive or Truthful

    # Load labels
    labels_df = pd.read_csv(labels_file)

    # Audio Preprocessing
    process_videos(videos_dir, audios_dir, transcripts_dir, labels_df)
    # Extract frames from videos
    extract_frames_from_videos(videos_dir, frames_dir, labels_df)


    # Encode labels
    label_encoder = LabelEncoder()
    labels_df['label_encoded'] = label_encoder.fit_transform(labels_df['label'])
    labels = labels_df['label_encoded'].values

    # Split data
    train_df, test_df = train_test_split(labels_df, test_size=0.2, random_state=42)

    # Text Data Loader
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_transcripts = [open(os.path.join(transcripts_dir, f"{vid}.txt")).read() for vid in train_df['video_id']]
    test_transcripts = [open(os.path.join(transcripts_dir, f"{vid}.txt")).read() for vid in test_df['video_id']]

    train_text_loader = create_text_data_loader(
        transcripts=train_transcripts,
        labels=train_df['label_encoded'].values,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length
    )

    test_text_loader = create_text_data_loader(
        transcripts=test_transcripts,
        labels=test_df['label_encoded'].values,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length
    )

    # Visual Data Loader
    def get_image_paths(df):
        image_paths = []
        for vid in df['video_id']:
            frame_dir = os.path.join(frames_dir, vid)
            frames = [os.path.join(frame_dir, img) for img in os.listdir(frame_dir) if img.endswith('.jpg')]
            image_paths.append(frames)
        return image_paths

    train_image_paths = get_image_paths(train_df)
    test_image_paths = get_image_paths(test_df)

    train_face_loader = create_face_data_loader(
        image_paths=train_image_paths,
        labels=train_df['label_encoded'].values,
        batch_size=batch_size
    )

    test_face_loader = create_face_data_loader(
        image_paths=test_image_paths,
        labels=test_df['label_encoded'].values,
        batch_size=batch_size
    )

    # Initialize models
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    text_feature_size = bert_model.config.hidden_size  # Usually 768
    face_feature_size = 512  # Output size of InceptionResnetV1

    multimodal_model = MultimodalModel(text_feature_size, face_feature_size, num_classes)

    # Move models to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_model = bert_model.to(device)
    multimodal_model = multimodal_model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(bert_model.parameters()) + list(multimodal_model.parameters()), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        multimodal_model.train()
        bert_model.train()
        total_loss = 0
        for text_batch, face_batch in zip(train_text_loader, train_face_loader):
            input_ids = text_batch['input_ids'].to(device)
            attention_mask = text_batch['attention_mask'].to(device)
            labels = text_batch['labels'].to(device)
            face_embeddings = face_batch['embeddings'].to(device)
            # Zero gradients
            optimizer.zero_grad()
            # Text feature extraction
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            text_features = outputs.pooler_output  # Shape: (batch_size, 768)
            # Forward pass
            outputs = multimodal_model(text_features, face_embeddings)
            # Compute loss
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_text_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # Evaluation
    multimodal_model.eval()
    bert_model.eval()
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for text_batch, face_batch in zip(test_text_loader, test_face_loader):
            input_ids = text_batch['input_ids'].to(device)
            attention_mask = text_batch['attention_mask'].to(device)
            labels = text_batch['labels'].to(device)
            face_embeddings = face_batch['embeddings'].to(device)
            # Text feature extraction
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            text_features = outputs.pooler_output
            # Forward pass
            outputs = multimodal_model(text_features, face_embeddings)
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.size(0)
    accuracy = correct_predictions.double() / total_predictions
    print(f'Test Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    main()
