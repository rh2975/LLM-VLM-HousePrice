import pandas as pd
import numpy as np
from transformers import DistilBertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import albumentations as A
import timm
from tqdm.auto import tqdm
import os
from logger import logger

class CFG:
    image_path = "./Categorized_Pictures"
    batch_size = 16
    num_workers = 0
    image_encoder_lr = 1e-5
    weight_decay = 1e-3
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_embedding = 768
    text_embedding = 768
    model_name = 'vit_base_patch16_clip_224'
    text_encoder_model = "distilbert-base-uncased"
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200
    pretrained = True
    trainable = True
    temperature = 1.0
    size = 224
    projection_dim = 30
    dropout = 0.2
    min_image_size = 150
    min_sharpness = 150.0

PSEUDO_PROMPTS = {
    "indoor": [
        "a bright indoor photo of a living room, bedroom, or kitchen in a residential house",
        "modern interior of a house showing spacious living area with furniture",
        "bedroom or bathroom inside a home with clean finishes",
        "indoor residential space with natural light and comfortable furniture"
    ],
    "outdoor": [
        "an outdoor photo of a house backyard, garden, facade, or swimming pool",
        "front facade or exterior view of a residential house",
        "backyard garden with lawn and landscaping around a home",
        "outdoor view of a house with driveway, garage or swimming pool area"
    ],
    "mixed": [
        "a property photo showing both indoor rooms and outdoor views through large windows or doors",
        "indoor living area with open doors leading to outdoor or garden",
        "bright room with large glass windows overlooking backyard or nature",
        "indoor-outdoor transition with dining or balcony view"
    ],
    "floorplan": [
        "a 2D floorplan diagram or architectural layout of a house with room labels",
        "technical blueprint drawing showing room arrangements"
    ]
}

# Image Encoder
class ImageEncoder(nn.Module):
    def __init__(self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

# Text Encoder
class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        self.model = DistilBertModel.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad = trainable
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state[:, self.target_token_idx, :]

# Projection Head
class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim=CFG.projection_dim, dropout=CFG.dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

# CLIP Model with Contrastive Loss
class CLIPModel(nn.Module):
    def __init__(self, temperature=CFG.temperature, image_embedding=CFG.image_embedding, text_embedding=CFG.text_embedding):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch, image_only=False):
        if image_only:
            image_features = self.image_encoder(batch["image"])
            image_embeddings = self.image_projection(image_features)
            return None, image_embeddings
        else:
            image_features = self.image_encoder(batch["image"])
            text_features = self.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            image_embeddings = self.image_projection(image_features)
            text_embeddings = self.text_projection(text_features)
            logits = (text_embeddings @ image_embeddings.T) / self.temperature
            labels = torch.arange(len(logits)).to(logits.device)
            loss_i = F.cross_entropy(logits, labels)
            loss_t = F.cross_entropy(logits.T, labels)
            loss = (loss_i + loss_t) / 2
            return loss, image_embeddings


def assign_category_with_prompts(image_embeddings, clip_model, tokenizer):
    category_embeddings = {}
    prompt_names = []
    prompt_texts = []
    
    for cat, prompts in PSEUDO_PROMPTS.items():
        for prompt in prompts:
            prompt_texts.append(prompt)
            prompt_names.append(cat)
    
    # Tokenize all prompts
    encoded = tokenizer(prompt_texts, padding=True, truncation=True, 
                       max_length=CFG.max_length, return_tensors='pt').to(CFG.device)
    
    with torch.no_grad():
        text_features = clip_model.text_encoder(encoded['input_ids'], encoded['attention_mask'])
        text_embeddings = clip_model.text_projection(text_features)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
        
        image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        
        similarity = (image_embeddings @ text_embeddings.T)  # [batch, num_prompts]
        
        best_prompt_idx = similarity.argmax(dim=1)
        predicted_categories = [prompt_names[i] for i in best_prompt_idx.cpu().numpy()]
    
    return predicted_categories


# Image Dataset with Quality Filtering and Category
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, encoded_captions, transforms, original_indices, categories):
        self.image_filenames = []
        self.captions = []
        self.encoded_captions = {key: [] for key in encoded_captions}
        self.valid_indices = []
        self.image_categories = []
        missing_files = 0
        invalid_images = 0
        for idx, (fname, orig_idx, cat) in enumerate(zip(image_filenames, original_indices, categories)):
            image_path = os.path.join(CFG.image_path, fname)
            if not os.path.exists(image_path):
                missing_files += 1
                logger.debug(f"Missing image file: {image_path}")
                continue
            image = cv2.imread(image_path)
            if image is None or image.size == 0:
                invalid_images += 1
                logger.debug(f"Invalid image: {image_path}")
                continue
            if image.shape[0] < CFG.min_image_size or image.shape[1] < CFG.min_image_size:
                logger.debug(f"Image too small: {image_path}, shape: {image.shape}")
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            if sharpness < CFG.min_sharpness:
                logger.debug(f"Image not sharp enough: {image_path}, sharpness: {sharpness}")
                continue
            self.image_filenames.append(fname)
            self.captions.append(captions[idx])
            self.valid_indices.append(orig_idx)
            self.image_categories.append(cat)
            for key, values in encoded_captions.items():
                self.encoded_captions[key].append(values[idx])
        logger.info(f"ImageDataset: {len(self.image_filenames)} valid images, {missing_files} missing files, {invalid_images} invalid images")
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {key: torch.tensor(values[idx]) for key, values in self.encoded_captions.items()}
        item['index'] = self.valid_indices[idx]
        image_path = os.path.join(CFG.image_path, self.image_filenames[idx])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]
        item['category'] = self.image_categories[idx]
        return item

    def __len__(self):
        return len(self.captions)


def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose([
            A.Resize(CFG.size, CFG.size),
            A.RandomCrop(CFG.size, CFG.size, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
            A.Normalize(max_pixel_value=255.0),
        ])
    return A.Compose([
        A.Resize(CFG.size, CFG.size),
        A.Normalize(max_pixel_value=255.0),
    ])


def extract_image_embeddings(image_df, clip_model, tokenizer, transforms):
    dataset = ImageDataset(
        image_df['img_id'].values,
        image_df['house_description'].fillna('').values,
        tokenizer(
            image_df['house_description'].fillna('').tolist(),
            padding=True,
            truncation=True,
            max_length=CFG.max_length,
            return_tensors='pt'
        ),
        transforms=transforms,
        original_indices=image_df.index,
        categories=image_df['category'].tolist() if 'category' in image_df.columns else None
    )
    
    if len(dataset) == 0:
        return None
        
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=CFG.batch_size, num_workers=CFG.num_workers,
        shuffle=False, collate_fn=lambda x: torch.utils.data.dataloader.default_collate(x)
    )
    
    embeddings = []
    ids = []
    categories = []
    
    clip_model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting image embeddings"):
            batch = {k: v.to(CFG.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            l, image_embeddings = clip_model(batch, image_only=True)
            embeddings.append(image_embeddings.cpu().numpy())
            ids.extend(batch['index'].cpu().numpy())
            
            batch_categories = assign_category_with_prompts(image_embeddings, clip_model, tokenizer)
            categories.extend(batch_categories)
    
    embeddings = np.vstack(embeddings)
    embedding_df = pd.DataFrame(embeddings, columns=[f'img_embed_{i}' for i in range(embeddings.shape[1])])
    embedding_df['ID'] = [image_df.loc[idx, 'ID'] for idx in ids]
    embedding_df['category'] = categories
    
    return embedding_df.groupby('ID').mean().reset_index()
