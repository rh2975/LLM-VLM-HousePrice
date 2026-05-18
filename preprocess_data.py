import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler, PolynomialFeatures
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
import re
from datetime import datetime
import time
from transformers import DistilBertTokenizer
import torch
from tqdm.auto import tqdm
import os
import umap
from llm_config import *
from clip_config import *
from logger import logger


def preprocess_data(csv_file, image_csv, max_rows=None, llama_rows=7000):
    logger.info(f"Loading data from {os.path.abspath(csv_file)}")
    df = pd.read_csv(csv_file)
    logger.info(f"DataFrame columns: {df.columns.tolist()}")
    if df.empty:
        logger.error("DataFrame is empty")
        raise ValueError("Main dataset is empty")
    logger.info(f"DataFrame shape: {df.shape}")

    image_df = pd.read_csv(image_csv)
    image_df['ID'] = image_df['img_id'].apply(
        lambda x: re.search(r'^\d+', str(x)).group(0) if re.search(r'^\d+', str(x)) else '0'
    )
    def extract_category(img_id):
        try:
            match = re.search(r'_(\w+)_(\d+)\.jpg$', str(img_id))
            return match.group(1) if match else 'unknown'
        except Exception as e:
            logger.debug(f"Invalid img_id format: {img_id}, error: {str(e)}")
            return 'unknown'
    image_df['category'] = image_df['img_id'].apply(extract_category)
    logger.info(f"Image dataset sample (first 5 IDs and categories):\n{image_df[['img_id', 'ID', 'category']].head().to_string()}")

    df = df.copy()
    df['ID'] = df['ID'].astype(str).str.strip()
    image_df['ID'] = image_df['ID'].astype(str).str.strip()

    logger.info(f"Main dataset sample (first 5 IDs):\n{df[['ID']].head().to_string()}")

    common_ids = set(df['ID']).intersection(set(image_df['ID']))
    coverage_percent = len(common_ids) * 100 / len(df) if len(df) > 0 else 0
    logger.info(f"Properties with matching images: {len(common_ids)} out of {len(df)} ({coverage_percent:.2f}%)")
    logger.info(f"Sample IDs from main dataset: {df['ID'].head().tolist()}")
    logger.info(f"Sample IDs from image dataset: {image_df['ID'].head().tolist()}")
    if coverage_percent == 0:
        logger.warning("Zero image coverage detected. Check ID formats or image CSV.")

    if len(df) == 0:
        raise ValueError("Main dataset is empty after processing")

    desired_features = [
        'distance_from_station', 'bedroom', 'bathroom', 'parking', 'property_type', 'price',
        'sold_date', 'description_heading', 'description_content', 'features', 'Lat_x', 'Lng_x',
        'Median_rent_weekly', 'Close_to_Shops', 'Close_to_Transport', 'Close_to_Schools',
        'school_ranking', 'Median_total_personal_income_weekly', 'Median_VCE_score', 'Air_Conditioning',
        'Swimming_Pool', 'Fireplace', 'Gym', 'Renovated', 'Locality_x'
    ]
    available_features = [f for f in desired_features if f in df.columns] + ['ID']
    df = df[available_features].copy()
    
    for col in ['description_heading', 'description_content', 'features']:
        if col in df.columns:
            df[col] = df[col].fillna('')
    
    for col in ['property_type']:
        if col in df.columns:
            df[col] = df[col].fillna('Other')
    
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop(['price'], errors='ignore')
    for col in numerical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    outliers = iso_forest.fit_predict(df[numerical_cols].fillna(df[numerical_cols].median()))
    df = df[outliers == 1]
    logger.info(f"Removed {np.sum(outliers == -1)} outliers")

    imputer = KNNImputer(n_neighbors=5)
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
    
    price_q99 = df['price'].quantile(0.99)
    price_q01 = df['price'].quantile(0.01)
    df = df[(df['price'] >= price_q01) & (df['price'] <= price_q99)]
    
    prop_type_counts = df['property_type'].value_counts()
    rare_types = prop_type_counts[prop_type_counts < 3].index
    df['property_type'] = df['property_type'].apply(lambda x: 'Other' if x in rare_types else x)
    
    current_year = datetime.now().year
    df['sold_year'] = pd.to_datetime(df['sold_date'], format='%a %d-%b-%y', errors='coerce').dt.year
    df['inflation_factor'] = df['sold_year'].apply(
        lambda x: 1.02 ** (current_year - x) if pd.notna(x) else 1.0
    )
    df['price'] = (df['price'] * df['inflation_factor']).astype(float)
    
    if max_rows is not None:
        df = df.head(max_rows)
    
    y = np.log1p(df['price']).astype(float)
    if 'distance_from_station' in df.columns:
        df['distance_from_station'] = (df['distance_from_station'].astype(float) / 1000).astype(float)
        df['dist_squared'] = df['distance_from_station'] ** 2
    
    if 'bedroom' in df.columns and 'bathroom' in df.columns:
        df['bed_bath_ratio'] = df['bedroom'] / (df['bathroom'] + 1)
    
    if 'Lat_x' in df.columns and 'Lng_x' in df.columns:
        df['lat_lng_interaction'] = df['Lat_x'] * df['Lng_x']
        df['dist_to_center'] = np.sqrt(
            (df['Lat_x'] - (-37.8136))**2 + (df['Lng_x'] - 144.9631)**2
        )
    
    if 'Median_rent_weekly' in df.columns and 'bedroom' in df.columns:
        df['rent_bedroom_interaction'] = df['Median_rent_weekly'] * df['bedroom']
    
    if 'school_ranking' in df.columns and 'Close_to_Schools' in df.columns:
        df['school_proximity_interaction'] = df['school_ranking'] * df['Close_to_Schools'].fillna(0)
    
    if 'sold_date' in df.columns:
        df['month_of_sale'] = pd.to_datetime(df['sold_date'], format='%a %d-%b-%y', errors='coerce').dt.month.fillna(0)
        df['day_of_sale'] = pd.to_datetime(df['sold_date'], format='%a %d-%b-%y', errors='coerce').dt.dayofweek.fillna(0)
        df['quarter'] = pd.to_datetime(df['sold_date'], format='%a %d-%b-%y', errors='coerce').dt.quarter.fillna(0)
        median_sale_year = df['sold_year'].median()
        df['market_trend'] = df['sold_year'].apply(lambda x: 1 if x > median_sale_year else 0)
    
    if 'Locality_x' in df.columns:
        locality_stats = df.groupby('Locality_x')['price'].agg(['mean', 'std']).reset_index()
        locality_stats.columns = ['Locality_x', 'locality_mean_price', 'locality_std_price']
        df = df.merge(locality_stats, on='Locality_x', how='left')
    
    cache_file = "extracted_features.csv"
    partial_cache_file = "partial_features.csv"
    if os.path.exists(cache_file):
        os.remove(cache_file)
    
    feature_names = [
        'proximity_minutes', 'proximity_km', 'near_shops', 'near_transport', 'near_schools',
        'is_quiet', 'is_busy', 'is_central', 'has_view', 'north_facing', 'urgency_level',
        'is_recently_renovated', 'is_new', 'is_old', 'has_luxury_finishes', 'sale_season'
    ]
    
    logger.info(f"Extracting context with LLaMA for first {llama_rows} rows...")
    start_time = time.time()
    context_data = []
    total_rows = len(df)
    for i, (idx, row) in enumerate(df.iterrows()):
        if i < llama_rows:
            features = extract_context_llama(row, i, min(llama_rows, total_rows))
        else:
            features = extract_context_rule_based(row)
        
        if len(features) != len(feature_names):
            logger.error(f"Row {i} (index {idx}): Expected {len(feature_names)} features, got {len(features)}. Using default.")
            features = [0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'none']
        
        context_data.append(features)
        
        if (i + 1) % 100 == 0:
            partial_df = pd.DataFrame(context_data, columns=feature_names, index=df.index[:len(context_data)])
            partial_df.to_csv(partial_cache_file)
            logger.info(f"Saved partial features for {i+1} rows to {partial_cache_file}")

    if len(context_data) != len(df):
        logger.error(f"Context data has {len(context_data)} rows, expected {len(df)}. Truncating or padding.")
        if len(context_data) > len(df):
            context_data = context_data[:len(df)]
        else:
            for _ in range(len(df) - len(context_data)):
                context_data.append([0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'none'])

    features_df = pd.DataFrame(context_data, columns=feature_names, index=df.index)

    for col in feature_names:
        if col in ['proximity_minutes', 'proximity_km']:
            features_df[col] = features_df[col].astype(float)
        elif col == 'sale_season':
            features_df[col] = features_df[col].astype(str)
        else:
            features_df[col] = features_df[col].astype(int)
    features_df.to_csv(cache_file)
    df[feature_names] = features_df
    logger.info(f"Context extraction completed in {time.time() - start_time:.2f} seconds")
    
    numerical_llama_cols = ['proximity_minutes', 'proximity_km']
    if numerical_llama_cols:
        scaler_llama = RobustScaler()
        df[numerical_llama_cols] = scaler_llama.fit_transform(df[numerical_llama_cols])
    
    if common_ids:
        clip_model = CLIPModel().to(CFG.device)
        tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
        transforms = get_transforms(mode="train")
        
        encoded_captions = tokenizer(
            image_df['house_description'].fillna('').tolist(),
            padding=True,
            truncation=True,
            max_length=CFG.max_length,
            return_tensors='pt'
        )
        dataset = ImageDataset(
            image_df['img_id'].values,
            image_df['house_description'].fillna('').values,
            encoded_captions,
            transforms=transforms,
            original_indices=image_df.index,
            categories=image_df['category'].tolist()
        )
        if len(dataset) > 0:
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=CFG.batch_size,
                num_workers=CFG.num_workers,
                shuffle=True,
                collate_fn=lambda x: torch.utils.data.dataloader.default_collate([item for item in x if item is not None])
            )
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=CFG.batch_size,
                num_workers=CFG.num_workers,
                shuffle=False,
                collate_fn=lambda x: torch.utils.data.dataloader.default_collate([item for item in x if item is not None])
            )
            optimizer = torch.optim.AdamW(clip_model.parameters(), lr=CFG.image_encoder_lr, weight_decay=CFG.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=1)
            clip_model.train()
            best_val_loss = float('inf')
            patience = 2
            epochs_no_improve = 0
            for epoch in range(CFG.epochs):
                total_train_loss = 0
                for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{CFG.epochs} (Train)"):
                    if batch is None:
                        continue
                    batch = {k: v.to(CFG.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    loss, _ = clip_model(batch)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(clip_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_train_loss += loss.item()
                avg_train_loss = total_train_loss / len(train_dataloader)
                
                total_val_loss = 0
                with torch.no_grad():
                    for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{CFG.epochs} (Val)"):
                        if batch is None:
                            continue
                        batch = {k: v.to(CFG.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                        loss, _ = clip_model(batch)
                        total_val_loss += loss.item()
                avg_val_loss = total_val_loss / len(val_dataloader)
                
                logger.info(f"Epoch {epoch+1}/{CFG.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                scheduler.step(avg_val_loss)
                if avg_val_loss < best_val_loss * 0.95:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    torch.save(clip_model.state_dict(), 'best_clip_model.pt')
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
            clip_model.load_state_dict(torch.load('best_clip_model.pt'))
            clip_model.eval()
            
            property_embeddings = extract_image_embeddings(image_df, clip_model, tokenizer, transforms)
            if property_embeddings is not None:
                df = df.merge(property_embeddings, on='ID', how='left')
                embed_cols = [col for col in property_embeddings.columns if col != 'ID']
                df['has_image'] = df[embed_cols].notna().any(axis=1).astype(int)
                df[embed_cols] = df[embed_cols].fillna(0)
                for col in [c for c in embed_cols if c != 'clip_score' and c != 'category']:
                    df[col] = df[col] * df['clip_score'].fillna(1.0)
                df['category_outdoor'] = (df['category'] == 'outdoor').astype(int)
                df['category_indoor'] = (df['category'] == 'indoor').astype(int)
                df['category_mixed'] = (df['category'] == 'mixed').astype(int)
                df['category_floorplan'] = (df['category'] == 'floorplan').astype(int)
                for cat in ['outdoor', 'indoor', 'mixed']:
                    cat_emb_df = property_embeddings[property_embeddings['category'] == cat].copy()
                    cat_emb_df.columns = [col if col in ['ID', 'clip_score'] else f'{cat}_{col}' for col in cat_emb_df.columns]
                    df = df.merge(cat_emb_df, on='ID', how='left')
                for col in df.columns:
                    if col.startswith(('outdoor_img_embed', 'indoor_img_embed', 'mixed_img_embed')):
                        df[col] = df[col].fillna(0)
                for cat in ['outdoor', 'indoor', 'mixed']:
                    for col in ['bedroom', 'distance_from_station', 'Median_rent_weekly']:
                        if col in df.columns:
                            for i in range(CFG.projection_dim):
                                df[f'{cat}_img_embed_{i}_{col}'] = df[f'{cat}_img_embed_{i}'] * df[col]
                
                # Apply UMAP to reduce embedding dimensionality
                embed_cols = [col for col in df.columns if col.startswith('img_embed') or col.startswith('outdoor_img_embed') or 
                             col.startswith('indoor_img_embed') or col.startswith('mixed_img_embed')]
                if embed_cols:
                    umap_reducer = umap.UMAP(n_components=10, random_state=42)
                    reduced_embeddings = umap_reducer.fit_transform(df[embed_cols].fillna(0))
                    reduced_cols = [f'umap_embed_{i}' for i in range(reduced_embeddings.shape[1])]
                    df[reduced_cols] = reduced_embeddings
                    df = df.drop(columns=embed_cols)
            else:
                df['has_image'] = 0
                for i in range(CFG.projection_dim):
                    df[f'img_embed_{i}'] = 0
                df['clip_score'] = 0
                df['category_outdoor'] = 0
                df['category_indoor'] = 0
                df['category_mixed'] = 0
                df['category_floorplan'] = 0
        else:
            df['has_image'] = 0
            for i in range(CFG.projection_dim):
                df[f'img_embed_{i}'] = 0
            df['clip_score'] = 0
            df['category_outdoor'] = 0
            df['category_indoor'] = 0
            df['category_mixed'] = 0
            df['category_floorplan'] = 0
    else:
        df['has_image'] = 0
        for i in range(CFG.projection_dim):
            df[f'img_embed_{i}'] = 0
        df['clip_score'] = 0
        df['category_outdoor'] = 0
        df['category_indoor'] = 0
        df['category_mixed'] = 0
        df['category_floorplan'] = 0
    
    poly_cols = ['bedroom', 'bathroom', 'distance_from_station', 'Median_rent_weekly']
    poly_cols = [col for col in poly_cols if col in df.columns]
    if poly_cols:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_features = poly.fit_transform(df[poly_cols].fillna(0))
        poly_names = [f"poly_{name.replace(' ', '_').replace('^', '_pow')}" for name in poly.get_feature_names_out(poly_cols)]
        poly_df = pd.DataFrame(poly_features, columns=poly_names, index=df.index)
        df = pd.concat([df, poly_df], axis=1)
    
    df['years_since_sale'] = df['sold_date'].apply(
        lambda x: current_year - pd.to_datetime(x, format='%a %d-%b-%y', errors='coerce').year if pd.notna(x) else 0
    )
    
    le = LabelEncoder()
    categorical_cols = ['property_type', 'Locality_x', 'geo_cluster', 'category', 'sale_season']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].fillna('Other'))
    
    X = df.drop(columns=['price', 'sold_date', 'description_heading', 'description_content', 
                         'features', 'inflation_factor', 'sold_year', 'ID'], errors='ignore')
    
    if X.columns.duplicated().any():
        cols = pd.Series(X.columns)
        for dup in cols[cols.duplicated()].unique():
            cols[cols[cols == dup].index.values.tolist()] = [f"{dup}_{i}" if i != 0 else dup for i in range(sum(cols == dup))]
        X.columns = cols
    
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X[numeric_cols].median())
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X[numeric_cols])
    X[numeric_cols] = X_scaled
    
    for col in X.columns:
        if X[col].dtype == 'object' and col != 'sale_season':
            logger.warning(f"Converting {col} from object to int")
            X[col] = X[col].astype(int)
    
    return X, y, le, scaler