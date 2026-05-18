import pandas as pd
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os
import subprocess
import gc
import json
from logger import logger

torch.cuda.empty_cache()
gc.collect()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

def log_gpu_memory():
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv'])
        logger.info("GPU Memory Usage:\n" + result.decode('utf-8'))
    except Exception as e:
        logger.warning(f"Failed to log GPU memory: {str(e)}")

model_name = "meta-llama/Llama-2-7b-hf"
try:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loaded LLaMA on {device}")
except Exception as e:
    logger.error(f"Failed to load LLaMA: {str(e)}")
    raise
log_gpu_memory()


def extract_context_llama(row, row_idx, total_rows, max_retries=5):
    if row_idx % 100 == 0:
        logger.info(f"Processed {row_idx}/{total_rows} rows")
    text = f"{row.get('description_heading', '')} {row.get('description_content', '')} {row.get('features', '')}".lower()
    text = re.sub(r'[^\w\s]', '', text)
    if not text.strip():
        logger.warning(f"Row {row_idx}: Empty description, using rule-based fallback")
        return extract_context_rule_based(row)
    
    required_keys = [
        'proximity_minutes', 'proximity_km', 'near_shops', 'near_transport', 'near_schools',
        'is_quiet', 'is_busy', 'is_central', 'has_view', 'north_facing', 'urgency_level',
        'is_recently_renovated', 'is_new', 'is_old', 'has_luxury_finishes', 'sale_season'
    ]
    
    sale_season = row.get('sold_date', '')
    if sale_season:
        try:
            sale_month = pd.to_datetime(sale_season, format='%a %d-%b-%y', errors='coerce').month
            sale_season = (
                'spring' if sale_month in [9, 10, 11] else
                'summer' if sale_month in [12, 1, 2] else
                'fall' if sale_month in [3, 4, 5] else
                'winter' if sale_month in [6, 7, 8] else
                'none'
            )
        except:
            sale_season = 'none'
    else:
        sale_season = 'none'
    
    prompt = f"""
    **INSTRUCTIONS**: Output ONLY a single JSON object with the specified fields based on the provided description. Do NOT include any examples, explanations, or extra text. Use 0 or 'none' for unknown values. Ensure the JSON is valid and contains exactly the fields listed below.

    **Fields**:
    - proximity_minutes (float): Minutes to nearest amenity (5.0 for "close to" or "nearby", 0.0 if not mentioned).
    - proximity_km (float): Kilometers to nearest amenity (1.0 for "close to" or "nearby", 0.0 if not mentioned).
    - near_shops (int): 1 if close to shops, 0 otherwise.
    - near_transport (int): 1 if close to transport, 0 otherwise.
    - near_schools (int): 1 if close to schools, 0 otherwise.
    - is_quiet (int): 1 if described as quiet or peaceful, 0 otherwise.
    - is_busy (int): 1 if described as busy or vibrant, 0 otherwise.
    - is_central (int): 1 if described as central or in the heart of an area, 0 otherwise.
    - has_view (int): 1 if mentions view, views, scenic, ocean, park, or lakeview, 0 otherwise.
    - north_facing (int): 1 if described as north-facing, 0 otherwise.
    - urgency_level (int): 0 (none), 1 (urgent or immediate), 2 (must sell or quick sale).
    - is_recently_renovated (int): 1 if described as renovated, updated, or modern, 0 otherwise.
    - is_new (int): 1 if described as new or brand new, 0 otherwise.
    - is_old (int): 1 if described as old or older, 0 otherwise.
    - has_luxury_finishes (int): 1 if described as luxury, premium, or high-end, 0 otherwise.
    - sale_season (str): "spring", "summer", "fall", "winter", or "none". Use: {sale_season}.

    **Description**: {text}
    """
    
    for attempt in range(max_retries):
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024, padding=True).to(device)
            outputs = model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.1)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        except Exception as e:
            logger.warning(f"Row {row_idx}, Attempt {attempt+1}: LLaMA generation failed ({str(e)}), retrying...")
            if attempt == max_retries - 1:
                logger.error(f"Row {row_idx}: All retries failed, falling back to rule-based. Description: {text[:100]}...")
                return extract_context_rule_based(row)
            continue
        
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                logger.warning(f"Row {row_idx}, Attempt {attempt+1}: No valid JSON found, retrying...")
                continue
            json_str = response[json_start:json_end]
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r'[\n\r]+', ' ', json_str)
            json_str = re.sub(r"'", '"', json_str)
            json_str = re.sub(r'(\w+):', r'"\1":', json_str)
            json_str = re.sub(r'"\s*,', '",', json_str)
            json_str = re.sub(r',\s*"', ',"', json_str)
            if not json_str.startswith('{'):
                json_str = '{' + json_str.lstrip('{')
            if not json_str.endswith('}'):
                json_str = json_str.rstrip(',') + '}'
            
            try:
                features = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Row {row_idx}, Attempt {attempt+1}: JSON decode failed ({str(e)}), attempting partial parse")
                features = {}
                pairs = re.findall(r'"(\w+)":\s*([0-9.]+\s*|"[^"]*"|[0-2])', json_str)
                for key, value in pairs:
                    if key in required_keys:
                        try:
                            value = value.strip()
                            if key in ['proximity_minutes', 'proximity_km']:
                                features[key] = float(value)
                            elif key == 'sale_season':
                                value = value.strip('"')
                                features[key] = value if value in ['spring', 'summer', 'fall', 'winter', 'none'] else 'none'
                            elif key == 'urgency_level':
                                features[key] = int(value) if value in ['0', '1', '2'] else 0
                            else:
                                features[key] = int(value)
                        except (ValueError, TypeError):
                            continue
                for key in required_keys:
                    if key not in features:
                        if key in ['proximity_minutes', 'proximity_km']:
                            features[key] = 0.0
                        elif key == 'sale_season':
                            features[key] = 'none'
                        else:
                            features[key] = 0
            
            if not all(k in features for k in required_keys):
                missing = set(required_keys) - set(features.keys())
                logger.warning(f"Row {row_idx}, Attempt {attempt+1}: Missing keys {missing}, retrying...")
                continue
            
            feature_values = []
            for key in required_keys:
                if key in ['proximity_minutes', 'proximity_km']:
                    feature_values.append(float(features.get(key, 0.0)))
                elif key == 'sale_season':
                    value = features.get(key, 'none')
                    feature_values.append(value if value in ['spring', 'summer', 'fall', 'winter', 'none'] else 'none')
                elif key == 'urgency_level':
                    value = features.get(key, 0)
                    feature_values.append(int(value) if value in [0, 1, 2] else 0)
                else:
                    feature_values.append(int(features.get(key, 0)))
            
            if features.get('near_shops', 0) == 1 and not any(w in text for w in ['shops', 'shopping', 'supermarket', 'retail']):
                logger.warning(f"Row {row_idx}: near_shops=1 but no shop-related terms found. Retrying...")
                continue
            if features.get('near_schools', 0) == 1 and not any(w in text for w in ['schools', 'school', 'college']):
                logger.warning(f"Row {row_idx}: near_schools=1 but no school-related terms found. Retrying...")
                continue
            
            logger.info(f"Row {row_idx}: Extracted {len(feature_values)} features")
            return feature_values
        
        except Exception as e:
            logger.warning(f"Row {row_idx}, Attempt {attempt+1}: Failed to parse LLaMA response ({str(e)}), retrying...")
            continue
    
    logger.error(f"Row {row_idx}: All retries failed, falling back to rule-based. Description: {text[:100]}...")
    return extract_context_rule_based(row)


# Rule-based feature extraction
def extract_context_rule_based(row):
    text = f"{row.get('description_heading', '')} {row.get('description_content', '')} {row.get('features', '')}".lower()
    text = re.sub(r'[^\w\s]', '', text)
    minutes_match = re.search(r'(\d+\.?\d*)\s*(minute|min)', text)
    proximity_minutes = float(minutes_match.group(1)) if minutes_match else (5.0 if any(w in text for w in ['close to', 'nearby', 'walking distance']) else 0.0)
    proximity_km = 1.0 if any(w in text for w in ['close to', 'nearby', 'walking distance']) else 0.0
    near_shops = 1 if any(w in text for w in ['shops', 'shopping', 'supermarket', 'retail']) else 0
    near_transport = 1 if any(w in text for w in ['transport', 'train station', 'bus', 'tram']) else 0
    near_schools = 1 if any(w in text for w in ['schools', 'school', 'college']) else 0
    is_quiet = 1 if any(w in text for w in ['quiet', 'peaceful', 'tranquil']) else 0
    is_busy = 1 if any(w in text for w in ['busy', 'vibrant', 'bustling']) else 0
    is_central = 1 if any(w in text for w in ['central', 'heart of']) else 0
    has_view = 1 if any(w in text for w in ['view', 'views', 'scenic', 'ocean', 'park', 'lakeview']) else 0
    north_facing = 1 if 'north facing' in text else 0
    urgency_level = 2 if any(w in text for w in ['must sell', 'quick sale']) else 1 if any(w in text for w in ['urgent', 'immediate']) else 0
    is_recently_renovated = 1 if any(w in text for w in ['renovated', 'updated', 'modern', 'opulent', 'luxury']) else 0
    is_new = 1 if any(w in text for w in ['new', 'brand new']) else 0
    is_old = 1 if any(w in text for w in ['old', 'older', 'heritage']) else 0
    has_luxury_finishes = 1 if any(w in text for w in ['luxury', 'premium', 'high-end']) else 0
    sale_season = row.get('sold_date', '')
    if sale_season:
        try:
            sale_month = pd.to_datetime(sale_season, format='%a %d-%b-%y', errors='coerce').month
            sale_season = (
                'spring' if sale_month in [9, 10, 11] else
                'summer' if sale_month in [12, 1, 2] else
                'fall' if sale_month in [3, 4, 5] else
                'winter' if sale_month in [6, 7, 8] else
                'none'
            )
        except:
            sale_season = 'none'
    else:
        sale_season = 'none'
    return [
        proximity_minutes, proximity_km, near_shops, near_transport, near_schools,
        is_quiet, is_busy, is_central, has_view, north_facing, urgency_level,
        is_recently_renovated, is_new, is_old, has_luxury_finishes, sale_season
    ]
