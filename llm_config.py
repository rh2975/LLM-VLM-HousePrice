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
        You are an expert real-estate analyst. You read a property Listing Description
        from ANY country, market, or writing style and identify the KEY spatio-temporal
        and contextual features that influence the property's value.
        
        You are NOT being given a predefined list of features. Decide and extract which
        attributes are salient in each description. Consider only these two families:
        - Spatio-temporal: location relative to amenities and landmarks (shops, public
          transport, schools, parks, water, city, employment hubs), proximity and
          accessibility, orientation/aspect, time in market context, etc.
        - Contextual: condition, age, build quality, prestige, ambiance, layout and
          space, indoor/outdoor features, suitability, seller intent, etc.
        
        Rules for the JSON output:
        - Output ONLY a single flat JSON object. Keys are concise canonical
          feature names; values are numbers.
        - Use 1 for a present/true qualitative attribute. Use an actual number for a
          quantity (e.g. minutes, kilometres, counts).
        - Reuse the SAME canonical name for the same concept across different listings
          (e.g. always "near_station", never "close_to_station" in one and
          "station_nearby" in another) so features are compatible.
        - Prefer widely-applicable names over one-off phrases.

    **Listing Description**: {text}
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


_FALLBACK_KEYWORDS = {
    "near_shops": ["shops", "shopping", "supermarket", "retail", "mall"],
    "near_transport": ["transport", "train", "station", "bus", "tram", "metro", "ferry"],
    "near_schools": ["school", "college", "university", "campus"],
    "near_parks": ["park", "reserve", "greenery", "playground"],
    "near_water": ["beach", "river", "lake", "ocean", "waterfront", "canal", "seaside"],
    "near_cbd": ["cbd", "city centre", "city center", "downtown", "business district"],
    "quiet": ["quiet", "peaceful", "tranquil", "serene", "private"],
    "busy": ["busy", "vibrant", "bustling", "lively"],
    "central": ["central", "heart of", "inner city", "inner-city"],
    "has_view": ["view", "views", "scenic", "outlook", "skyline"],
    "north_facing": ["north facing", "north-facing", "northerly"],
    "recently_renovated": ["renovated", "updated", "refurbished", "modern", "contemporary"],
    "new_build": ["brand new", "newly built", "new home", "near new"],
    "period_or_old": ["older", "period", "heritage", "original condition"],
    "luxury_finishes": ["luxury", "premium", "high-end", "high end", "designer", "opulent"],
    "outdoor_space": ["garden", "yard", "courtyard", "balcony", "deck", "terrace", "patio", "alfresco"],
    "parking": ["garage", "carport", "parking", "off-street", "off street"],
    "family_friendly": ["family", "families", "child", "kids"],
    "investment_appeal": ["investment", "investor", "rental", "tenant", "yield"],
    "development_potential": ["subdivide", "subdivision", "dual occupancy", "duplex", "stca"],
    "energy_efficient": ["solar", "energy rating", "energy-efficient", "sustainable", "double glazing"],
    "move_in_ready": ["move in ready", "move-in ready", "turnkey", "nothing to do"],
    "needs_work": ["renovator", "fixer", "needs work", "potential to improve", "tlc"],
    "security_features": ["gated", "secure", "alarm", "intercom", "cctv"],
    "spacious": ["spacious", "generous", "expansive", "large", "roomy"],
}


def fallback_extract_features(text: str) -> dict:
    t = re.sub(r"[^\w\s]", " ", str(text).lower())
    out = {}
    for feat, kws in _FALLBACK_KEYWORDS.items():
        if any(w in t for w in kws):
            out[feat] = 1.0
    m = re.search(r"(\d+\.?\d*)\s*(minute|min)\b", t)
    if m:
        try:
            out["proximity_minutes"] = float(m.group(1))
        except ValueError:
            pass
    if any(w in t for w in ["must sell", "quick sale", "mortgagee", "urgent", "deceased estate"]):
        out["urgent_sale"] = 1.0
    return out
