from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from datasets import load_dataset
import pandas as pd
import os
import util
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = str(util.get_free_gpu())

def define_model():
    model_name = "facebook/nllb-200-1.3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to("cuda")
    return tokenizer, model

def define_data():
    dataset = load_dataset("allenai/wildjailbreak", "eval")["train"]
    filtered = dataset.filter(lambda x: x["label"] == 1).remove_columns(["data_type", "label"])
    return filtered

if __name__ == "__main__":
    tokenizer, model = define_model()
    dataset = define_data()
    entries = [{"language": "English", "text": x["adversarial"]} for x in dataset]
    language_map = {
        "German": "deu_Latn",
        "Chinese": "zho_Hans",
        "Hindi": "hin_Deva",
        "Arabic": "arb_Arab",
        "Japanese": "jpn_Jpan",
        "Russian": "rus_Cyrl",
        "Spanish": "spa_Latn",
        "French": "fra_Latn",
        "Bengali": "ben_Beng"
    }
    batch_size=8
    texts = [x["adversarial"] for x in dataset]
    for lang, code in language_map.items():
        print(f"Translating to {lang}...")
        translator = pipeline(
            task="translation",
            model=model,
            tokenizer=tokenizer,
            src_lang="eng_Latn",
            tgt_lang=code,
            max_length=1000,
            device=0
        )
        translations = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]
            outputs = translator(batch, batch_size=batch_size)
            translations.extend([output['translation_text'] for output in outputs])
        # Store translations
        for translated_text in translations:
            entries.append({"language": lang, "text": translated_text})

    df = pd.DataFrame(entries)
    df.to_csv("translated_all.csv", index=False)
    print("Dataset saved as 'translated_all.csv'")
