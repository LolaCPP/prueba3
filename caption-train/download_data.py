
from datasets import load_dataset
from pathlib import Path

ds = load_dataset("LolaCPP/dataset_captioning_TFG")

# Si el dataset tiene split train, usamos ese
if isinstance(ds, dict) and "train" in ds:
    ds = ds["train"]

output_dir = Path("./data/dataset_captioning_TFG")
output_dir.mkdir(parents=True, exist_ok=True)

print(ds)
print(ds.column_names)

for i, example in enumerate(ds):
    # Ajusta estos nombres si tus columnas se llaman distinto
    image = example["image"]

    if "text" in example:
        caption = example["text"]
    elif "caption" in example:
        caption = example["caption"]
    elif "captions" in example:
        caption = example["captions"]
    else:
        raise ValueError(f"No encuentro columna de caption. Columnas: {ds.column_names}")

    image_path = output_dir / f"{i:06d}.jpg"
    caption_path = output_dir / f"{i:06d}.txt"

    image.save(image_path)
    caption_path.write_text(str(caption), encoding="utf-8")

print(f"Exportadas {len(ds)} imágenes a {output_dir}")