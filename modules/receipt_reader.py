"""
Receipt Reader Module
---------------------
AI Backends:
1. Groq API - LLaMA 4 Scout Vision 
2. Donut CORD-v2 - local CPU model
"""

import json
import re
import base64
import time
from io import BytesIO
from typing import Optional

from PIL import Image
from modules.models import BillData, BillItem, AdditionalCharge


EXTRACTION_PROMPT = """
You are a receipt data extractor. Analyze the receipt image carefully.
Return ONLY a valid JSON object. No markdown, no explanation, no code fences.

Required format:
{
  "items": [
    {
      "name": "item name here",
      "quantity": 1,
      "price_per_item": 10000,
      "total_price": 10000
    }
  ],
  "subtotal": 50000,
  "additional_charges": [
    {"name": "Tax 11%", "amount": 5500},
    {"name": "Service Charge", "amount": 2500}
  ],
  "total": 58000
}

Rules:
- All prices must be plain numbers, no currency symbols.
- quantity defaults to 1 if unclear.
- price_per_item = total_price / quantity if not shown.
- subtotal = sum of all item total_prices BEFORE tax/service.
- additional_charges includes tax, service charge, discounts (negative for discounts).
- total = subtotal + all additional_charges combined.
- Return ONLY the JSON object. Start with { and end with }. Nothing else.
"""


def _image_to_base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _parse_number(s) -> float:
    if isinstance(s, (int, float)):
        return float(s)
    cleaned = re.sub(r"[^\d.]", "", str(s))
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def _clean_json(text: str) -> str:
    """Clean model output to extract valid JSON."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = re.sub(r"//.*?\n", "\n", text)
    return text


def _dict_to_billdata(raw: dict) -> BillData:
    bill = BillData()
    for it in raw.get("items", []):
        bill.items.append(BillItem(
            name=str(it.get("name", "Unknown Item")),
            quantity=float(it.get("quantity", 1)),
            price_per_item=float(it.get("price_per_item", 0)),
            total_price=float(it.get("total_price", 0)),
        ))
    bill.subtotal = float(raw.get("subtotal", 0))
    for ch in raw.get("additional_charges", []):
        bill.additional_charges.append(AdditionalCharge(
            name=str(ch.get("name", "Extra Charge")),
            amount=float(ch.get("amount", 0)),
        ))
    bill.total = float(raw.get("total", 0))
    if bill.total == 0:
        bill.total = bill.recalculate_total()
    if bill.subtotal == 0:
        bill.subtotal = sum(i.total_price for i in bill.items)
    return bill


def _call_groq(client, b64: str, prompt: str, temperature: float = 0.1) -> str:
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
        max_tokens=1500,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def _read_with_groq(image: Image.Image, api_key: str) -> tuple[dict, float]:
    from groq import Groq

    b64 = _image_to_base64(image)
    client = Groq(api_key=api_key)
    print(f"DEBUG key: {api_key[:8]}...")

    start = time.time()
    raw_text = _clean_json(_call_groq(client, b64, EXTRACTION_PROMPT, temperature=0.1))

    try:
        result = json.loads(raw_text)
        elapsed = time.time() - start
        return result, elapsed
    except json.JSONDecodeError:
        strict_prompt = EXTRACTION_PROMPT + "\n\nIMPORTANT: Output ONLY the JSON object starting with { and ending with }. Absolutely nothing else."
        raw_text2 = _clean_json(_call_groq(client, b64, strict_prompt, temperature=0.0))
        elapsed = time.time() - start
        try:
            return json.loads(raw_text2), elapsed
        except json.JSONDecodeError as e:
            raise ValueError(f"Model returned invalid JSON after retry.\nError: {e}\nRaw: {raw_text2}")


def _read_with_donut(image: Image.Image) -> tuple[dict, float]:
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    import torch

    model_name = "naver-clova-ix/donut-base-finetuned-cord-v2"
    processor = DonutProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.eval()

    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    pixel_values = processor(image, return_tensors="pt").pixel_values

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
    elapsed = time.time() - start

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(
        processor.tokenizer.pad_token, ""
    )
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
    parsed = processor.token2json(sequence)

    items = []
    subtotal = 0.0
    menu_items = parsed.get("menu", [])
    if isinstance(menu_items, dict):
        menu_items = [menu_items]
    for m in menu_items:
        qty = _parse_number(m.get("cnt", "1"))
        total_price = _parse_number(m.get("price", "0"))
        unit_price = _parse_number(m.get("unitprice", str(total_price)))
        if qty > 0 and unit_price == 0 and total_price > 0:
            unit_price = total_price / qty
        if unit_price > 0 and total_price == 0:
            total_price = unit_price * qty
        subtotal += total_price
        items.append({
            "name": m.get("nm", "Unknown Item"),
            "quantity": qty,
            "price_per_item": unit_price,
            "total_price": total_price,
        })

    total_data = parsed.get("total", {})
    if isinstance(total_data, list):
        total_data = total_data[0] if total_data else {}
    grand_total = _parse_number(total_data.get("total_price", str(subtotal)))
    tax = _parse_number(total_data.get("tax_price", "0"))
    charges = [{"name": "Tax", "amount": tax}] if tax > 0 else []

    return {
        "items": items,
        "subtotal": subtotal,
        "additional_charges": charges,
        "total": grand_total if grand_total > 0 else subtotal + sum(c["amount"] for c in charges),
    }, elapsed


def read_receipt(
    image: Image.Image,
    backend: str = "groq",
    groq_api_key: Optional[str] = None,
) -> tuple[BillData, float]:
    if backend == "groq":
        if not groq_api_key:
            raise ValueError("Groq API key not found. Please set GROQ_API_KEY in .streamlit/secrets.toml")
        raw, elapsed = _read_with_groq(image, groq_api_key)
    elif backend == "donut":
        raw, elapsed = _read_with_donut(image)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Use 'groq' or 'donut'.")

    return _dict_to_billdata(raw), elapsed
