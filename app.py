import io
import json
import os
import re

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from rapidfuzz import fuzz, process

st.set_page_config(page_title="Handwritten Medical Notes OCR", page_icon="📝")

# Gemini model names are tried in order (Google renames models often).
# Override the first choice with GEMINI_MODEL in Streamlit Secrets if needed.
FALLBACK_MODELS = ["gemini-flash-latest", "gemini-flash-lite-latest", "gemini-3.6-flash", "gemini-2.5-flash"]

ABBREVIATIONS = {
    "OD": "once a day", "BD": "twice a day", "BID": "twice a day", "TDS": "three times a day",
    "TID": "three times a day", "QID": "four times a day", "HS": "at bedtime",
    "SOS": "only if needed", "PRN": "only if needed", "AC": "before food", "PC": "after food",
    "STAT": "immediately", "TAB": "tablet", "CAP": "capsule", "SYP": "syrup", "INJ": "injection",
}

PROMPT = """You are transcribing a handwritten medical note or prescription from an image.

Rules:
- Transcribe EXACTLY what is written. Never invent or "correct" drug names, doses or numbers.
- If a word is hard to read, give your best reading followed by [?], e.g. Amoxicillin[?].
- If something is completely unreadable, write [illegible].
- Keep the original line breaks in full_text (use \\n).
- Keep abbreviations (BD, TDS, OD, SOS, tab, cap, etc.) as written in full_text,
  and put the plain-English meaning only in the medicines list.

Return ONLY valid JSON in this shape:
{
  "full_text": "",
  "patient": "",
  "doctor": "",
  "date": "",
  "diagnosis_or_notes": "",
  "medicines": [
    {"name": "", "dose": "", "frequency": "", "duration": "", "remarks": "", "confidence": "high|medium|low"}
  ]
}
Use empty strings for fields that are not present."""


# ---------------- Engine 1: Gemini (free tier) ----------------
@st.cache_resource
def get_gemini():
    try:
        key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None, "GEMINI_API_KEY was not found in Streamlit Secrets"
    try:
        from google import genai
        return genai.Client(api_key=key), ""
    except Exception as e:
        return None, f"could not load the google-genai library: {e}"


def gemini_models():
    models = list(FALLBACK_MODELS)
    try:
        custom = st.secrets["GEMINI_MODEL"]
        models = [custom] + [m for m in models if m != custom]
    except Exception:
        pass
    return models


def run_gemini(client, pil_image):
    from google.genai import types

    img = ImageOps.exif_transpose(pil_image).convert("RGB")
    img.thumbnail((2000, 2000))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    jpeg = buf.getvalue()

    last_error = None
    for model in gemini_models():
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), PROMPT],
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0),
            )
            raw = (resp.text or "").strip()
            raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = {"full_text": raw}
            return {
                "engine": f"Gemini AI ({model})",
                "text": data.get("full_text", ""),
                "medicines": data.get("medicines") or [],
                "details": {k: data.get(k, "") for k in ("patient", "doctor", "date", "diagnosis_or_notes")},
                "words": [],
            }
        except Exception as e:
            last_error = e
    raise RuntimeError(last_error)


# ---------------- Engine 2: Local OCR (EasyOCR) ----------------
@st.cache_resource(show_spinner="Loading local OCR model (first run only)...")
def load_reader():
    import easyocr  # imported lazily so the app starts fast when Gemini works
    return easyocr.Reader(["en"], gpu=False)


def make_variants(pil_image, multi=True):
    img = np.array(ImageOps.exif_transpose(pil_image).convert("RGB"))
    h, w = img.shape[:2]
    scale = 1600 / max(h, w)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    variants = {"Original": gray}
    if multi:
        contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        denoised = cv2.fastNlMeansDenoising(contrast, None, 10, 7, 21)
        bw = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
        variants["Contrast + denoise"] = denoised
        variants["Black & white"] = bw
    return variants


def group_lines(results):
    items = []
    for bbox, text, conf in results:
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        items.append({"text": text, "x": min(xs), "yc": sum(ys) / 4, "h": max(ys) - min(ys)})
    items.sort(key=lambda i: i["yc"])
    lines = []
    for it in items:
        if lines and abs(it["yc"] - lines[-1]["yc"]) < 0.6 * max(it["h"], lines[-1]["h"]):
            ln = lines[-1]
            ln["items"].append(it)
            ln["yc"] = sum(i["yc"] for i in ln["items"]) / len(ln["items"])
            ln["h"] = max(ln["h"], it["h"])
        else:
            lines.append({"items": [it], "yc": it["yc"], "h": it["h"]})
    return "\n".join(" ".join(i["text"] for i in sorted(l["items"], key=lambda i: i["x"])) for l in lines)


def run_local(pil_image, multi):
    reader = load_reader()
    best = None
    for name, arr in make_variants(pil_image, multi).items():
        res = reader.readtext(arr, detail=1, paragraph=False, text_threshold=0.5, low_text=0.3)
        score = sum(conf for _, _, conf in res)
        if best is None or score > best[0]:
            best = (score, name, res)
    _, variant, res = best
    return {
        "engine": f"Local OCR ({variant})",
        "text": group_lines(res),
        "medicines": [],
        "details": {"patient": "", "doctor": "", "date": "", "diagnosis_or_notes": ""},
        "words": [(t, round(float(c), 2)) for _, t, c in res],
    }


# ---------------- Smart pipeline ----------------
def extract(pil_image, use_gemini, multi):
    note = ""
    if use_gemini:
        client, err = get_gemini()
        if client is None:
            note = f"Gemini is not available ({err}), so the local OCR was used."
        else:
            try:
                return run_gemini(client, pil_image) | {"note": ""}
            except Exception as e:
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    note = "Gemini free limit reached, so the local OCR was used instead."
                else:
                    note = f"Gemini failed, so the local OCR was used instead. Reason: {msg[:400]}"
    try:
        return run_local(pil_image, multi) | {"note": note}
    except Exception as e:
        raise RuntimeError(f"{note} Local OCR also failed: {e}".strip())


# ---------------- Helpers ----------------
@st.cache_data
def load_drugs():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drugs.txt")
    try:
        with open(path, encoding="utf-8") as f:
            return sorted({l.strip().lower() for l in f if l.strip()})
    except FileNotFoundError:
        return []


def suggest_drugs(text, drugs):
    rows, seen = [], set()
    for tok in re.findall(r"[A-Za-z]{4,}", text):
        low = tok.lower()
        if low in seen or low in drugs:
            seen.add(low)
            continue
        seen.add(low)
        match = process.extractOne(low, drugs, scorer=fuzz.WRatio, score_cutoff=78)
        if match:
            rows.append({"Apply": False, "Written": tok, "Suggested drug": match[0], "Match %": round(match[1])})
    return rows


def apply_replacements(text, pairs):
    for old, new in pairs:
        text = re.sub(rf"\b{re.escape(old)}\b", new, text, flags=re.IGNORECASE)
    return text


# ---------------- UI ----------------
st.title("📝 Handwritten Medical Notes OCR")
st.write("Upload an image of a handwritten prescription, and we'll extract the text for you.")

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

with st.expander("⚙️ Settings", expanded=False):
    use_gemini = st.checkbox(
        "Use Gemini AI first (more accurate, but the image is sent to Google)", value=True
    )
    multi = st.checkbox("Local OCR: try image enhancements (slower, better)", value=True)
    st.caption(
        "If Gemini hits its free limit or is unavailable, the app automatically falls back to local OCR. "
        "Untick the first box to keep everything private on this server. "
        "Don't upload real patient details to the free Gemini tier."
    )

uploaded = st.file_uploader(
    "Upload an image (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
    key=f"uploader_{st.session_state.uploader_key}",
)

if uploaded:
    file_id = f"{uploaded.name}-{uploaded.size}-{use_gemini}-{multi}"
    pil = Image.open(uploaded)

    if st.session_state.get("result_id") != file_id:
        with st.spinner("Reading handwriting... ⏳"):
            try:
                st.session_state.result = extract(pil, use_gemini, multi)
            except Exception as e:
                st.error(f"Could not read the image: {e}")
                st.stop()
        st.session_state.result_id = file_id

    result = st.session_state.result
    st.image(ImageOps.exif_transpose(pil), caption="Uploaded Image", use_container_width=True)

    if result["note"]:
        st.info(result["note"])
    st.caption(f"Engine used: {result['engine']}")

    st.subheader("📝 Extracted Text")
    st.caption("[?] marks uncertain words and [illegible] marks unreadable ones.")
    st.text(result["text"] or "(nothing detected)")

    drugs = load_drugs()
    pairs = []
    rows = suggest_drugs(result["text"], drugs) if drugs else []
    if rows:
        st.subheader("💊 Possible medicine-name corrections")
        st.caption("Tick only the ones that are right. Add more names to drugs.txt to improve this.")
        edited = st.data_editor(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True,
            disabled=["Written", "Suggested drug", "Match %"],
            key=f"sugg_{file_id}",
        )
        pairs = [(r["Written"], r["Suggested drug"]) for _, r in edited.iterrows() if r["Apply"]]

    st.subheader("✏️ Final text (editable)")
    final_text = st.text_area(
        "Final text",
        value=apply_replacements(result["text"], pairs),
        height=220,
        key=f"final_{file_id}_{hash(tuple(pairs))}",
        label_visibility="collapsed",
    )

    meds = result["medicines"]
    if meds:
        st.subheader("💊 Medicines detected")
        st.dataframe(pd.DataFrame(meds), hide_index=True, use_container_width=True)

    details = result["details"]
    if any(details.values()):
        st.subheader("🗂️ Other details")
        for k, v in details.items():
            if v:
                st.write(f"**{k.replace('_', ' ').title()}:** {v}")

    found = []
    for tok in re.findall(r"[A-Za-z]+", final_text):
        if tok.upper() in ABBREVIATIONS and tok.upper() not in [f["Abbreviation"] for f in found]:
            found.append({"Abbreviation": tok.upper(), "Meaning": ABBREVIATIONS[tok.upper()]})
    if found:
        st.subheader("📖 Abbreviations found")
        st.dataframe(pd.DataFrame(found), hide_index=True, use_container_width=True)

    low = [(w, c) for w, c in result["words"] if c < 0.5]
    if low:
        with st.expander(f"⚠️ {len(low)} low-confidence words (check these carefully)"):
            st.dataframe(pd.DataFrame(low, columns=["Word", "Confidence"]), hide_index=True)

    st.warning("⚠️ OCR can make mistakes. Always confirm medicine names and doses with the doctor or a pharmacist.")

    out = pd.DataFrame(
        [
            {
                "Image": uploaded.name,
                "Extracted Text": final_text,
                "Patient": details["patient"],
                "Doctor": details["doctor"],
                "Date": details["date"],
                "Medicines": "; ".join(
                    " ".join(filter(None, [m.get("name"), m.get("dose"), m.get("frequency"), m.get("duration")]))
                    for m in meds
                ),
                "Engine": result["engine"],
            }
        ]
    )
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "✅ Download CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="extracted_text.csv",
            mime="text/csv",
        )
    with col2:
        if st.button("❌ Start Over"):
            st.session_state.uploader_key += 1
            st.session_state.pop("result", None)
            st.session_state.pop("result_id", None)
            st.rerun()
