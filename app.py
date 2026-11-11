import hashlib
import hmac
import io
import json
import os
import re

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps
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
Some areas may be blacked out for privacy. Ignore the black boxes.

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

# Values that must be checked by a human before download
RISK = re.compile(
    r"(?<![\d/])\d{2,3}\s*/\s*\d{2,3}(?![\d/])"                                   # blood pressure like 110/70
    r"|\d+(?:\.\d+)?\s*(?:mg/dl|mmol/l|mmhg|mcg|mg|ug|ml|iu|units?|sachets?|bpm|g|%)"  # doses / readings
    r"|\S*\[\?\]|\[illegible\]",                                                  # uncertain words
    re.IGNORECASE,
)

PRIVATE_LABELS = ["name", "uhid", "patient", "mrn", "phone", "mobile", "address", "regno", "ipno"]


# ---------------- Password gate ----------------
def password_gate():
    """Returns True if the app has a password set. Stops the script until it is entered."""
    try:
        pw = str(st.secrets["APP_PASSWORD"])
    except Exception:
        return False
    if st.session_state.get("auth"):
        return True
    entered = st.text_input("🔒 Password", type="password")
    if entered:
        if hmac.compare_digest(entered, pw):
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("Wrong password.")
    st.stop()


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

    img = pil_image.convert("RGB")
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
    import easyocr  # imported lazily
    return easyocr.Reader(["en"], gpu=False)


def make_variants(pil_image, multi=True):
    img = np.array(pil_image.convert("RGB"))
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


# ---------------- Privacy: black out patient details BEFORE anything is sent ----------------
def redact(pil_image, auto, bands):
    """Black out Name/UHID/etc. lines (found locally) and any manual bands. Returns (image, labels_found)."""
    img = ImageOps.exif_transpose(pil_image).convert("RGB")
    W, H = img.size
    found = 0
    draw = ImageDraw.Draw(img)
    if auto:
        scale = 1600 / max(W, H)
        small = img.resize((max(1, int(W * scale)), max(1, int(H * scale))))
        for bbox, text, conf in load_reader().readtext(np.array(small), detail=1, paragraph=False):
            t = re.sub(r"[^a-z]", "", text.lower())
            if len(t) >= 3 and any(l in t or fuzz.ratio(t, l) >= 80 for l in PRIVATE_LABELS):
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                x1, y1, y2 = min(xs) / scale, min(ys) / scale, max(ys) / scale
                h = y2 - y1
                draw.rectangle([max(0, x1 - 5), max(0, y1 - 0.8 * h), W, min(H, y2 + 1.5 * h)], fill="black")
                found += 1
    for lo, hi in bands:
        if hi > lo:
            draw.rectangle([0, H * lo / 100, W, H * hi / 100], fill="black")
    return img, found


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
has_password = password_gate()

st.write("Upload an image of a handwritten prescription, and we'll extract the text for you.")

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

with st.expander("⚙️ Settings", expanded=False):
    use_gemini = st.checkbox(
        "Use Gemini AI first (more accurate, but the image is sent to Google)", value=True
    )
    multi = st.checkbox("Local OCR: try image enhancements (slower, better)", value=True)
    auto_hide = st.checkbox("Automatically black out Name / UHID / patient ID lines before processing", value=True)
    st.caption("Manual blackout bands (% of image height, top to bottom). Use these for anything private the auto-hide misses.")
    band1 = st.slider("Blackout band 1", 0, 100, (0, 0))
    band2 = st.slider("Blackout band 2", 0, 100, (0, 0))
    st.caption(
        "If Gemini hits its free limit or is unavailable, the app falls back to local OCR. "
        "Untick the first box to keep everything on this server."
    )
    if not has_password:
        st.warning("🔓 No APP_PASSWORD secret is set, so anyone with the link can use this app and your free quota.")

uploaded = st.file_uploader(
    "Upload an image (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
    key=f"uploader_{st.session_state.uploader_key}",
)

if uploaded:
    bands = (band1, band2)
    red_key = f"{uploaded.name}-{uploaded.size}-{auto_hide}-{bands}"
    file_id = f"{red_key}-{use_gemini}-{multi}"

    # Step 1: redact locally and show exactly what would be sent
    if st.session_state.get("red_key") != red_key:
        with st.spinner("Hiding patient details... ⏳"):
            try:
                img, found = redact(Image.open(uploaded), auto_hide, bands)
            except Exception as e:
                st.error(f"Could not prepare the image: {e}")
                st.stop()
        st.session_state.red_img, st.session_state.red_found = img, found
        st.session_state.red_key = red_key

    send_img = st.session_state.red_img
    st.image(send_img, caption="This is the image that will be processed (black boxes are hidden)", use_container_width=True)
    if auto_hide and st.session_state.red_found == 0:
        st.warning("Couldn't find Name/ID labels automatically. Check the image above, and use the manual blackout bands in Settings if anything private is visible.")
    elif auto_hide:
        st.caption(f"Hid {st.session_state.red_found} patient-detail line(s). Check the image above before continuing.")

    # Step 2: only extract when the user clicks
    if st.button("🔍 Extract text", type="primary"):
        with st.spinner("Reading handwriting... ⏳"):
            try:
                st.session_state.result = extract(send_img, use_gemini, multi)
            except Exception as e:
                st.error(f"Could not read the image: {e}")
                st.stop()
        st.session_state.result_id = file_id

    if st.session_state.get("result_id") == file_id:
        result = st.session_state.result

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

        found_abbr = []
        for tok in re.findall(r"[A-Za-z]+", final_text):
            if tok.upper() in ABBREVIATIONS and tok.upper() not in [f["Abbreviation"] for f in found_abbr]:
                found_abbr.append({"Abbreviation": tok.upper(), "Meaning": ABBREVIATIONS[tok.upper()]})
        if found_abbr:
            st.subheader("📖 Abbreviations found")
            st.dataframe(pd.DataFrame(found_abbr), hide_index=True, use_container_width=True)

        low = [(w, c) for w, c in result["words"] if c < 0.5]
        if low:
            with st.expander(f"⚠️ {len(low)} low-confidence words (check these carefully)"):
                st.dataframe(pd.DataFrame(low, columns=["Word", "Confidence"]), hide_index=True)

        # Verification of risky values
        risky = []
        for m in RISK.finditer(final_text):
            tok = re.sub(r"\s+", " ", m.group(0)).strip()
            if tok not in risky:
                risky.append(tok)

        all_verified = True
        if risky:
            st.subheader("✅ Verify before download")
            st.caption("Compare each value with the original note, then tick it. The download unlocks when all are ticked.")
            for tok in risky:
                key = f"ver_{hashlib.md5((file_id + tok).encode()).hexdigest()[:12]}"
                if not st.checkbox(tok, key=key):
                    all_verified = False

        st.warning("⚠️ OCR can make mistakes. Always confirm medicine names and doses with the doctor or a pharmacist.")

        out = pd.DataFrame(
            [
                {
                    "Image": uploaded.name,
                    "Extracted Text": final_text,
                    "Date": details["date"],
                    "Doctor": details["doctor"],
                    "Medicines": "; ".join(
                        " ".join(filter(None, [m.get("name"), m.get("dose"), m.get("frequency"), m.get("duration")]))
                        for m in meds
                    ),
                    "Engine": result["engine"],
                    "Values verified": "yes" if all_verified else "no",
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
                disabled=not all_verified,
            )
            if not all_verified:
                st.caption("Tick every value above to enable the download.")
        with col2:
            if st.button("❌ Start Over"):
                st.session_state.uploader_key += 1
                for k in ("result", "result_id", "red_key", "red_img", "red_found"):
                    st.session_state.pop(k, None)
                st.rerun()
