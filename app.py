import hashlib
import hmac
import io
import json
import os
import re
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps
from rapidfuzz import fuzz, process

st.set_page_config(page_title="Handwritten Medical Notes OCR", page_icon="📝")

MAX_FILES = 10          # per batch
GEMINI_GAP_SECONDS = 4  # pause between Gemini calls to respect free-tier rate limits

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
<<LANG_RULE>>
Return ONLY valid JSON in this shape:
{
  "full_text": "",
  "patient": "",
  "doctor": "",
  "date": "",
  "diagnosis_or_notes": "",<<TRANSLATION_FIELD>>
  "medicines": [
    {"name": "", "dose": "", "frequency": "", "duration": "", "remarks": "", "confidence": "high|medium|low"}
  ]
}
Use empty strings for fields that are not present."""

LANG_RULE = (
    "- The note may contain English, Kannada, Telugu, Hindi or other Indian languages. Transcribe non-English "
    "text in its original script in full_text, and put an English translation of ONLY the non-English parts "
    "in translation_en (use an empty string if everything is English)."
)

# Values that must be checked by a human before download
RISK = re.compile(
    r"(?<![\d/])\d{2,3}\s*/\s*\d{2,3}(?![\d/])"                                        # blood pressure like 110/70
    r"|\d+(?:\.\d+)?\s*(?:mg/dl|mmol/l|mmhg|mcg|mg|ug|ml|iu|units?|sachets?|bpm|g|%)"  # doses / readings
    r"|\S*\[\?\]|\[illegible\]",                                                       # uncertain words
    re.IGNORECASE,
)

PRIVATE_LABELS = ["name", "uhid", "patient", "mrn", "phone", "mobile", "address", "regno", "ipno"]


def build_prompt(translate):
    p = PROMPT.replace("<<LANG_RULE>>", LANG_RULE if translate else "")
    return p.replace("<<TRANSLATION_FIELD>>", '\n  "translation_en": "",' if translate else "")


# ---------------- Password gate (optional) ----------------
def password_gate():
    """Returns True if a password is configured. Stops the script until it is entered."""
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


def run_gemini(client, pil_image, translate):
    from google.genai import types

    img = pil_image.convert("RGB")
    img.thumbnail((2000, 2000))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    jpeg = buf.getvalue()
    prompt = build_prompt(translate)

    last_error = None
    for model in gemini_models():
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), prompt],
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
                "details": {
                    k: data.get(k, "")
                    for k in ("patient", "doctor", "date", "diagnosis_or_notes", "translation_en")
                },
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
        "details": {"patient": "", "doctor": "", "date": "", "diagnosis_or_notes": "", "translation_en": ""},
        "words": [(t, round(float(c), 2)) for _, t, c in res],
    }


# ---------------- Privacy: black out patient details BEFORE anything is sent ----------------
def redact(pil_image, auto, bands):
    """Black out Name/UHID/etc. lines (found locally) and manual bands. Returns (jpeg_bytes, labels_found)."""
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
    img.thumbnail((2000, 2000))  # keep memory small: store compressed bytes, not big images
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue(), found


# ---------------- Smart pipeline ----------------
def extract(pil_image, use_gemini, multi, translate):
    note = ""
    if use_gemini:
        client, err = get_gemini()
        if client is None:
            note = f"Gemini is not available ({err}), so the local OCR was used."
        else:
            try:
                return run_gemini(client, pil_image, translate) | {"note": ""}
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


def to_excel_bytes(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Notes")
    return buf.getvalue()


def render_result(filename, file_id, result):
    """Shows one note's result. Returns (csv_row_dict, all_values_verified)."""
    if result["note"]:
        st.info(result["note"])
    st.caption(f"Engine used: {result['engine']}")

    st.markdown("**📝 Extracted text**")
    st.caption("[?] marks uncertain words and [illegible] marks unreadable ones.")
    st.text(result["text"] or "(nothing detected)")

    drugs = load_drugs()
    pairs = []
    rows = suggest_drugs(result["text"], drugs) if drugs else []
    if rows:
        st.markdown("**💊 Possible medicine-name corrections**")
        st.caption("Tick only the ones that are right. Add more names to drugs.txt to improve this.")
        edited = st.data_editor(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True,
            disabled=["Written", "Suggested drug", "Match %"],
            key=f"sugg_{file_id}",
        )
        pairs = [(r["Written"], r["Suggested drug"]) for _, r in edited.iterrows() if r["Apply"]]

    st.markdown("**✏️ Final text (editable)**")
    final_text = st.text_area(
        "Final text",
        value=apply_replacements(result["text"], pairs),
        height=220,
        key=f"final_{file_id}_{hash(tuple(pairs))}",
        label_visibility="collapsed",
    )

    details = result["details"]
    if details.get("translation_en"):
        st.markdown("**🌐 English translation of non-English text**")
        st.write(details["translation_en"])

    meds = result["medicines"]
    if meds:
        st.markdown("**💊 Medicines detected**")
        st.dataframe(pd.DataFrame(meds), hide_index=True, use_container_width=True)

    others = {k: v for k, v in details.items() if v and k not in ("translation_en", "patient")}
    if others:
        st.markdown("**🗂️ Other details**")
        for k, v in others.items():
            st.write(f"**{k.replace('_', ' ').title()}:** {v}")

    found_abbr = []
    for tok in re.findall(r"[A-Za-z]+", final_text):
        if tok.upper() in ABBREVIATIONS and tok.upper() not in [f["Abbreviation"] for f in found_abbr]:
            found_abbr.append({"Abbreviation": tok.upper(), "Meaning": ABBREVIATIONS[tok.upper()]})
    if found_abbr:
        st.markdown("**📖 Abbreviations found**")
        st.dataframe(pd.DataFrame(found_abbr), hide_index=True, use_container_width=True)

    low = [(w, c) for w, c in result["words"] if c < 0.5]
    if low:
        with st.expander(f"⚠️ {len(low)} low-confidence words (check these carefully)"):
            st.dataframe(pd.DataFrame(low, columns=["Word", "Confidence"]), hide_index=True)

    risky = []
    for m in RISK.finditer(final_text):
        tok = re.sub(r"\s+", " ", m.group(0)).strip()
        if tok not in risky:
            risky.append(tok)

    all_verified = True
    if risky:
        st.markdown("**✅ Verify before download**")
        st.caption("Compare each value with the original note, then tick it.")
        for tok in risky:
            key = f"ver_{hashlib.md5((file_id + tok).encode()).hexdigest()[:12]}"
            if not st.checkbox(tok, key=key):
                all_verified = False

    row = {
        "Image": filename,
        "Extracted Text": final_text,
        "English translation": details.get("translation_en", ""),
        "Date": details.get("date", ""),
        "Doctor": details.get("doctor", ""),
        "Medicines": "; ".join(
            " ".join(filter(None, [m.get("name"), m.get("dose"), m.get("frequency"), m.get("duration")]))
            for m in meds
        ),
        "Engine": result["engine"],
        "Values verified": "yes" if all_verified else "no",
    }
    return row, all_verified


# ---------------- UI ----------------
st.title("📝 Handwritten Medical Notes OCR")
has_password = password_gate()

st.write("Upload images of handwritten prescriptions or notes, and we'll extract the text for you.")

for k, default in (("uploader_key", 0), ("red", {}), ("results", {})):
    if k not in st.session_state:
        st.session_state[k] = default

with st.expander("⚙️ Settings", expanded=False):
    use_gemini = st.checkbox(
        "Use Gemini AI first (more accurate, but the image is sent to Google)", value=True
    )
    translate = st.checkbox("Read Kannada / Telugu / Hindi text and add an English translation", value=True)
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

files = st.file_uploader(
    f"Upload images (JPG/PNG, up to {MAX_FILES} at a time)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}",
)

if files:
    if len(files) > MAX_FILES:
        st.warning(f"Only the first {MAX_FILES} images will be used.")
        files = files[:MAX_FILES]

    bands = (band1, band2)
    entries = []  # (uploaded_file, red_key, file_id)

    # Step 1: redact each image locally and show exactly what would be sent
    for f in files:
        red_key = f"{f.name}-{f.size}-{auto_hide}-{bands}"
        file_id = f"{red_key}-{use_gemini}-{multi}-{translate}"
        if red_key not in st.session_state.red:
            with st.spinner(f"Hiding patient details in {f.name}... ⏳"):
                try:
                    st.session_state.red[red_key] = redact(Image.open(f), auto_hide, bands)
                except Exception as e:
                    st.error(f"Could not prepare {f.name}: {e}")
                    continue
        entries.append((f, red_key, file_id))

    for f, red_key, _ in entries:
        img_bytes, found = st.session_state.red[red_key]
        with st.expander(f"👁️ Preview of what will be processed: {f.name}", expanded=len(entries) == 1):
            st.image(img_bytes, caption="Black boxes are hidden before processing", use_container_width=True)
            if auto_hide and found == 0:
                st.warning("Couldn't find Name/ID labels automatically. If anything private is visible, use the manual blackout bands in Settings.")
            elif auto_hide:
                st.caption(f"Hid {found} patient-detail line(s). Check the image before continuing.")

    # Step 2: extract only when the user clicks
    pending = [e for e in entries if e[2] not in st.session_state.results]
    if pending:
        label = "🔍 Extract text" if len(entries) == 1 else f"🔍 Extract text from {len(pending)} image(s)"
        if st.button(label, type="primary"):
            progress = st.progress(0.0)
            status = st.empty()
            failures = []
            for i, (f, red_key, file_id) in enumerate(pending):
                status.write(f"Reading {i + 1} of {len(pending)}: {f.name} ⏳")
                try:
                    pil = Image.open(io.BytesIO(st.session_state.red[red_key][0]))
                    st.session_state.results[file_id] = extract(pil, use_gemini, multi, translate)
                except Exception as e:
                    failures.append(f"{f.name}: {e}")
                progress.progress((i + 1) / len(pending))
                if use_gemini and i < len(pending) - 1:
                    time.sleep(GEMINI_GAP_SECONDS)  # stay under free-tier per-minute limits
            status.empty()
            progress.empty()
            for msg in failures:
                st.error(f"Could not read {msg}")

    # Step 3: show results
    done = [e for e in entries if e[2] in st.session_state.results]
    if done:
        rows, verified_flags = [], []
        if len(done) == 1:
            f, _, file_id = done[0]
            row, ok = render_result(f.name, file_id, st.session_state.results[file_id])
            rows.append(row)
            verified_flags.append(ok)
        else:
            tabs = st.tabs([f"{i + 1}. {e[0].name[:18]}" for i, e in enumerate(done)])
            for tab, (f, _, file_id) in zip(tabs, done):
                with tab:
                    row, ok = render_result(f.name, file_id, st.session_state.results[file_id])
                    rows.append(row)
                    verified_flags.append(ok)

        st.divider()
        st.warning("⚠️ OCR can make mistakes. Always confirm medicine names and doses with the doctor or a pharmacist.")

        all_verified = all(verified_flags)
        df = pd.DataFrame(rows)
        st.subheader("📥 Download")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "✅ Download CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="extracted_text.csv",
                mime="text/csv",
                disabled=not all_verified,
            )
        with col2:
            try:
                st.download_button(
                    "📊 Download Excel",
                    data=to_excel_bytes(df),
                    file_name="extracted_text.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    disabled=not all_verified,
                )
            except Exception:
                st.caption("Excel export unavailable (openpyxl missing).")
        with col3:
            if st.button("❌ Start Over"):
                st.session_state.uploader_key += 1
                st.session_state.red = {}
                st.session_state.results = {}
                st.rerun()
        if not all_verified:
            st.caption("Tick every value in every note to enable the downloads.")
