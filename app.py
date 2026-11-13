import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI
from urllib.parse import urlparse, parse_qs

app = FastAPI(title="Kholasegar")
app.mount("/static", StaticFiles(directory="static"), name="static")

class SummaryRequest(BaseModel):
    url: HttpUrl
    detail: str = "balanced"  # short | balanced | detailed

def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower().replace("www.", "")
    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
    elif host in ("youtube.com", "m.youtube.com", "music.youtube.com"):
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        elif parsed.path.startswith(("/embed/", "/shorts/")):
            video_id = parsed.path.split("/")[2]
        else:
            video_id = ""
    else:
        video_id = ""
    if not video_id or len(video_id) != 11:
        raise ValueError("لینک معتبر یوتیوب نیست.")
    return video_id

def get_transcript(video_id: str):
    api = YouTubeTranscriptApi()
    try:
        transcript = api.fetch(video_id, languages=["fa", "fa-IR"])
        language = "فارسی"
    except Exception:
        try:
            transcript_list = api.list(video_id)
            available = list(transcript_list)
            chosen = next((t for t in available if t.is_translatable), None)
            if not chosen:
                raise RuntimeError("هیچ زیرنویس قابل استفاده‌ای پیدا نشد.")
            transcript = chosen.translate("fa").fetch()
            language = "ترجمه‌شده به فارسی"
        except Exception as exc:
            raise RuntimeError(
                "زیرنویس فارسی یا زیرنویس قابل‌ترجمه برای این ویدیو پیدا نشد. "
                "ممکن است ویدیو زیرنویس نداشته باشد یا دسترسی یوتیوب محدود شده باشد."
            ) from exc
    parts = []
    for item in transcript:
        text = item.text.strip()
        if text:
            parts.append(text)
    return " ".join(parts), language

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/api/summarize")
def summarize(req: SummaryRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(500, "کلید OPENAI_API_KEY در فایل .env تنظیم نشده است.")
    try:
        video_id = extract_video_id(str(req.url))
        transcript, language = get_transcript(video_id)
        if len(transcript) < 80:
            raise RuntimeError("متن زیرنویس این ویدیو خیلی کوتاه است.")
        if len(transcript) > 120000:
            transcript = transcript[:120000] + " ..."

        detail_instructions = {
            "short": "خلاصه‌ای کوتاه در ۵ تا ۷ نکته بنویس.",
            "balanced": "خلاصه‌ای حرفه‌ای و متعادل با تیترها، نکات کلیدی و جمع‌بندی بنویس.",
            "detailed": "خلاصه‌ای مفصل و ساختاریافته بنویس؛ مفاهیم، مثال‌ها، استدلال‌ها و نتیجه‌گیری را پوشش بده."
        }
        instruction = detail_instructions.get(req.detail, detail_instructions["balanced"])
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.3,
            messages=[
                {"role": "system", "content":
                 "تو یک ویراستار و تحلیل‌گر حرفه‌ای محتوای ویدیویی هستی. "
                 "خروجی را فقط به زبان فارسی روان، دقیق و راست‌به‌چپ بنویس. "
                 "چیزی را که در متن نیست جعل نکن. اگر بخشی مبهم است، صریح بگو. "
                 "ساختار پیشنهادی: عنوان، موضوع اصلی، خلاصه، نکات کلیدی، "
                 "مثال‌ها یا آمار مهم، نتیجه‌گیری و ۳ نکته قابل‌اجرا. "
                 "زمان‌بندی نساز مگر اینکه زمان در متن موجود باشد."},
                {"role": "user", "content":
                 f"سطح جزئیات: {instruction}\nزبان زیرنویس: {language}\n\n"
                 f"متن ویدیو:\n{transcript}"}
            ]
        )
        result = response.choices[0].message.content or "خلاصه‌ای تولید نشد."
        return {"summary": result, "language": language, "video_id": video_id}
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"خطا در پردازش ویدیو: {str(exc)}")

