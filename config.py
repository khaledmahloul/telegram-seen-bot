import os
from dotenv import load_dotenv

# 1. تحميل المتغيرات من ملف .env
load_dotenv()

# 2. التحقق من وجود التوكنات
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not TELEGRAM_TOKEN:
    raise SystemExit("⚠️ TELEGRAM_BOT_TOKEN غير معرّف. ضع TOKEN في ملف .env (TELEGRAM_BOT_TOKEN).")

if not GROQ_API_KEY:
    raise SystemExit("⚠️ GROQ_API_KEY غير معرّف. ضع GROQ_API_KEY في ملف .env.")

# 3. إعدادات Groq
GROQ_MODELS = {
    "fast": "llama-3.1-8b-instant",
    "balanced": "mixtral-8x7b-32768",
    "powerful": "llama-3.1-70b-versatile"
}

DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_TOKENS = 8192
TEMPERATURE = 1
# عدد الرسائل التاريخية المحفوظة (قابل للتعديل عبر ENV)
HISTORY_LENGTH = int(os.getenv("HISTORY_LENGTH", "6"))

# اسم النموذج الاحتياطي (يمكن تغييره عبر .env)
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "mixtral-8x7b-32768")

# إعدادات المشرفين (ADMIN_ID أو قائمة أسماء المستخدمين مفصولة بفواصل)
# ادعم ADMIN_ID مفرد أو ADMIN_IDS (قائمة من المعرفات مفصولة بفواصل).
_admin_ids_raw = os.getenv("ADMIN_IDS") or os.getenv("ADMIN_ID")
ADMIN_IDS = []
if _admin_ids_raw:
    try:
        ADMIN_IDS = [int(x.strip()) for x in _admin_ids_raw.split(',') if x.strip()]
    except ValueError:
        raise SystemExit("⚠️ ADMIN_IDS/ADMIN_ID يجب أن تحتوي على أرقام صحيحة مفصولة بفواصل.")
# تبقي قيمة ADMIN_ID للتماشي مع الكود القديم
ADMIN_ID = ADMIN_IDS[0] if ADMIN_IDS else None
_admin_usernames_raw = os.getenv("ADMIN_USERNAMES", "")
ADMIN_USERNAMES = [u.strip().lstrip('@') for u in _admin_usernames_raw.split(',') if u.strip()]

# ========== ConfigService (Singleton-like) ==========
class ConfigService:
    """خدمة لإدارة إعدادات وقاعدة المعرفة بشكل مركزي."""
    def __init__(self, kb_dir: str = "knowledge_base", prompts_dir: str = "prompts"):
        self.kb_dir = kb_dir
        self.prompts_dir = prompts_dir
        self.system_prompt = ""
        # حالات تشغيلية قابلة للتغيير بواسطة الأدمن
        self.admin_mode = False         # عند True: يسمح فقط للأدمن بالتفاعل
        self.streaming_enabled = True   # يمكن تعطيل/تمكين البث من الأدمن
        self.reload()

    def _read_file(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""

    def load_file(self, filename: str) -> str:
        """إرجاع محتوى ملف ضمن knowledge_base أو prompts (بحسب الاسم)."""
        # افحص مجلد knowledge_base أولاً
        kb_path = os.path.join(self.kb_dir, filename)
        if os.path.exists(kb_path):
            return self._read_file(kb_path)
        # ثم مجلد prompts
        prompt_path = os.path.join(self.prompts_dir, filename)
        if os.path.exists(prompt_path):
            return self._read_file(prompt_path)
        return ""

    def load_all_files(self) -> dict:
        """إرجاع dict لجميع ملفات knowledge_base: {filename: content}"""
        result = {}
        if not os.path.isdir(self.kb_dir):
            return result
        for name in os.listdir(self.kb_dir):
            path = os.path.join(self.kb_dir, name)
            if os.path.isfile(path):
                result[name] = self._read_file(path)
        return result

    def reload(self):
        """إعادة تحميل أي محتوى متغير (مثلاً system_prompt)"""
        self.system_prompt = self.load_file("system_prompt.txt") or self.system_prompt

    def get_system_prompt(self) -> str:
        return self.system_prompt or ""

    def get_stats(self) -> dict:
        files = self.load_all_files()
        return {"files_count": len(files), "prompt_length": len(self.get_system_prompt())}

    # --- واجهات إدارية لحالة البوت ---
    def set_admin_mode(self, enabled: bool):
        self.admin_mode = bool(enabled)

    def is_admin_mode(self) -> bool:
        return bool(self.admin_mode)

    def set_streaming_enabled(self, enabled: bool):
        self.streaming_enabled = bool(enabled)

    def is_streaming_enabled(self) -> bool:
        return bool(self.streaming_enabled)

# إنشاء مثيل مُستخدم في باقي المشروع
config_service = ConfigService()

