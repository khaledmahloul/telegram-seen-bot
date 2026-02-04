#!/usr/bin/env python3
"""
بوت متجر إلكتروني مع ذكاء اصطناعي
يدعم اللغة العربية ويعتمد على ملفات معرفة ثابتة
"""

import logging
import os
import asyncio
import time
from typing import Optional
from collections import defaultdict
import json
import tempfile
from datetime import datetime

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)
from telegram.error import BadRequest

from groq import Groq
from config import (
    TELEGRAM_TOKEN,
    DEFAULT_MODEL,
    FALLBACK_MODEL,     # ← إضافة الاستيراد
    GROQ_API_KEY,
    MAX_TOKENS,
    TEMPERATURE,
    config_service,
    ADMIN_ID,
    ADMIN_IDS,
    ADMIN_USERNAMES,
    HISTORY_LENGTH
)

# ============= إعدادات التسجيل =============
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
TELEGRAM_MESSAGE_MAX = 4096  # Telegram message maximum length

# ============= تهيئة Groq =============
try:
    client = Groq(api_key=GROQ_API_KEY)
    logger.info("✅ تم توصيل Groq بنجاح!")
except Exception as e:
    logger.exception("❌ خطأ في تهيئة Groq", exc_info=e)
    logger.error("🔧 تأكد من:\n1. صحة GROQ_API_KEY في ملف .env\n2. اتصال الإنترنت")
    raise SystemExit(1)

# ============= متغيرات البوت =============
# تخزين محادثات كل مستخدم
user_conversations = defaultdict(list)

# helper: استخراج نص آمن من chunk/response
def _extract_text_from_chunk(chunk) -> str:
    # ...الصق دالة _extract_text_from_chunk كما أعطيتها لك...
    try:
        choice = chunk.choices[0]
    except Exception:
        return ""
    delta = getattr(choice, "delta", None)
    if isinstance(delta, dict):
        return delta.get("content", "") or ""
    if hasattr(delta, "content"):
        try:
            return getattr(delta, "content") or ""
        except Exception:
            pass
    msg = getattr(choice, "message", None)
    if msg:
        content = getattr(msg, "content", None)
        if content:
            return content
        if isinstance(msg, dict):
            return msg.get("content", "") or ""
    for attr in ("text", "output_text", "content"):
        val = getattr(choice, attr, None)
        if val:
            return val
    try:
        return (choice.get("text") or choice.get("content") or "")
    except Exception:
        return ""

def _extract_text_from_response(response) -> str:
    # ...الصق دالة _extract_text_from_response كما أعطيتها لك...
    try:
        choice = response.choices[0]
    except Exception:
        try:
            return response["choices"][0]["message"]["content"]
        except Exception:
            return ""
    msg = getattr(choice, "message", None)
    if msg:
        content = getattr(msg, "content", None)
        if content:
            return content
        if isinstance(msg, dict):
            return msg.get("content", "") or ""
    for attr in ("text", "content", "output_text"):
        val = getattr(choice, attr, None)
        if val:
            return val
    try:
        return choice.get("text") or choice.get("content") or ""
    except Exception:
        return ""

# --- مساعدة: تقسيم النص الطويل إلى أجزاء مقبولة من تلغرام ---

def split_text(text: str, max_len: int = 4000) -> list:
    """قسّم النص إلى قطع بطول أقصى max_len (يحاول التجزئة عند سطور/مسافات)."""
    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        idx = text.rfind('\n', 0, max_len)
        if idx == -1:
            idx = text.rfind(' ', 0, max_len)
        if idx == -1:
            idx = max_len
        chunks.append(text[:idx].rstrip())
        text = text[idx:].lstrip()
    return chunks


async def send_long_text(bot, chat_id: int, text: str, parse_mode='Markdown'):
    for chunk in split_text(text, max_len=TELEGRAM_MESSAGE_MAX - 20):
        await bot.send_message(chat_id=chat_id, text=chunk, parse_mode=parse_mode)


async def safe_edit_final_message(context, sent, text: str):
    """حاول تحرير الرسالة النهائية، وإذا كانت كبيرة، اقطعها وأرسل الباقي كرسائل جديدة."""
    chunks = split_text(text, max_len=TELEGRAM_MESSAGE_MAX - 20)
    try:
        await context.bot.edit_message_text(chat_id=sent.chat_id, message_id=sent.message_id, text=chunks[0], parse_mode='Markdown')
    except BadRequest as e:
        msg = str(e).lower()
        if "message is not modified" in msg:
            logger.debug("⚠️ تجاهل BadRequest: الرسالة لم تتغير")
        else:
            await context.bot.send_message(chat_id=sent.chat_id, text=chunks[0], parse_mode='Markdown')
    except Exception:
        await context.bot.send_message(chat_id=sent.chat_id, text=chunks[0], parse_mode='Markdown')

    for chunk in chunks[1:]:
        await context.bot.send_message(chat_id=sent.chat_id, text=chunk, parse_mode='Markdown')

# ============= الدوال الأساسية =============
def is_admin_user(user) -> bool:
    """التحقق من أن المستخدم مسؤول"""
    if not user:
        return False
    uid = getattr(user, "id", None)
    if uid is not None:
        try:
            if ADMIN_IDS and uid in ADMIN_IDS:
                return True
        except Exception:
            pass
        if ADMIN_ID is not None and uid == ADMIN_ID:
            return True
    username = getattr(user, "username", "")
    if username and username.lstrip('@') in ADMIN_USERNAMES:
        return True
    return False

async def get_ai_response(user_message: str, user_id: int, on_chunk=None) -> Optional[str]:
    """
    يجرب النموذج الأساسي ثم النموذج الاحتياطي تلقائياً عند الفشل.
    يدعم البث (stream=True) ثم fallback لطلب غير متدفق لكل نموذج إن فشل البث.
    """
    try:
        history = user_conversations.get(user_id, [])
        base_messages = [{"role": "system", "content": config_service.get_system_prompt()}] + history[-4:] + [{"role": "user", "content": user_message}]

        models_to_try = [DEFAULT_MODEL]
        if FALLBACK_MODEL and FALLBACK_MODEL != DEFAULT_MODEL:
            models_to_try.append(FALLBACK_MODEL)

        last_exception = None

        for model in models_to_try:
            logger.info(f"🔁 محاولة استخدام النموذج: {model} للمستخدم {user_id}")

            # اختر أسلوب الطلب بناءً على إعداد البث
            stream_enabled = config_service.is_streaming_enabled()

            if stream_enabled:
                try:
                    stream_iter = client.chat.completions.create(
                        messages=base_messages,
                        model=model,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                        top_p=1,
                        reasoning_effort="medium",
                        stream=True
                    )

                    full_reply = ""
                    if hasattr(stream_iter, "__aiter__"):
                        async for chunk in stream_iter:
                            part = _extract_text_from_chunk(chunk)
                            if part:
                                full_reply += part
                                if on_chunk:
                                    try:
                                        await on_chunk(part)
                                    except Exception:
                                        logger.debug("⚠️ on_chunk raised", exc_info=True)
                    else:
                        for chunk in stream_iter:
                            part = _extract_text_from_chunk(chunk)
                            if part:
                                full_reply += part
                                if on_chunk:
                                    try:
                                        if asyncio.iscoroutinefunction(on_chunk):
                                            await on_chunk(part)
                                        else:
                                            on_chunk(part)
                                    except Exception:
                                        logger.debug("⚠️ on_chunk raised", exc_info=True)

                    # نجاح: حفظ المحادثة وإرجاع الرد
                    logger.info(f"✅ تم الحصول على رد بنجاح من النموذج: {model}")
                    history.append({"role": "user", "content": user_message})
                    history.append({"role": "assistant", "content": full_reply})
                    user_conversations[user_id] = history[-HISTORY_LENGTH:]
                    return full_reply

                except Exception as e_stream:
                    logger.warning(f"⚠️ Streaming failed for model {model}: {e_stream}", exc_info=True)
                    last_exception = e_stream
                    # جرب طلب غير متدفق لنفس النموذج بعد فشل البث

            # إما أن البث معطّل أو فشل؛ جرب non-stream
            try:
                response = client.chat.completions.create(
                    messages=base_messages,
                    model=model,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    top_p=1,
                    reasoning_effort="medium",
                    stream=False,
                    stop=None
                )
                ai_reply = _extract_text_from_response(response)
                if ai_reply:
                    logger.info(f"✅ تم الحصول على رد (non-stream) من النموذج: {model}")
                    history.append({"role": "user", "content": user_message})
                    history.append({"role": "assistant", "content": ai_reply})
                    user_conversations[user_id] = history[-HISTORY_LENGTH:]
                    return ai_reply
                else:
                    logger.error(f"❌ رد non-stream من النموذج {model} بلا نص. response repr محفوظ.")
                    logger.debug(repr(response))
            except Exception as e_non_stream:
                logger.exception(f"🔥 Non-stream failed for model {model}", exc_info=e_non_stream)
                last_exception = e_non_stream

            # إن فشل هذا النموذج، تابع للنموذج التالي في القائمة

        # إذا وصلت هنا، فكل النماذج فشلت
        logger.error("🔥 كل النماذج فشلت في توليد رد.", exc_info=last_exception)
        return None

    except Exception as e:
        logger.exception("🔥 Unexpected error in get_ai_response", exc_info=e)
        return None

# ============= أوامر البوت =============
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """أمر /start - ترحيب بالمستخدم"""
    user = update.effective_user
    
    welcome_message = f"""
    🎉 **مرحباً {user.first_name}!** 🎉

أهلاً بك في **تِك ستور** 🤖
المتجر الإلكتروني للأجهزة الذكية

✨ **كيف يمكنني مساعدتك؟**

📱 **اسألني عن:**
- أسعار المنتجات (هواتف، تابلت)
- المواصفات الفنية
- العروض والخصومات
- مدة التوصيل والشحن
- وسائل الدفع المتاحة
- سياسة الإرجاع والضمان

💬 **مثال:** "كم سعر آيفون 15؟" أو "هل التوصيل مجاني؟"

🔧 **الأوامر المتاحة:**
/start - عرض هذه الرسالة
/help - المساعدة والأوامر
/products - رؤية المنتجات
/faq - الأسئلة الشائعة
/clear - مسح محادثتنا السابقة

اكتب رسالتك وسأرد عليك فوراً! ⚡
    """
    
    await update.message.reply_text(welcome_message, parse_mode='Markdown')
    logger.info(f"👤 المستخدم {user.id} بدأ المحادثة")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """أمر /help - عرض المساعدة"""
    help_text = """
    🆘 **دليل الاستخدام**

**كيفية الاستخدام:**
1. اكتب سؤالك مباشرة وسأرد عليك
2. استخدم الأوامر للوصول السريع للمعلومات

**قائمة الأوامر:**
• `/products` - عرض قائمة المنتجات
• `/faq` - الأسئلة الشائعة
• `/policies` - سياسات المتجر
• `/clear` - مسح ذاكرة المحادثة
• `/help` - هذه الرسالة
"""

    # إضافة قسم الأوامر الإدارية فقط إذا كان البوت في وضع الصيانة
    if config_service.is_admin_mode():
        help_text += """

**أوامر إدارية (للمشرفين فقط):**
• `/set_admin_mode on|off` - تفعيل/تعطيل وضع الصيانة
• `/set_streaming on|off` - تفعيل/تعطيل البث
• `/admin_status` - عرض حالات الأدمن والإعدادات
"""

    help_text += """

**نصائح:**
- استخدم `/clear` إذا أردت بدء محادثة جديدة
- البوت يجيب بناءً على معلومات المتجر فقط
- إذا لم يعرف الإجابة، سيخبرك بذلك بصراحة
- الأسعار والعروض محدثة دائماً

**مثال على الأسئلة:**
- "ما سعر سامسونج S24؟"
- "هل هناك خصم اليوم؟"
- "كم مدة التوصيل لجدّة؟"
- "كيف يمكنني الإرجاع؟"
    """

    await update.message.reply_text(help_text, parse_mode='Markdown')

async def products_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """أمر /products - عرض المنتجات"""
    products = config_service.load_file("products.txt")
    
    if not products or len(products) < 50:
        await update.message.reply_text(
            "📭 لم يتم إضافة المنتجات بعد.\n"
            "الرجاء إضافة ملف products.txt في مجلد knowledge_base"
        )
        return
    
    # إرسال نظرة عامة (مقسمة آلياً إذا كانت طويلة)
    preview = products.strip()
    if len(preview) > 3500:
        preview = preview[:3500] + "\n\n... (المزيد من المنتجات في الملف)"

    await send_long_text(context.bot, update.effective_chat.id, f"📱 **منتجاتنا:**\n\n{preview}")

async def faq_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """أمر /faq - الأسئلة الشائعة"""
    # استخدم config_service بدلاً من knowledge_base مباشرة
    faq = config_service.load_file("faq.txt")

    if not faq or len(faq) < 50:
        await update.message.reply_text(
            "❓ لم يتم إضافة الأسئلة الشائعة بعد.\n"
            "الرجاء إضافة ملف faq.txt في مجلد knowledge_base"
        )
        return

    preview = faq.strip()
    if len(preview) > 3500:
        preview = preview[:3500] + "\n\n... (المزيد من الأسئلة في الملف)"

    await send_long_text(context.bot, update.effective_chat.id, f"❓ **الأسئلة الشائعة:**\n\n{preview}")

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """أمر /clear - مسح محادثة المستخدم"""
    user_id = update.effective_user.id
    
    if user_id in user_conversations:
        user_conversations[user_id] = []
        await update.message.reply_text(
            "✅ تم مسح محادثتنا السابقة!\n"
            "يمكنك الآن بدء محادثة جديدة 🆕"
        )
        logger.info(f"🧹 المستخدم {user_id} مسح المحادثة")
    else:
        await update.message.reply_text("💭 لا توجد محادثة سابقة لمسحها.")

async def reload_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """أمر /reload - إعادة تحميل قاعدة المعرفة"""
    await update.message.reply_text("🔄 جاري تحديث معلومات المتجر...")
    
    # إعادة تحميل قاعدة المعرفة
    config_service.reload()
    stats = config_service.get_stats()
    await update.message.reply_text(
        f"✅ تم التحديث بنجاح!\n• الملفات: {stats['files_count']}\n• طول الـ Prompt: {stats['prompt_length']} حرف"
    )


async def set_admin_mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/set_admin_mode on|off - تفعيل/تعطيل وضع الصيانة (الأدمن فقط)"""
    user = update.effective_user
    if not is_admin_user(user):
        await update.message.reply_text("🔒 غير مسموح. هذا الأمر للأدمن فقط.")
        logger.warning(f"محاولة غير مصرح بها لتغيير وضع الأدمن من {user.id}")
        return
    if not context.args:
        await update.message.reply_text("الاستخدام: /set_admin_mode on|off")
        return
    arg = context.args[0].lower()
    if arg in ("on", "1", "true", "enable", "enabled"):
        config_service.set_admin_mode(True)
        await update.message.reply_text("🔧 تم تفعيل وضع الصيانة. فقط الأدمن يمكنه الآن التفاعل.")
        logger.info(f"Admin mode enabled by {user.id}")
    elif arg in ("off", "0", "false", "disable", "disabled"):
        config_service.set_admin_mode(False)
        await update.message.reply_text("✅ تم تعطيل وضع الصيانة. البوت متاح للجميع.")
        logger.info(f"Admin mode disabled by {user.id}")
    else:
        await update.message.reply_text("القيمة غير صحيحة. استخدم: /set_admin_mode on|off")


async def set_streaming_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/set_streaming on|off - تمكين/تعطيل البث (streaming) (الأدمن فقط)"""
    user = update.effective_user
    if not is_admin_user(user):
        await update.message.reply_text("🔒 غير مسموح. هذا الأمر للأدمن فقط.")
        logger.warning(f"محاولة غير مصرح بها لتغيير وضع البث من {user.id}")
        return
    if not context.args:
        await update.message.reply_text("الاستخدام: /set_streaming on|off")
        return
    arg = context.args[0].lower()
    if arg in ("on", "1", "true", "enable", "enabled"):
        config_service.set_streaming_enabled(True)
        await update.message.reply_text("🔁 تم تفعيل البث (streaming).")
        logger.info(f"Streaming enabled by {user.id}")
    elif arg in ("off", "0", "false", "disable", "disabled"):
        config_service.set_streaming_enabled(False)
        await update.message.reply_text("⛔ تم تعطيل البث. سيتم استخدام الطلبات غير المتدفقة.")
        logger.info(f"Streaming disabled by {user.id}")
    else:
        await update.message.reply_text("القيمة غير صحيحة. استخدم: /set_streaming on|off")


async def admin_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/admin_status - اعرض حالة الإعدادات الإدارية (الأدمن فقط)"""
    user = update.effective_user
    if not is_admin_user(user):
        await update.message.reply_text("🔒 غير مسموح. هذا الأمر للأدمن فقط.")
        return
    admins = ', '.join([str(x) for x in (ADMIN_IDS or [])]) or str(ADMIN_ID or "لم يتم التعيين")
    usernames = ', '.join(ADMIN_USERNAMES) or "لم يتم التعيين"
    await update.message.reply_text(
        f"🔐 Admins: {admins}\n👥 Usernames: {usernames}\n🔧 Admin mode: {config_service.is_admin_mode()}\n🔁 Streaming: {config_service.is_streaming_enabled()}\nHISTORY_LENGTH: {HISTORY_LENGTH}"
    )


async def list_conversations_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/list_conversations - عرض قائمة المعرفات (user_id) وعدد رسائل كل محادثة (الأدمن فقط)"""
    user = update.effective_user
    if not is_admin_user(user):
        await update.message.reply_text("🔒 غير مسموح. هذا الأمر للأدمن فقط.")
        return
    if not user_conversations:
        await update.message.reply_text("لا توجد محادثات محفوظة حالياً.")
        return
    lines = []
    for uid, conv in user_conversations.items():
        lines.append(f"• {uid}: {len(conv)} رسالة")
    text = "قائمة المحادثات الحالية:\n\n" + "\n".join(lines)
    # قد تكون طويلة؛ استخدم send_long_text
    await send_long_text(context.bot, update.effective_chat.id, text)


async def export_conversations_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/export_conversations [user_id|all] [limit]
    يُصدر سجل المحادثات كملف JSON. limit اختياري (آخر N رسائل لكل محادثة).
    مثال: /export_conversations 7345972348 100
             /export_conversations all
    """
    user = update.effective_user
    if not is_admin_user(user):
        await update.message.reply_text("🔒 غير مسموح. هذا الأمر للأدمن فقط.")
        return

    args = context.args or []
    target = 'all'
    limit = None
    if args:
        target = args[0].lower()
        if len(args) > 1:
            try:
                limit = int(args[1])
            except Exception:
                await update.message.reply_text("القيمة الثانية يجب أن تكون عدداً صحيحاً يمثل الحد الأقصى لعدد الرسائل لكل محادثة.")
                return

    data = {
        'exported_at': datetime.utcnow().isoformat() + 'Z',
        'exported_by': getattr(user, 'id', 'unknown'),
        'conversations': {}
    }

    def _limited(conv_list):
        if limit is None:
            return conv_list
        return conv_list[-limit:]

    if target in ('all', '*'):
        for uid, conv in user_conversations.items():
            data['conversations'][str(uid)] = _limited(conv)
        if not data['conversations']:
            await update.message.reply_text('لا توجد محادثات للتصدير.')
            return
    else:
        # expect numeric user id
        try:
            uid = int(target)
        except Exception:
            await update.message.reply_text('يجب تمرير user_id رقمي أو كلمة all.')
            return
        conv = user_conversations.get(uid)
        if not conv:
            await update.message.reply_text(f'لا توجد محادثة محفوظة للمستخدم {uid}.')
            return
        data['conversations'][str(uid)] = _limited(conv)

    # اكتب الملف مؤقتاً وأرسله
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    filename = f"conversations_{target}_{ts}.json"
    try:
        with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8', suffix='.json') as tf:
            json.dump(data, tf, ensure_ascii=False, indent=2)
            tmpname = tf.name
        # أرسل الملف
        with open(tmpname, 'rb') as fh:
            await context.bot.send_document(chat_id=update.effective_chat.id, document=fh, filename=filename)
        # أطبع سجلًا
        logger.info(f"Exported conversations ({target}) by {user.id}")
    except Exception:
        logger.exception('فشل تصدير المحادثات', exc_info=True)
        await update.message.reply_text('❌ حدث خطأ أثناء إعداد ملف التصدير.')
    finally:
        try:
            import os
            if 'tmpname' in locals() and os.path.exists(tmpname):
                os.remove(tmpname)
        except Exception:
            pass


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالجة الرسائل العادية من المستخدمين مع دعم البث (streaming) مع مؤشر 'typing' دوري"""
    user_id = update.effective_user.id
    user_message = update.message.text

    if user_message.startswith('/'):
        return

    logger.info(f"📩 من {user_id}: {user_message}")

    if config_service.is_admin_mode() and not is_admin_user(update.effective_user):
        await update.message.reply_text("البوت في وضع الصيانة — التفاعل مقصور على الأدمن")
        return
    # إرسال رسالة مبدئية تُعدّل أثناء البث
    sent = await update.message.reply_text("⏳ جاري التفكير...")

    current_text = ""
    last_edit = time.time()

    # buffer خاص بالتجزئة وtask لإفراغه بعد هدوء التدفق
    buffer_since_last_edit = ""
    flush_task = None
    FLUSH_DELAY = 0.6  # وقت الانتظار قبل الإفراغ بعد توقف التدفق (بالثواني)
    MIN_CHARS_TO_FORCE = 6  # عدد أحرف لتسرّع التحديث إذا تراكمت

    async def _edit_sent(text: str):
        nonlocal last_edit
        try:
            # اقتطاع النص الطويل للاحتفاظ بآخر ما قيل وعدم تجاوز حد تلغرام
            if len(text) > TELEGRAM_MESSAGE_MAX - 20:
                text = text[-(TELEGRAM_MESSAGE_MAX - 20):]
            await context.bot.edit_message_text(
                chat_id=sent.chat_id,
                message_id=sent.message_id,
                text=text,
                parse_mode='Markdown'
            )
            last_edit = time.time()
        except BadRequest as e:
            msg = str(e).lower()
            if "message is not modified" in msg:
                logger.debug("⚠️ تجاهل BadRequest: الرسالة لم تتغير")
            else:
                logger.debug("⚠️ BadRequest during edit", exc_info=True)
        except Exception:
            logger.exception("⚠️ Failed to edit message", exc_info=True)

    async def _flush_after_delay():
        nonlocal buffer_since_last_edit, flush_task
        try:
            await asyncio.sleep(FLUSH_DELAY)
            if buffer_since_last_edit:
                await _edit_sent(current_text)
                buffer_since_last_edit = ""
        except asyncio.CancelledError:
            return
        finally:
            flush_task = None

    # مهمة دورية لإظهار typing كل 2.5 ثانية تقريباً
    stop_typing = asyncio.Event()

    async def _keep_typing():
        try:
            while not stop_typing.is_set():
                try:
                    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
                except Exception:
                    logger.debug("⚠️ send_chat_action failed", exc_info=True)
                try:
                    await asyncio.wait_for(stop_typing.wait(), timeout=2.5)
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            return

    typing_task = asyncio.create_task(_keep_typing())

    async def on_chunk(part: str):
        nonlocal current_text, last_edit, sent, buffer_since_last_edit, flush_task
        current_text += part
        buffer_since_last_edit += part

        # إذا تراكم نص كافٍ أو مرّ وقت طويل منذ آخر تعديل، حدث فوراً
        now = time.time()
        if len(buffer_since_last_edit) >= MIN_CHARS_TO_FORCE or (now - last_edit) > 1.2:
            if flush_task and not flush_task.done():
                flush_task.cancel()
                flush_task = None
            await _edit_sent(current_text)
            buffer_since_last_edit = ""
            return

        # جدولة إفراغ بعد هدوء (تُلغى عند وصول chunk جديد)
        if flush_task and not flush_task.done():
            flush_task.cancel()
        flush_task = asyncio.create_task(_flush_after_delay())

    # استدعاء الـ AI مع callback البث
    ai_reply = None
    try:
        ai_reply = await get_ai_response(user_message, user_id, on_chunk=on_chunk)
    finally:
        # أوقف أي flush مجدول فوراً وافراغ الباقي قبل الإيقاف
        if flush_task and not flush_task.done():
            flush_task.cancel()
            flush_task = None
        # إذا بقي شيء غير مفرغ، حرّره الآن
        if buffer_since_last_edit:
            try:
                await _edit_sent(current_text)
            except Exception:
                pass
            buffer_since_last_edit = ""

        # أوقف مهمة typing بأمان بعد انتهاء/فشل الاستجابة
        stop_typing.set()
        try:
            await asyncio.wait_for(typing_task, timeout=3.0)
        except (asyncio.TimeoutError, Exception):
            typing_task.cancel()
            try:
                await typing_task
            except Exception:
                pass

    # إنهاء/عرض النتيجة النهائية
    if ai_reply:
        try:
            await safe_edit_final_message(context, sent, ai_reply)
        except Exception:
            # كحل احتياطي، أرسل كرد مستقل
            try:
                await update.message.reply_text(ai_reply, parse_mode='Markdown')
            except Exception:
                logger.exception("⚠️ Failed to send final AI reply", exc_info=True)
        logger.info(f"📤 إلى {user_id}: {ai_reply[:50]}...")
    else:
        try:
            await context.bot.edit_message_text(
                chat_id=sent.chat_id,
                message_id=sent.message_id,
                text="⚠️ عذراً، حدث خطأ في المعالجة.\nالرجاء المحاولة مرة أخرى بعد قليل."
            )
        except Exception:
            await update.message.reply_text(
                "⚠️ عذراً، حدث خطأ في المعالجة.\nالرجاء المحاولة مرة أخرى بعد قليل."
            )
            
            
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالجة الأخطاء العامة - يسجل traceback ويبلغ المستخدم بلطف"""
    logger.exception("🔥 Unhandled exception in update handling", exc_info=context.error)
    if update and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "❌ حدث خطأ غير متوقع.\n"
                "تم تسجيل الخطأ في السجلات. إذا استمر، أبلغ المسؤول."
            )
        except Exception:
            logger.debug("⚠️ Failed to send error message to user", exc_info=True)

# ============= الدالة الرئيسية =============
def main():
    """بدء تشغيل البوت"""
    print("=" * 50)
    print("🤖 بدء تشغيل بوت المتجر الذكي")
    print("=" * 50)
    
    # 1. التحقق من الملفات المهمة
    required_files = [
        "knowledge_base/products.txt",
        "knowledge_base/faq.txt",
        "prompts/system_prompt.txt"
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"⚠️  تحذير: الملف {file} غير موجود")
            print(f"   سأقوم بإنشاء ملف فارغ...")
            os.makedirs(os.path.dirname(file), exist_ok=True)
            with open(file, 'w', encoding='utf-8') as f:
                basename = os.path.basename(file)
                if basename == "products.txt":
                    f.write("# products.txt\n# مثال:\n# اسم المنتج | الوصف | السعر\n# iPhone 15 | هاتف ذكي 128GB | 3999\n")
                elif basename == "faq.txt":
                    f.write("# faq.txt\n# مثال:\n# السؤال: كيف يمكنني الإرجاع؟\n# الإجابة: يرجى التواصل خلال 14 يومًا عبر دعم العملاء.\n")
                elif basename == "system_prompt.txt":
                    f.write("# system_prompt.txt\n# أمثلة على System prompt لتحسين إجابات الذكاء الاصطناعي\n# أنت مساعد دعم للمتجر، أجب باختصار وباللغة العربية، واذكر فقط المعلومات المتوفرة في قاعدة المعرفة.\n")
                else:
                    f.write(f"# هذا الملف: {basename}\n")
    
    # 2. إنشاء تطبيق التلجرام
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.bot_data['config'] = config_service
    
    # 3. إضافة الأوامر
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("products", products_command))
    application.add_handler(CommandHandler("faq", faq_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(CommandHandler("reload", reload_command))
    # أوامر إدارية (محجوزة للأدمن)
    application.add_handler(CommandHandler("set_admin_mode", set_admin_mode_command))
    application.add_handler(CommandHandler("set_streaming", set_streaming_command))
    application.add_handler(CommandHandler("admin_status", admin_status_command))
    application.add_handler(CommandHandler("list_conversations", list_conversations_command))
    application.add_handler(CommandHandler("export_conversations", export_conversations_command))
    
    # 4. إضافة معالج الرسائل العادية
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        handle_message
    ))
    
    # 5. إضافة معالج الأخطاء
    application.add_error_handler(error_handler)
    
    # 6. بدء البوت
    print("✅ البوت جاهز للعمل!")
    # استخدم config_service للحصول على الإحصاءات والنصوص
    kb_files = config_service.load_all_files()
    system_prompt = config_service.get_system_prompt()
    print(f"📁 قاعدة المعرفة: {len(kb_files)} ملف")
    print(f"🧠 System Prompt: {len(system_prompt)} حرف")
    print("=" * 50)
    print("🚀 البوت يعمل الآن...")
    print("💡 اذهب إلى Telegram وابحث عن بوتك")
    print("💬 ابدأ المحادثة بـ /start")
    print("=" * 50)
    
    application.run_polling()

if __name__ == "__main__":
    main()