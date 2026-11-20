import os
import time
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any

import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sklearn.ensemble import IsolationForest

# ==============================================================================
# 1. КОНФИГУРАЦИЯ И ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ
# ==============================================================================
app = FastAPI(
    title="BCC Qorgau Anti-Fraud Engine",
    description="Predictive Behavioral & Graph Anti-Fraud SDK + Live SOC 505/605",
    version="2.4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Токен Telegram-бота берется из Environment Variables (Render / Railway) или из строки ниже
BOT_TOKEN: str = os.getenv("BOT_TOKEN", "ВАШ_ТОКЕН_БОТА")
DEFAULT_CHAT_ID: str = os.getenv("CHAT_ID", "")

# Реестр всех пользователей, которые нажали /start в Telegram-боте
subscribed_chats: set[str] = set()
if DEFAULT_CHAT_ID:
    subscribed_chats.add(str(DEFAULT_CHAT_ID))

last_active_chat: str = str(DEFAULT_CHAT_ID) if DEFAULT_CHAT_ID else ""

# ==============================================================================
# 2. ОБУЧЕНИЕ ML-МОДЕЛИ (ISOLATION FOREST) НА ПСИХОМОТОРИКЕ ВВОДА
# ==============================================================================
rng = np.random.RandomState(42)

# Признаки нормального ввода:
# [avg_key_delay_ms, max_pause_sec, backspace_count, total_input_time_sec]
X_normal_train = np.column_stack([
    rng.normal(215, 45, 300),    # средняя задержка между цифрами (мс)
    rng.normal(0.35, 0.15, 300), # максимальная пауза (сек)
    rng.poisson(0.25, 300),      # количество исправлений Backspace
    rng.normal(2.4, 0.7, 300)    # общее время ввода суммы (сек)
])

iso_forest = IsolationForest(
    n_estimators=150,
    contamination=0.05,
    random_state=42
)
iso_forest.fit(X_normal_train)

# ==============================================================================
# 3. СОСТОЯНИЕ СИСТЕМЫ, SOC-ЖУРНАЛ И СТОП-БАЗЫ (БЦК + НБРК)
# ==============================================================================
tx_counter: int = 100
soc_logs: List[Dict[str, Any]] = []

NBRK_BLACKLIST_PHONES = {
    "77006660000",
    "+77006660000",
    "+7 700 666-00-00",
}

latest_tx_state: Dict[str, Any] = {
    "id": 0,
    "status": "APPROVED",
    "action_title": "",
    "action_desc": "",
    "learned_baseline": False,
    "target_chat": "",
    "lang": "ru"
}

# ==============================================================================
# 4. ДВУЯЗЫЧНЫЙ СЛОВАРЬ ВЕРДИКТОВ И SHAP-ОБЪЯСНЕНИЙ (RU / KK)
# ==============================================================================
TEXTS = {
    "ru": {
        "low_friction": "✅ Low-Friction Filter: сумма до 50 000 ₸ без признаков давления — мгновенный перевод.",
        "nbrk_hit": "⛔ Совпадение с дроп-счетом из стоп-базы Антифрод-центра НБРК (+95)",
        "rat_hit": "🚨 Обнаружена трансляция/захват экрана (UIScreen.isCaptured / AnyDesk) (+60)",
        "path_hit": "⚠️ Аномальный маршрут в bcc.kz: досрочное закрытие депозита «Рахмет» → СМП (+35)",
        "call_hit": "📞 Активный голосовой вызов (CallKit: isOnCall=True) во время перевода (+30)",
        "loan_hit": "💳 Онлайн-кредит БЦК оформлен менее 15 минут назад (+25)",
        "mule_hit": "🔄 Признаки транзитного дропа / недавняя смена SIM-карты (IMSI) (+30)",
        "inv_hit": "🧾 Новый ИП (< 3 дней), несовпадение MCC и жалобы клиентов на QR-счет (+45)",
        "biz_hit": "🏢 BCC Business (605): новый контрагент + крупное платежное поручение > 1 млн ₸ (+40)",
        "paste_hit": "📋 Контекстный Copy-Paste суммы во время активного звонка / аномалии (+20)",
        "iso_hit": "⌨️ Isolation Forest: диктовка цифр (пауза {pause:.1f} сек, исправлений: {bs}) (+20)",
        "base_hit": "🧠 Personal Baseline: поведенческий паттерн совпал с самообученной нормой клиента (-35)",
        "norm_ok": "✅ Поведенческая биометрия, маршрут в bcc.kz и граф получателя в пределах нормы.",

        "blocked_title": "Операция заблокирована BCC Qorgau 🛑",
        "blocked_desc": "Высокий риск социальной инженерии. Запрос на подтверждение отправлен доверенному лицу в Telegram.",
        "nbrk_title": "Жёсткая блокировка по стоп-базе НБРК ⛔",
        "nbrk_desc": "Счет получателя находится в реестре дропперов Антифрод-центра НБРК. Перевод невозможен.",
        "cooldown_title": "Период охлаждения (15 мин) ⏳",
        "cooldown_desc": "Обнаружены признаки давления. Подтвердите перевод через доверенное лицо в Telegram или BCC FaceID.",
        "friction_title": "Дополнительная проверка безопасности ⚠️",
        "friction_desc": "Перевод новому получателю. Подтвердите, что действуете самостоятельно без звонка.",
        "approved_title": "Перевод выполнен (bcc.kz) ✅",
        "approved_desc": "Проверка BCC Qorgau пройдена. Средства отправлены по СМП.",

        "tg_approve_title": "Подтверждено доверенным лицом ✅",
        "tg_approve_desc": "Доверенное лицо подтвердило перевод в Telegram. Профиль добавлен в Personal Baseline.",
        "tg_block_title": "Заблокировано доверенным лицом 🛑",
        "tg_block_desc": "Доверенное лицо отклонило перевод! Инцидент передан в SOC БЦК (линия 505/605).",
        "faceid_title": "Подтверждено по BCC FaceID ✅",
        "faceid_desc": "Биометрическая видео-проверка с активным Liveness пройдена: признаков принуждения не обнаружено."
    },
    "kk": {
        "low_friction": "✅ Low-Friction сүзгісі: 50 000 ₸ дейінгі сома, қысым белгілері жоқ — лезде аударым.",
        "nbrk_hit": "⛔ ҚР Ұлттық Банкі Антифрод-орталығының дроп-базасымен сәйкестік (+95)",
        "rat_hit": "🚨 Экранды трансляциялау/жазу анықталды (UIScreen.isCaptured / AnyDesk) (+60)",
        "path_hit": "⚠️ bcc.kz ішіндегі ауытқулы бағыт: «Рахмет» депозитін мерзімінен бұрын жабу → СМП (+35)",
        "call_hit": "📞 Аударым кезінде белсенді дауыстық қоңырау (CallKit: isOnCall=True) (+30)",
        "loan_hit": "💳 БЦК онлайн-несиесі 15 минут бұрын ғана рәсімделген (+25)",
        "mule_hit": "🔄 Транзиттік дроп белгілері / SIM-картаны жақында ауыстыру (IMSI) (+30)",
        "inv_hit": "🧾 Жаңа ЖК (< 3 күн), MCC сәйкессіздігі және QR-шотқа шағымдар (+45)",
        "biz_hit": "🏢 BCC Business (605): жаңа контрагент + 1 млн ₸ асатын ірі төлем тапсырмасы (+40)",
        "paste_hit": "📋 Контекстік Copy-Paste: қоңырау кезінде соманы буферден қою (+20)",
        "iso_hit": "⌨️ Isolation Forest: цифрларды біреудің айтуымен теру (кідіріс {pause:.1f} сек, түзету: {bs}) (+20)",
        "base_hit": "🧠 Personal Baseline: мінез-құлық моделі клиенттің қалыпты нормасына сәйкес (-35)",
        "norm_ok": "✅ Мінез-құлық биометриясы, bcc.kz бағыты және алушы графы қалыпты.",

        "blocked_title": "Операцияны BCC Qorgau бұғаттады 🛑",
        "blocked_desc": "Әлеуметтік инженерия қаупі жоғары. Растау сұрауы Telegram-дағы сенімді тұлғаға жіберілді.",
        "nbrk_title": "ҚР ҰБ стоп-базасы бойынша қатаң бұғаттау ⛔",
        "nbrk_desc": "Алушы шоты ҚР ҰБ Антифрод-орталығының дропперлер тізімінде. Аударым мүмкін емес.",
        "cooldown_title": "Салқындату кезеңі (15 мин) ⏳",
        "cooldown_desc": "Қысым белгілері анықталды. Аударымды сенімді тұлға арқылы немесе BCC FaceID арқылы растаңыз.",
        "friction_title": "Қосымша қауіпсіздік тексерісі ⚠️",
        "friction_desc": "Жаңа алушыға аударым. Қоңыраусыз өз еркіңізбен әрекет ететініңізді растаңыз.",
        "approved_title": "Аударым орындалды (bcc.kz) ✅",
        "approved_desc": "BCC Qorgau тексерісі сәтті өтті. Қаражат СМП арқылы жіберілді.",

        "tg_approve_title": "Сенімді тұлға растады ✅",
        "tg_approve_desc": "Сенімді тұлға аударымды Telegram-да растады. Профиль Personal Baseline-ға қосылды.",
        "tg_block_title": "Сенімді тұлға бұғаттады 🛑",
        "tg_block_desc": "Сенімді тұлға аударымнан бас тартты! Оқиға БЦК SOC (505/605 желісі) қызметіне берілді.",
        "faceid_title": "BCC FaceID арқылы расталды ✅",
        "faceid_desc": "Белсенді Liveness бар биометриялық бейне-тексеріс сәтті өтті: мәжбүрлеу белгілері жоқ."
    }
}

# ==============================================================================
# 5. PYDANTIC МОДЕЛЬ ВХОДЯЩЕЙ ТРАНЗАКЦИИ
# ==============================================================================
class TransactionTelemetry(BaseModel):
    lang: str = "ru"
    custom_chat_id: Optional[str] = ""
    recipient_type: str = "known"
    recipient: str = "Получатель"
    amount: float = 0.0
    balance: float = 600000.0
    clickstream_anomaly: bool = False
    active_unknown_call: bool = False
    recent_loan: bool = False
    remote_access_rat: bool = False
    transit_mule_inflow: bool = False
    sim_swapped: bool = False
    was_pasted: bool = False
    has_baseline: bool = False
    avg_key_delay_ms: float = 220.0
    max_pause_sec: float = 0.0
    backspace_count: int = 0
    total_input_time_sec: float = 2.0

# ==============================================================================
# 6. TELEGRAM BOT (МНОГОПОЛЬЗОВАТЕЛЬСКИЙ POLLING + INLINE КНОПКИ + КОДЫ 1/2)
# ==============================================================================
def tg_send(chat_id: str, text: str, reply_markup: Optional[dict] = None) -> None:
    if not BOT_TOKEN or "ВАШ_ТОКЕН" in BOT_TOKEN or not chat_id:
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload: Dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML"
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception:
        pass


def handle_trusted_decision(cid: str, decision: str) -> None:
    global latest_tx_state
    if latest_tx_state["id"] <= 0:
        tg_send(cid, "ℹ️ Сейчас нет активной транзакции, ожидающей подтверждения.")
        return

    lang = latest_tx_state.get("lang", "ru")
    t = TEXTS.get(lang, TEXTS["ru"])

    if decision == "approve":
        latest_tx_state["status"] = "APPROVED"
        latest_tx_state["action_title"] = t["tg_approve_title"]
        latest_tx_state["action_desc"] = t["tg_approve_desc"]
        latest_tx_state["learned_baseline"] = True
        if soc_logs:
            soc_logs[0]["status"] = "APPROVED"
        tg_send(cid, f"✅ Операция <b>#{latest_tx_state['id']}</b> подтверждена! Клиент в bcc.kz получил разрешение на перевод.")
    elif decision == "block":
        latest_tx_state["status"] = "BLOCKED"
        latest_tx_state["action_title"] = t["tg_block_title"]
        latest_tx_state["action_desc"] = t["tg_block_desc"]
        if soc_logs:
            soc_logs[0]["status"] = "BLOCKED"
        tg_send(cid, f"🛑 Операция <b>#{latest_tx_state['id']}</b> заблокирована! Инцидент передан в службу безопасности БЦК (505/605).")


def telegram_polling_worker() -> None:
    global last_active_chat
    offset = 0
    while True:
        if not BOT_TOKEN or "ВАШ_ТОКЕН" in BOT_TOKEN:
            time.sleep(5)
            continue
        try:
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
            resp = requests.get(
                url,
                params={"offset": offset, "timeout": 20},
                timeout=25
            ).json()

            for upd in resp.get("result", []):
                offset = upd["update_id"] + 1

                # 1. Текстовые команды (/start, /id, 1, 2)
                if "message" in upd:
                    msg = upd["message"]
                    cid = str(msg["chat"]["id"])
                    text = msg.get("text", "").strip()
                    first_name = msg.get("from", {}).get("first_name", "Участник")

                    subscribed_chats.add(cid)
                    last_active_chat = cid

                    if text.startswith("/start") or text.startswith("/id"):
                        tg_send(
                            cid,
                            f"🛡️ <b>BCC Qorgau Anti-Fraud Bot подключен!</b>\n\n"
                            f"Здравствуйте, <b>{first_name}</b>!\n"
                            f"Ваш личный <b>Telegram ID</b>: <code>{cid}</code>\n\n"
                            f"📌 <b>Как протестировать:</b>\n"
                            f"1. Вы уже в списке активных получателей — просто запустите любой опасный сценарий на сайте (например, <i>«2. Депозит Рахмет → СМП»</i>).\n"
                            f"2. Или вставьте ваш ID <code>{cid}</code> в поле <b>TG ID</b> в верхней панели сайта.\n"
                            f"3. При получении тревоги нажмите кнопку под сообщением или отправьте цифру <b>1</b> (Разрешить) / <b>2</b> (Заблокировать)."
                        )
                    elif text == "1":
                        handle_trusted_decision(cid, "approve")
                    elif text == "2":
                        handle_trusted_decision(cid, "block")

                # 2. Нажатия Inline-кнопок под алертом
                if "callback_query" in upd:
                    cb = upd["callback_query"]
                    cid = str(cb["message"]["chat"]["id"])
                    cb_data = cb.get("data", "")

                    try:
                        requests.post(
                            f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
                            json={"callback_query_id": cb["id"]},
                            timeout=3
                        )
                    except Exception:
                        pass

                    if cb_data.startswith("approve_"):
                        handle_trusted_decision(cid, "approve")
                    elif cb_data.startswith("block_"):
                        handle_trusted_decision(cid, "block")

        except Exception:
            time.sleep(2)


threading.Thread(target=telegram_polling_worker, daemon=True).start()

# ==============================================================================
# 7. API ЭНДПОИНТЫ (ВЕБ-ИНТЕРФЕЙС, СКОРИНГ, FACEID, LIVE SOC)
# ==============================================================================
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(base_dir, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "BCC Qorgau",
        "subscribed_tg_users": len(subscribed_chats)
    }


@app.get("/api/tx-status")
async def get_tx_status():
    return {
        "state": latest_tx_state,
        "logs": soc_logs[:12],
        "active_users": len(subscribed_chats)
    }


@app.post("/api/video-verify")
async def video_verify():
    lang = latest_tx_state.get("lang", "ru")
    t = TEXTS.get(lang, TEXTS["ru"])

    latest_tx_state["status"] = "APPROVED"
    latest_tx_state["action_title"] = t["faceid_title"]
    latest_tx_state["action_desc"] = t["faceid_desc"]
    if soc_logs:
        soc_logs[0]["status"] = "APPROVED"

    return {
        "state": latest_tx_state,
        "logs": soc_logs[:12]
    }


@app.post("/api/analyze")
async def analyze_transaction(tx: TransactionTelemetry):
    global tx_counter, latest_tx_state
    tx_counter += 1

    lang = tx.lang if tx.lang in TEXTS else "ru"
    t = TEXTS[lang]

    risk: int = 5
    reasons: List[str] = []
    main_vector: str = "Норма (Whitelist)"
    is_hard_nbrk: bool = False

    clean_recipient_digits = "".join(ch for ch in tx.recipient if ch.isdigit())
    if tx.recipient_type == "nbrk_blacklist" or "7006660000" in clean_recipient_digits:
        is_hard_nbrk = True

    # 1. Low-Friction фильтр: небольшие переводы без критических угроз не тормозятся
    if (
        tx.amount <= 50000
        and not is_hard_nbrk
        and not tx.remote_access_rat
        and not tx.active_unknown_call
        and tx.recipient_type == "known"
    ):
        risk = 8
        reasons.append(t["low_friction"])
    else:
        # 2. Жесткая проверка по стоп-базе Антифрод-центра НБРК
        if is_hard_nbrk:
            risk = 99
            main_vector = "Стоп-база НБРК"
            reasons.append(t["nbrk_hit"])

        # 3. Детектор трансляции экрана / удаленного доступа (AnyDesk / UIScreen.isCaptured)
        if tx.remote_access_rat:
            risk += 60
            main_vector = "Screen Share / RAT"
            reasons.append(t["rat_hit"])

        # 4. Предиктивный Clickstream-маршрут в bcc.kz (Депозит «Рахмет» -> #картакарта -> СМП)
        if tx.clickstream_anomaly:
            risk += 35
            if not is_hard_nbrk and not tx.remote_access_rat:
                main_vector = "Депозит Рахмет → СМП"
            reasons.append(t["path_hit"])

        # 5. Активный голосовой вызов (CallKit / Telecom State)
        if tx.active_unknown_call:
            risk += 30
            if main_vector == "Норма (Whitelist)":
                main_vector = "CallKit + СМП"
            reasons.append(t["call_hit"])

        # 6. Свежий онлайн-кредит БЦК (< 15 минут)
        if tx.recent_loan:
            risk += 25
            reasons.append(t["loan_hit"])

        # 7. Транзитный дроп / смена IMSI SIM-карты
        if tx.transit_mule_inflow or tx.sim_swapped:
            risk += 30
            if main_vector == "Норма (Whitelist)":
                main_vector = "Дроп / IMSI SIM"
            reasons.append(t["mule_hit"])

        # 8. Фейковый QR-счет BCC Pay
        if tx.recipient_type == "fake_invoice":
            risk += 45
            main_vector = "Фейк-счет QR"
            reasons.append(t["inv_hit"])

        # 9. Корпоративный контур BCC Business (линия 605)
        if tx.recipient_type == "bcc_business":
            risk += 40
            main_vector = "BCC Business (605)"
            reasons.append(t["biz_hit"])

        # 10. Контекстный Copy-Paste (учитывается только при наличии сопутствующих угроз)
        if tx.was_pasted:
            risk += 20
            reasons.append(t["paste_hit"])

        # 11. ML-анализ психомоторики ввода цифр (Isolation Forest)
        features = np.array([[
            tx.avg_key_delay_ms,
            tx.max_pause_sec,
            tx.backspace_count,
            tx.total_input_time_sec
        ]])
        iso_pred = iso_forest.predict(features)[0]
        if tx.max_pause_sec >= 2.5 or (iso_pred == -1 and tx.amount > 50000 and tx.recipient_type != "known"):
            risk += 20
            reasons.append(t["iso_hit"].format(pause=tx.max_pause_sec, bs=tx.backspace_count))

        # 12. Personal Baseline (самообученная поведенческая норма клиента)
        if tx.has_baseline and not is_hard_nbrk and not tx.remote_access_rat:
            risk = max(10, risk - 35)
            reasons.append(t["base_hit"])

    risk = min(99, max(5, risk))

    # Итоговое решение Anti-Fraud движка
    if is_hard_nbrk:
        status = "BLOCKED"
        action_title = t["nbrk_title"]
        action_desc = t["nbrk_desc"]
    elif risk >= 75:
        status = "BLOCKED"
        action_title = t["blocked_title"]
        action_desc = t["blocked_desc"]
    elif risk >= 45:
        status = "COOLDOWN"
        action_title = t["cooldown_title"]
        action_desc = t["cooldown_desc"]
    elif risk >= 30:
        status = "FRICTION"
        action_title = t["friction_title"]
        action_desc = t["friction_desc"]
    else:
        status = "APPROVED"
        action_title = t["approved_title"]
        action_desc = t["approved_desc"]
        if not reasons:
            reasons.append(t["norm_ok"])

    latest_tx_state = {
        "id": tx_counter,
        "status": status,
        "action_title": action_title,
        "action_desc": action_desc,
        "learned_baseline": False,
        "lang": lang
    }

    soc_logs.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "tx_id": f"#{tx_counter}",
        "recipient": tx.recipient[:30],
        "amount": f"{int(tx.amount):,} ₸".replace(",", " "),
        "risk": risk,
        "vector": main_vector,
        "status": status
    })

    # Отправка уведомления в Telegram доверенному лицу (если не жесткий черный список НБРК)
    if status in ["BLOCKED", "COOLDOWN"] and not is_hard_nbrk:
        alert_text = (
            f"🚨 <b>BCC Qorgau × bcc.kz — ТРЕВОГА #{tx_counter}</b>\n\n"
            f"👤 <b>Получатель:</b> {tx.recipient}\n"
            f"💰 <b>Сумма:</b> {int(tx.amount):,} ₸\n"
            f"⚠️ <b>Индекс риска AI:</b> {risk}/100 ({main_vector})\n\n"
            f"<i>Клиент может находиться под воздействием телефонных мошенников. Подтвердить операцию?</i>\n"
            f"(Нажмите кнопку ниже или отправьте <b>1</b> — Разрешить / <b>2</b> — Заблокировать)"
        )
        inline_kb = {
            "inline_keyboard": [[
                {"text": "✅ Разрешить (1)", "callback_data": f"approve_{tx_counter}"},
                {"text": "🛑 Заблокировать (2)", "callback_data": f"block_{tx_counter}"}
            ]]
        }

        targets = set()
        custom_id = (tx.custom_chat_id or "").strip()
        if custom_id:
            targets.add(custom_id)
        elif subscribed_chats:
            targets.update(subscribed_chats)

        for tid in targets:
            tg_send(tid, alert_text, inline_kb)

    return JSONResponse({
        "tx_id": tx_counter,
        "status": status,
        "risk_score": risk,
        "action_title": action_title,
        "action_desc": action_desc,
        "reasons": reasons,
        "logs": soc_logs[:12]
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
