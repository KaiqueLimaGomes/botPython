from botcity.core import DesktopBot
from botcity.maestro import *
import pyautogui
import time
import re
from collections import Counter
import math
import random
import threading

import pytesseract
import cv2
import numpy as np

# =========================
# BOTCITY / OCR SETUP
# =========================
BotMaestroSDK.RAISE_NOT_CONNECTED = False
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# =========================
# CONFIG
# =========================
HUD_BOX = (202, 36, 59, 24)            # HUD X,Y (x,y,w,h)
HUD_BOX_VALOR = (1394, 260, 39, 22)    # HUD do valor/pontos (x,y,w,h)
LEVEL_BOX = (1316, 188, 31, 27)        # HUD do level (x,y,w,h)

CEMITERIO_ROUTE = [
    (134, 139),
    (134, 150),
    (134, 166),
    (134, 189),
    (140, 203),
    (141, 215),
    (133, 224),  # final
]
TARGET_FUTURO_LOR = (133, 224)

BUTTON_BOX = (223, 60, 26, 28)

FUTURO_TIMEOUT_S = 220

LEVEL_CHECK_EVERY_S = 60
TARGET_LEVEL = 100

MOVE_CMD = "/move arena"
CHAT_OPEN_WAIT_S = 5
CHAT_AFTER_TYPE_WAIT_S = 5
AFTER_MOVE_WAIT_S = 10

INITIAL_C_OPEN_DELAY = 1.10
INITIAL_C_READ_DELAY = 0.80
INITIAL_C_CLOSE_DELAY = 0.35

# =========================
# LIMITES / OCR
# =========================
X_MIN, X_MAX = 0, 255
Y_MIN, Y_MAX = 0, 255
MAX_JUMP = 30
_last_good = None

# =========================
# NAVEGAÇÃO POR SETAS (VETORIAL)
# =========================
AFTER_KEY_PAUSE_S = 2.5
HOLD_S = 0.10

STUCK_SAME_READS = 6

X_BAND_MIN = 133
X_BAND_MAX = 135

LANE_WEIGHT = 2.0
WRONG_WAY_WEIGHT = 0.6

KEY_UP = "up"
KEY_DOWN = "down"
KEY_LEFT = "left"
KEY_RIGHT = "right"

# Vetores medidos por você
ACTIONS = [
    {"name": "UP",          "keys": [KEY_UP],                "dx": -3, "dy": +3},
    {"name": "DOWN",        "keys": [KEY_DOWN],              "dx": +3, "dy": -3},
    {"name": "LEFT",        "keys": [KEY_LEFT],              "dx": -3, "dy": -3},
    {"name": "RIGHT",       "keys": [KEY_RIGHT],             "dx": +3, "dy": +3},

    {"name": "UP+LEFT",     "keys": [KEY_UP, KEY_LEFT],      "dx": -4, "dy":  0},
    {"name": "UP+RIGHT",    "keys": [KEY_UP, KEY_RIGHT],     "dx":  0, "dy": +4},
    {"name": "DOWN+RIGHT",  "keys": [KEY_DOWN, KEY_RIGHT],   "dx": +4, "dy":  0},
    {"name": "DOWN+LEFT",   "keys": [KEY_DOWN, KEY_LEFT],    "dx":  0, "dy": -4},
]

# =========================
# STUCK / ESCAPE CONFIG
# =========================
ESCAPE_TRIES = 6
ESCAPE_TOPK = 4
ESCAPE_RANDOM_EPS = 0.25
ESCAPE_MAX_GLOBAL = 3

# =========================
# LOG
# =========================
def log(msg: str):
    print(msg, flush=True)

def not_found(label: str):
    log(f"Element not found: {label}")

# =========================
# INPUT HELPERS
# =========================
def click_box_center(box):
    x, y, w, h = box
    px = x + (w // 2)
    py = y + (h // 2)
    pyautogui.click(px, py)

def _key_down(k: str):
    pyautogui.keyDown(k)

def _key_up(k: str):
    pyautogui.keyUp(k)

def press_keys_simultaneous(keys, hold_s=HOLD_S, after_pause_s=AFTER_KEY_PAUSE_S):
    """
    Pressiona (keyDown) todas as teclas AO MESMO TEMPO (via threads),
    segura por hold_s (mesmo tempo para todas),
    e solta (keyUp) todas AO MESMO TEMPO.
    """
    keys = list(keys)

    # DOWN simultâneo
    threads = []
    for k in keys:
        t = threading.Thread(target=_key_down, args=(k,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    time.sleep(max(0.01, hold_s))

    # UP simultâneo
    threads = []
    for k in keys:
        t = threading.Thread(target=_key_up, args=(k,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    time.sleep(after_pause_s)

# =========================
# LUGAR (Lorencia / Arena)
# =========================
def ensure_lorencia(bot: DesktopBot) -> bool:
    try:
        if bot.find("Lorencia", matching=0.75, waiting_time=2500):
            log("Lorencia detectada (template).")
            return True
    except Exception:
        pass

    try:
        if bot.find_text("Lorencia", waiting_time=2500):
            log("Lorencia detectada (texto).")
            return True
    except Exception:
        pass

    not_found("Lorencia (template/texto)")
    return False

def ensure_arena(bot: DesktopBot) -> bool:
    for name in ("Arena", "arena", "ARENA"):
        try:
            if bot.find(name, matching=0.75, waiting_time=8000):
                log("Arena detectada (template).")
                return True
        except Exception:
            pass

    try:
        if bot.find_text("Arena", waiting_time=8000):
            log("Arena detectada (texto).")
            return True
    except Exception:
        pass

    not_found("Arena (template/texto)")
    return False

# =========================
# OCR
# =========================
def preprocess_for_ocr(rgb_img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    th = cv2.dilate(th, kernel, iterations=1)
    return th

def ocr_read_coords_once(hud_box) -> tuple[int, int] | None:
    x, y, w, h = hud_box
    shot = pyautogui.screenshot(region=(x, y, w, h))
    rgb = np.array(shot)

    proc = preprocess_for_ocr(rgb)
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789,'
    text = pytesseract.image_to_string(proc, config=config)

    matches = re.findall(r'(\d{1,3})\s*,\s*(\d{1,3})', text)
    if not matches:
        return None

    xs, ys = matches[-1]
    xx, yy = int(xs), int(ys)

    if not (X_MIN <= xx <= X_MAX and Y_MIN <= yy <= Y_MAX):
        return None

    return xx, yy

def ocr_read_value_once(hud_box) -> int | None:
    x, y, w, h = hud_box
    shot = pyautogui.screenshot(region=(x, y, w, h))
    rgb = np.array(shot)

    proc = preprocess_for_ocr(rgb)
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(proc, config=config)

    matches = re.findall(r'\d+', text)
    if not matches:
        return None

    try:
        return int(matches[0])
    except (ValueError, IndexError):
        return None

def get_current_xy_filtered(hud_box, samples=7, delay=0.07) -> tuple[int, int] | None:
    global _last_good
    vals = []
    for _ in range(samples):
        v = ocr_read_coords_once(hud_box)
        if v:
            vals.append(v)
        time.sleep(delay)

    if not vals:
        return None

    candidate = Counter(vals).most_common(1)[0][0]

    if _last_good is None:
        _last_good = candidate
        return candidate

    lx, ly = _last_good
    cx, cy = candidate

    if abs(cx - lx) > MAX_JUMP or abs(cy - ly) > MAX_JUMP:
        return _last_good

    _last_good = candidate
    return candidate

def get_current_value_filtered(hud_box, samples=6, delay=0.08) -> int | None:
    vals = []
    for _ in range(samples):
        v = ocr_read_value_once(hud_box)
        if v is not None:
            vals.append(v)
        time.sleep(delay)

    if not vals:
        return None

    return Counter(vals).most_common(1)[0][0]

def get_level_filtered(samples=7, delay=0.08) -> int | None:
    vals = []
    for _ in range(samples):
        v = ocr_read_value_once(LEVEL_BOX)
        if v is not None:
            vals.append(v)
        time.sleep(delay)
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]

# =========================
# PLANNER
# =========================
def squared_dist(x1, y1, x2, y2):
    dx = (x2 - x1)
    dy = (y2 - y1)
    return dx*dx + dy*dy

def lane_penalty(nx: int, enforce_lane: bool) -> float:
    if not enforce_lane:
        return 0.0
    if X_BAND_MIN <= nx <= X_BAND_MAX:
        return 0.0
    if nx < X_BAND_MIN:
        return (X_BAND_MIN - nx) * LANE_WEIGHT
    return (nx - X_BAND_MAX) * LANE_WEIGHT

def wrong_way_penalty(dx_to_target: int, dy_to_target: int, ax: int, ay: int) -> float:
    pen = 0.0
    if dx_to_target > 0 and ax < 0: pen += abs(ax) * WRONG_WAY_WEIGHT
    if dx_to_target < 0 and ax > 0: pen += abs(ax) * WRONG_WAY_WEIGHT
    if dy_to_target > 0 and ay < 0: pen += abs(ay) * WRONG_WAY_WEIGHT
    if dy_to_target < 0 and ay > 0: pen += abs(ay) * WRONG_WAY_WEIGHT
    return pen

def score_action(cur, target, enforce_lane: bool, action) -> float:
    cx, cy = cur
    tx, ty = target
    dx = tx - cx
    dy = ty - cy
    nx = cx + action["dx"]
    ny = cy + action["dy"]
    score = squared_dist(nx, ny, tx, ty)
    score += lane_penalty(nx, enforce_lane)
    score += wrong_way_penalty(dx, dy, action["dx"], action["dy"])
    return score

def rank_actions(cur, target, enforce_lane: bool):
    ranked = []
    for a in ACTIONS:
        ranked.append((score_action(cur, target, enforce_lane, a), a))
    ranked.sort(key=lambda t: t[0])
    return ranked

def choose_best_action(cur, target, enforce_lane: bool):
    return rank_actions(cur, target, enforce_lane)[0][1]

def log_navigation_profile():
    log("[NAV] MODO SETAS (VETORIAL) ativo:")
    log(f"      after_key_pause={AFTER_KEY_PAUSE_S}s | hold={HOLD_S}s | stuck_reads={STUCK_SAME_READS}")
    log(f"      X_BAND (coluna 134): {X_BAND_MIN}..{X_BAND_MAX} | lane_weight={LANE_WEIGHT} | wrong_way_weight={WRONG_WAY_WEIGHT}")
    log("      ações:")
    for a in ACTIONS:
        log(f"        - {a['name']:>11} keys={a['keys']} delta=({a['dx']:+d},{a['dy']:+d})")

# =========================
# ESCAPE WHEN STUCK
# =========================
def escape_when_stuck(hud_box, target_xy, enforce_lane: bool) -> bool:
    cur0 = get_current_xy_filtered(hud_box, samples=7, delay=0.07)
    if not cur0:
        return False

    ranked = rank_actions(cur0, target_xy, enforce_lane)
    top = [a for _, a in ranked[:ESCAPE_TOPK]]

    pool = top[:] + [a for _, a in ranked[ESCAPE_TOPK:]]
    seen = set()
    pool2 = []
    for a in pool:
        if a["name"] not in seen:
            pool2.append(a)
            seen.add(a["name"])
    pool = pool2

    log(f"[ESCAPE] Tentando destravar: tries={ESCAPE_TRIES} topK={ESCAPE_TOPK}")

    for i in range(1, ESCAPE_TRIES + 1):
        if random.random() < ESCAPE_RANDOM_EPS:
            a = random.choice(top)
            why = "random_topK"
        else:
            a = pool[(i - 1) % len(pool)]
            why = "cycle_pool"

        log(f"[ESCAPE] {i}/{ESCAPE_TRIES} -> {a['name']} ({why}) keys={a['keys']} delta=({a['dx']},{a['dy']})")
        press_keys_simultaneous(a["keys"], hold_s=HOLD_S)

        cur1 = get_current_xy_filtered(hud_box, samples=7, delay=0.07)
        if not cur1:
            continue

        if cur1 != cur0:
            log(f"[ESCAPE] Sucesso: moveu de {cur0} -> {cur1}")
            return True

    log("[ESCAPE] Falhou: não conseguiu sair do lugar.")
    return False

# =========================
# WALK (VETORIAL)
# =========================
def walk_to_arrows(hud_box, target_xy: tuple[int, int], label="ALVO",
                   tol_x=1, tol_y=2,
                   timeout_s=90, enforce_x_lane=False) -> bool:
    log(f"Indo até {label} {target_xy} (ARROWS-VECT) tol_x={tol_x} tol_y={tol_y} timeout={timeout_s}s lane={enforce_x_lane}")
    start = time.time()

    last_xy = None
    same_reads = 0
    ocr_fail = 0
    step = 0
    escape_cycles = 0

    while True:
        if time.time() - start > timeout_s:
            log(f"[TIMEOUT] walk_to_arrows {label} excedeu {timeout_s}s")
            return False

        cur = get_current_xy_filtered(hud_box, samples=7, delay=0.07)
        if not cur:
            ocr_fail += 1
            log(f"[WARN] OCR não leu coordenadas... ({ocr_fail})")
            time.sleep(0.2)
            if ocr_fail >= 3:
                log("[ACTION] OCR falhando -> passo reset (DOWN+LEFT)")
                press_keys_simultaneous([KEY_DOWN, KEY_LEFT], hold_s=HOLD_S)
                ocr_fail = 0
            continue

        ocr_fail = 0
        cx, cy = cur
        tx, ty = target_xy
        dx = tx - cx
        dy = ty - cy
        step += 1

        log(f"[step {step}] [XY] atual=({cx},{cy}) alvo=({tx},{ty}) dx={dx} dy={dy}")

        if abs(dx) <= tol_x and abs(dy) <= tol_y:
            log(f"Chegou em {label} (<=tol_x={tol_x}, tol_y={tol_y}).")
            return True

        if last_xy is not None and cur == last_xy:
            same_reads += 1
        else:
            same_reads = 0
        last_xy = cur

        if same_reads >= STUCK_SAME_READS:
            log(f"[STUCK] {same_reads} leituras iguais -> ESCAPE (tentar outras direções)")
            ok = escape_when_stuck(hud_box, target_xy, enforce_x_lane)
            same_reads = 0
            if ok:
                escape_cycles = 0
                continue
            escape_cycles += 1
            if escape_cycles >= ESCAPE_MAX_GLOBAL:
                log("[STUCK] muitas tentativas de escape sem sucesso -> aborta este alvo")
                return False
            continue

        a = choose_best_action(cur, target_xy, enforce_x_lane)
        log(f"[MOVE] {a['name']} keys={a['keys']} delta=({a['dx']},{a['dy']})")
        press_keys_simultaneous(a["keys"], hold_s=HOLD_S)

# =========================
# ROTA CHECKPOINTS
# =========================
def walk_route_with_checkpoints_arrows(hud_box, checkpoints, total_timeout_s=220) -> bool:
    log(f"[ROUTE] (ARROWS-VECT) Iniciando rota com {len(checkpoints)} checkpoints. Timeout total={total_timeout_s}s")
    start = time.time()

    for idx, cp in enumerate(checkpoints, 1):
        remaining = total_timeout_s - (time.time() - start)
        if remaining <= 0:
            log("[ROUTE][TIMEOUT] acabou o tempo total da rota")
            return False

        base = 80 if idx <= 2 else 65
        slice_timeout = max(base, int(remaining / (len(checkpoints) - idx + 1)))

        enforce_lane = (cp[0] == 134)
        tol_x = 1 if enforce_lane else 2
        tol_y = 2

        log(f"[ROUTE] CP{idx}/{len(checkpoints)} -> {cp} timeout={slice_timeout}s lane={enforce_lane}")

        ok = walk_to_arrows(
            hud_box,
            cp,
            label=f"CP{idx}",
            tol_x=tol_x,
            tol_y=tol_y,
            timeout_s=slice_timeout,
            enforce_x_lane=enforce_lane
        )
        if not ok:
            log(f"[ROUTE] Falhou no checkpoint {idx}: {cp}")
            return False

    log("[ROUTE] (ARROWS-VECT) Rota concluída com sucesso.")
    return True

# =========================
# JIN (clique offsets, mantido)
# =========================
def click_relative_safe(dx: int, dy: int, center_y_ratio=0.55):
    w, h = pyautogui.size()
    cx = w // 2
    cy = int(h * center_y_ratio)
    pyautogui.click(cx + dx, cy + dy)

def go_exact_arrows(hud_box, xy: tuple[int,int], label: str) -> bool:
    return walk_to_arrows(hud_box, xy, label=label, tol_x=1, tol_y=1, timeout_s=120, enforce_x_lane=False)

def click_jin_by_offsets(hud_box, base_xy=(135,126)) -> bool:
    if not go_exact_arrows(hud_box, base_xy, "BASE_JIN"):
        return False

    offsets = [
        (0, -180), (0, -160), (0, -140), (0, -120),
        (40, -180), (-40, -180),
        (60, -160), (-60, -160),
        (80, -140), (-80, -140),
        (100, -120), (-100, -120),
        (120, -110), (-120, -110),
        (140, -100), (-140, -100),
        (160, -90), (-160, -90),
        (0, -100),
    ]

    log("[JIN] Tentando cliques por offsets (sem mover)...")

    for i, (dx, dy) in enumerate(offsets, 1):
        before = get_current_xy_filtered(hud_box, samples=7, delay=0.07)
        if not before:
            log("[JIN] OCR falhou antes do clique, tentando próximo...")
            continue

        log(f"[JIN] tentativa {i}/{len(offsets)} click({dx},{dy}) from={before}")
        click_relative_safe(dx, dy)
        time.sleep(0.60)

        after = get_current_xy_filtered(hud_box, samples=7, delay=0.07)
        if not after:
            log("[JIN] OCR falhou após clique, tentando próximo...")
            continue

        moved = (abs(after[0] - before[0]) > 1) or (abs(after[1] - before[1]) > 1)
        if moved:
            log(f"[JIN] virou movimento (before={before} after={after}) -> próximo offset")
            continue

        log(f"[JIN] clique sem mover (before={before} after={after}) -> OK provável")
        return True

    log("[JIN] Não conseguiu clicar sem mover. Ajuste offsets.")
    return False

# =========================
# STATS / CHAT
# =========================
def get_initial_level_from_c(open_delay=INITIAL_C_OPEN_DELAY,
                             read_delay=INITIAL_C_READ_DELAY,
                             close_delay=INITIAL_C_CLOSE_DELAY) -> int | None:
    pyautogui.press('c')
    time.sleep(open_delay)
    time.sleep(read_delay)
    lvl = get_level_filtered(samples=8, delay=0.09)
    pyautogui.press('c')
    time.sleep(close_delay)
    return lvl

def read_points_from_c(hud_box_valor, open_delay=0.35, read_delay=0.35, close_delay=0.15) -> int | None:
    pyautogui.press('c')
    time.sleep(open_delay)
    time.sleep(read_delay)
    val = get_current_value_filtered(hud_box_valor, samples=6, delay=0.08)
    pyautogui.press('c')
    time.sleep(close_delay)
    return val

def send_chat_command(cmd: str, value: int,
                      open_delay=0.35,
                      type_delay=0.15,
                      send_delay=0.35):
    log(f"[CHAT] enviando: {cmd} {value}")
    pyautogui.press('enter')
    time.sleep(open_delay)
    pyautogui.write(f"{cmd} {value}", interval=0.02)
    time.sleep(type_delay)
    pyautogui.press('enter')
    time.sleep(send_delay)

def apply_stat_with_validation(cmd: str, value: int, hud_box_valor,
                               retries=2,
                               open_delay=0.35,
                               type_delay=0.15,
                               send_delay=0.35) -> bool:
    before = read_points_from_c(hud_box_valor)
    log(f"[VAL] antes de {cmd}: pontos={before}")

    for attempt in range(1, retries + 1):
        send_chat_command(cmd, value, open_delay=open_delay, type_delay=type_delay, send_delay=send_delay)

        after = read_points_from_c(hud_box_valor)
        log(f"[VAL] depois de {cmd} (tentativa {attempt}/{retries}): pontos={after}")

        if after is None or before is None:
            log("[VAL] OCR do valor falhou -> tentando novamente")
            time.sleep(0.3)
            continue

        if after < before:
            log(f"[VAL] OK: pontos diminuíram ({before} -> {after})")
            return True

        log(f"[VAL] pontos não mudaram ({before} -> {after}) -> retry")
        time.sleep(0.35)

    log(f"[VAL] Falhou em validar aplicação do comando {cmd}.")
    return False

# =========================
# UP: CEMITÉRIO -> LVL 100 -> MOVE ARENA
# =========================
def go_cemiterio_and_up_until_100_then_move_arena(bot: DesktopBot):
    log(f"[UP] Indo para CEMITÉRIO via checkpoints (ARROWS-VECT) timeout total {FUTURO_TIMEOUT_S}s...")

    ok = walk_route_with_checkpoints_arrows(
        HUD_BOX,
        CEMITERIO_ROUTE,
        total_timeout_s=FUTURO_TIMEOUT_S
    )
    if not ok:
        not_found("não conseguiu completar a rota do cemitério dentro do tempo")
        return

    log(f"[UP] Chegou no CEMITÉRIO FINAL {TARGET_FUTURO_LOR}. Clicando botão {BUTTON_BOX}...")
    click_box_center(BUTTON_BOX)
    time.sleep(0.8)

    log("[UP] Monitorando level (1 minuto) até >= 100...")
    while True:
        lvl = get_level_filtered(samples=7, delay=0.08)
        log(f"[LEVEL] atual: {lvl}")

        if lvl is not None and lvl >= TARGET_LEVEL:
            log(f"[UP] Level {lvl} atingiu >= {TARGET_LEVEL}.")
            break

        time.sleep(LEVEL_CHECK_EVERY_S)

    log("[UP] Fechando janela com tecla 'c'...")
    pyautogui.press('c')
    time.sleep(0.4)

    log("[UP] Enviando /move arena...")
    pyautogui.press('enter')
    time.sleep(CHAT_OPEN_WAIT_S)

    pyautogui.write(MOVE_CMD, interval=0.02)
    time.sleep(CHAT_AFTER_TYPE_WAIT_S)

    pyautogui.press('enter')
    log("[UP] /move arena enviado. Aguardando 10s...")
    time.sleep(AFTER_MOVE_WAIT_S)

    log("[UP] Verificando se está em Arena...")
    if ensure_arena(bot):
        log("[OK] Entrou em Arena.")
    else:
        log("[WARN] Não confirmei Arena. (Se possível, adicione Arena.png nos resources.)")

# =========================
# MAIN
# =========================
def main():
    maestro = BotMaestroSDK.from_sys_args()
    execution = maestro.get_execution()
    log(f"Task ID is: {execution.task_id}")
    log(f"Task Parameters are: {execution.parameters}")

    bot = DesktopBot()

    if not ensure_lorencia(bot):
        return

    log(f"HUD_BOX = {HUD_BOX}")
    log_navigation_profile()

    level = get_initial_level_from_c()
    log(f"[LEVEL] level filtrado (início): {level}")

    if level is not None and 1 < level < 100:
        log("[FLOW] level > 1 e < 100 -> ir para cemitério (ARROWS-VECT) e upar até 100 -> arena.")
        go_cemiterio_and_up_until_100_then_move_arena(bot)
        return

    if level is None:
        log("[FLOW] Não consegui ler level -> por segurança, indo cemitério.")
        go_cemiterio_and_up_until_100_then_move_arena(bot)
        return

    if level >= 100:
        log("[FLOW] level >= 100 -> enviando /move arena direto.")
        pyautogui.press('c')
        time.sleep(0.2)
        pyautogui.press('enter')
        time.sleep(CHAT_OPEN_WAIT_S)
        pyautogui.write(MOVE_CMD, interval=0.02)
        time.sleep(CHAT_AFTER_TYPE_WAIT_S)
        pyautogui.press('enter')
        time.sleep(AFTER_MOVE_WAIT_S)
        ensure_arena(bot)
        return

    log("[FLOW] level == 1 -> verificar pontos antes de distribuir.")

    valor = read_points_from_c(
        HUD_BOX_VALOR,
        open_delay=INITIAL_C_OPEN_DELAY,
        read_delay=INITIAL_C_READ_DELAY,
        close_delay=INITIAL_C_CLOSE_DELAY
    )
    if valor is None:
        log("[WARN] Não foi possível ler o valor/pontos. Indo cemitério por segurança.")
        go_cemiterio_and_up_until_100_then_move_arena(bot)
        return

    log(f"[PONTOS] valor lido: {valor}")

    if valor == 0:
        log("[FLOW] level==1 e pontos==0 -> ir direto pro cemitério.")
        go_cemiterio_and_up_until_100_then_move_arena(bot)
        return

    BASE_JIN = (135, 126)
    if not click_jin_by_offsets(HUD_BOX, base_xy=BASE_JIN):
        not_found("Clique no Jin por offsets falhou")
        return

    log("Jin clicado (provável) por offsets.")

    valor2 = read_points_from_c(
        HUD_BOX_VALOR,
        open_delay=INITIAL_C_OPEN_DELAY,
        read_delay=INITIAL_C_READ_DELAY,
        close_delay=INITIAL_C_CLOSE_DELAY
    )
    if valor2 is None:
        log("[WARN] Não foi possível ler pontos após Jin. Indo cemitério.")
        go_cemiterio_and_up_until_100_then_move_arena(bot)
        return

    log(f"[PONTOS] valor lido pós-Jin: {valor2}")

    if valor2 == 0:
        log("[FLOW] Pós-Jin: pontos==0 -> ir cemitério.")
        go_cemiterio_and_up_until_100_then_move_arena(bot)
        return

    forca = int(round(valor2 * 0.55))
    agilidade = int(round(valor2 * 0.25))
    vitalidade = int(round(valor2 * 0.15))
    energia = int(round(valor2 * 0.05))

    log(f"Calculado -> Força: {forca}, Agilidade: {agilidade}, Vitalidade: {vitalidade}, Energia: {energia}")

    cmds = [('/f', forca), ('/a', agilidade), ('/v', vitalidade), ('/e', energia)]
    for cmd, val in cmds:
        ok = apply_stat_with_validation(cmd, val, HUD_BOX_VALOR, retries=2)
        if not ok:
            log(f"[WARN] Não consegui confirmar {cmd}. Continuando...")

    final_points = read_points_from_c(HUD_BOX_VALOR)
    log(f"[FINAL] pontos após aplicar tudo: {final_points}")

    log("[FLOW] Distribuição concluída -> ir cemitério (ARROWS-VECT) e upar até 100 -> arena.")
    go_cemiterio_and_up_until_100_then_move_arena(bot)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\nEncerrado pelo usuário (Ctrl+C).")
