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
# CONFIG (HUD / BOXES)
# =========================
HUD_BOX = (202, 36, 59, 24)            # HUD X,Y (x,y,w,h)
HUD_BOX_VALOR = (1394, 260, 39, 22)    # HUD do valor/pontos (x,y,w,h)
LEVEL_BOX = (1316, 188, 31, 27)        # HUD do level (x,y,w,h)

# =========================
# ROTAS
# =========================
# Lorencia -> Cemitério
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

# Arena -> Spot Play
ARENA_ROUTE = [
    (56, 47),
    (38, 48),
    (38, 66),
    (38, 74),
    (29, 75),  # final
]
TARGET_ARENA_FINAL = (29, 75)

# Botão HUD
Btn_play = (223, 60, 26, 28)

# =========================
# TEMPOS / LIMITES
# =========================
FUTURO_TIMEOUT_S = 220

LEVEL_CHECK_EVERY_S = 60
TARGET_LEVEL_CEMITERIO = 100
TARGET_LEVEL_RESET = 380

# Chat
CHAT_OPEN_WAIT_S = 0.30
CHAT_AFTER_TYPE_WAIT_S = 0.20
CHAT_AFTER_SEND_WAIT_S = 0.50

MOVE_CMD_ARENA = "/move arena"
RESET_CMD = "/reset"

# Janela C (pedido)
C_OPEN_WAIT_S = 5.0      # aguarda 5s após abrir C (carregar dados)
C_CLOSE_WAIT_S = 0.20

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
    """Pressiona TODAS as teclas ao mesmo tempo pela MESMA duração."""
    keys = list(keys)

    threads = []
    for k in keys:
        t = threading.Thread(target=_key_down, args=(k,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    time.sleep(max(0.01, hold_s))

    threads = []
    for k in keys:
        t = threading.Thread(target=_key_up, args=(k,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    time.sleep(after_pause_s)

def focus_game_window():
    w, h = pyautogui.size()
    pyautogui.click(w // 2, int(h * 0.55))
    time.sleep(0.05)

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

def get_level_filtered(samples=8, delay=0.09) -> int | None:
    vals = []
    for _ in range(samples):
        v = ocr_read_value_once(LEVEL_BOX)
        if v is not None:
            vals.append(v)
        time.sleep(delay)
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]

def get_points_filtered(samples=8, delay=0.09) -> int | None:
    vals = []
    for _ in range(samples):
        v = ocr_read_value_once(HUD_BOX_VALOR)
        if v is not None:
            vals.append(v)
        time.sleep(delay)
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]

# =========================
# JANELA C (LEVEL / POINTS)
# =========================
def open_c_wait():
    focus_game_window()
    pyautogui.press('c')
    time.sleep(C_OPEN_WAIT_S)

def close_c():
    focus_game_window()
    pyautogui.press('c')
    time.sleep(C_CLOSE_WAIT_S)

def read_level_with_c() -> int | None:
    open_c_wait()
    lvl = get_level_filtered(samples=8, delay=0.09)
    close_c()
    return lvl

def read_points_with_c() -> int | None:
    open_c_wait()
    pts = get_points_filtered(samples=8, delay=0.09)
    close_c()
    return pts

# =========================
# CHAT HELPERS
# =========================
def send_chat_line(text: str):
    focus_game_window()
    pyautogui.press('enter')
    time.sleep(CHAT_OPEN_WAIT_S)
    pyautogui.write(text, interval=0.02)
    time.sleep(CHAT_AFTER_TYPE_WAIT_S)
    pyautogui.press('enter')
    time.sleep(CHAT_AFTER_SEND_WAIT_S)

# =========================
# DISTRIBUIÇÃO DE PONTOS (55/25/15/5)
# =========================
def apply_stat_with_validation(cmd: str, value: int, retries=2) -> bool:
    before = read_points_with_c()
    log(f"[VAL] antes de {cmd}: pontos={before}")

    for attempt in range(1, retries + 1):
        log(f"[CHAT] enviando: {cmd} {value}")
        send_chat_line(f"{cmd} {value}")

        after = read_points_with_c()
        log(f"[VAL] depois de {cmd} (tentativa {attempt}/{retries}): pontos={after}")

        if before is None or after is None:
            log("[VAL] OCR falhou -> retry")
            continue

        if after < before:
            log(f"[VAL] OK: pontos diminuíram ({before} -> {after})")
            return True

        log(f"[VAL] pontos não mudaram ({before} -> {after}) -> retry")

    log(f"[VAL] Falhou em validar {cmd}.")
    return False

def distribute_points_55_25_15_5():
    pts = read_points_with_c()
    log(f"[PONTOS] lidos: {pts}")

    if pts is None or pts <= 0:
        log("[PONTOS] sem pontos para distribuir.")
        return

    f = int(round(pts * 0.55))
    a = int(round(pts * 0.25))
    v = int(round(pts * 0.15))
    e = int(round(pts * 0.05))

    total = f + a + v + e
    if total > pts:
        diff = total - pts
        f = max(0, f - diff)

    log(f"[DIST] Força={f} Agi={a} Vit={v} Ene={e}")

    for cmd, val in [('/f', f), ('/a', a), ('/v', v), ('/e', e)]:
        if val <= 0:
            continue
        ok = apply_stat_with_validation(cmd, val, retries=2)
        if not ok:
            log(f"[WARN] não confirmei {cmd}, seguindo...")

    final_pts = read_points_with_c()
    log(f"[DIST] pontos finais: {final_pts}")

# =========================
# JIN (reuso do seu método offsets)
# =========================
def click_relative_safe(dx: int, dy: int, center_y_ratio=0.55):
    w, h = pyautogui.size()
    cx = w // 2
    cy = int(h * center_y_ratio)
    pyautogui.click(cx + dx, cy + dy)

def click_jin_by_offsets(hud_box, base_xy=(135, 126)) -> bool:
    """
    Mesma lógica que você já usava: tenta clicar Jin por offsets sem mover.
    Ajuste os offsets conforme seu client.
    """
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

def jin_interaction():
    """
    NOVA REGRA (pedido):
      - Só roda quando level == 1
      - Antes disso, distribui pontos
    """
    log("[FLOW] level == 1 -> distribuir pontos e depois interação com Jin.")
    distribute_points_55_25_15_5()

    # Se quiser forçar o char ir para a base do Jin antes, você pode chamar walk_to aqui.
    # Eu mantive simples: apenas executa o clique por offsets.
    ok = click_jin_by_offsets(HUD_BOX, base_xy=(135, 126))
    if ok:
        log("[JIN] interação feita (provável).")
    else:
        log("[JIN] falhou interação. Seguindo mesmo assim.")

# =========================
# PLANNER / SCORE
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
        focus_game_window()
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
                focus_game_window()
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
            log(f"[STUCK] {same_reads} leituras iguais -> ESCAPE (outras direções)")
            ok = escape_when_stuck(hud_box, target_xy, enforce_x_lane)
            same_reads = 0
            if ok:
                escape_cycles = 0
                continue
            escape_cycles += 1
            if escape_cycles >= ESCAPE_MAX_GLOBAL:
                log("[STUCK] muitas tentativas sem sucesso -> aborta este alvo")
                return False
            continue

        a = choose_best_action(cur, target_xy, enforce_x_lane)
        log(f"[MOVE] {a['name']} keys={a['keys']} delta=({a['dx']},{a['dy']})")
        focus_game_window()
        press_keys_simultaneous(a["keys"], hold_s=HOLD_S)

# =========================
# ROTA CHECKPOINTS
# =========================
def walk_route_with_checkpoints_arrows(hud_box, checkpoints, total_timeout_s=220, lane_for_x_134=False) -> bool:
    log(f"[ROUTE] (ARROWS-VECT) Iniciando rota com {len(checkpoints)} checkpoints. Timeout total={total_timeout_s}s")
    start = time.time()

    for idx, cp in enumerate(checkpoints, 1):
        remaining = total_timeout_s - (time.time() - start)
        if remaining <= 0:
            log("[ROUTE][TIMEOUT] acabou o tempo total da rota")
            return False

        base = 80 if idx <= 2 else 65
        slice_timeout = max(base, int(remaining / (len(checkpoints) - idx + 1)))

        enforce_lane = (lane_for_x_134 and cp[0] == 134)
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
# MOVE / RESET
# =========================
def send_move_arena_and_verify(bot: DesktopBot) -> bool:
    log("[MOVE] enviando /move arena...")
    send_chat_line(MOVE_CMD_ARENA)
    log("[MOVE] aguardando 5s...")
    time.sleep(5.0)
    log("[MOVE] verificando Arena...")
    return ensure_arena(bot)

def send_reset_and_wait_lorencia(bot: DesktopBot, wait_s=12.0) -> bool:
    log("[RESET] enviando /reset...")
    send_chat_line(RESET_CMD)
    log(f"[RESET] aguardando {wait_s:.0f}s para voltar...")
    time.sleep(wait_s)
    log("[RESET] verificando Lorencia...")
    return ensure_lorencia(bot)

# =========================
# CEMITÉRIO: andar + upar até 100 + ir arena
# =========================
def go_cemiterio_and_up_until_100_then_move_arena(bot: DesktopBot):
    log(f"[UP] Indo para CEMITÉRIO via checkpoints timeout total {FUTURO_TIMEOUT_S}s...")
    ok = walk_route_with_checkpoints_arrows(HUD_BOX, CEMITERIO_ROUTE, total_timeout_s=FUTURO_TIMEOUT_S, lane_for_x_134=True)
    if not ok:
        not_found("não conseguiu completar a rota do cemitério")
        return

    log(f"[UP] Chegou no CEMITÉRIO FINAL {TARGET_FUTURO_LOR}. Clicando Btn_play {Btn_play}...")
    click_box_center(Btn_play)
    time.sleep(0.8)

    log("[UP] Monitorando level a cada 60s até >= 100 (C abre -> espera 5s -> lê -> fecha)...")
    while True:
        lvl = read_level_with_c()
        log(f"[LEVEL] atual: {lvl}")

        if lvl is not None and lvl >= TARGET_LEVEL_CEMITERIO:
            log(f"[UP] Level {lvl} >= {TARGET_LEVEL_CEMITERIO}. Indo Arena.")
            break

        time.sleep(LEVEL_CHECK_EVERY_S)

    ok_arena = send_move_arena_and_verify(bot)
    if ok_arena:
        log("[OK] Entrou em Arena.")
    else:
        log("[WARN] Não confirmei Arena ainda.")

# =========================
# ARENA: distribuir pontos + navegar spot + clicar play + upar até 380 + /reset
# =========================
def arena_flow(bot: DesktopBot):
    log("[ARENA] Entrou em Arena -> distribuir pontos (55/25/15/5).")
    distribute_points_55_25_15_5()

    log("[ARENA] Indo para rota Arena -> destino final.")
    ok = walk_route_with_checkpoints_arrows(HUD_BOX, ARENA_ROUTE, total_timeout_s=360, lane_for_x_134=False)
    if not ok:
        not_found("falhou rota na arena")
        return

    log(f"[ARENA] Chegou no destino {TARGET_ARENA_FINAL}. Clicando Btn_play {Btn_play}...")
    click_box_center(Btn_play)
    time.sleep(0.8)

    log("[ARENA] Monitorando level a cada 60s até >= 380...")
    while True:
        lvl = read_level_with_c()
        log(f"[LEVEL] atual: {lvl}")

        if lvl is not None and lvl >= TARGET_LEVEL_RESET:
            log(f"[ARENA] Level {lvl} >= {TARGET_LEVEL_RESET}. Enviando /reset.")
            break

        time.sleep(LEVEL_CHECK_EVERY_S)

    ok_lor = send_reset_and_wait_lorencia(bot, wait_s=12.0)
    if ok_lor:
        log("[OK] Voltou para Lorencia após reset. Reiniciando loop.")
    else:
        log("[WARN] Não confirmei Lorencia após reset (talvez delay maior).")

# =========================
# MAIN LOOP (REGRA NOVA)
# =========================
def main_loop():
    maestro = BotMaestroSDK.from_sys_args()
    execution = maestro.get_execution()
    log(f"Task ID is: {execution.task_id}")
    log(f"Task Parameters are: {execution.parameters}")

    bot = DesktopBot()

    while True:
        # Garante Lorencia
        if not ensure_lorencia(bot):
            log("[FLOW] Não detectei Lorencia. Tentando novamente em 5s...")
            time.sleep(5)
            continue

        # Lê level
        level = read_level_with_c()
        log(f"[LEVEL] inicial: {level}")

        # Se não leu, re-tenta (evita jogar fora o fluxo)
        if level is None:
            log("[FLOW] não consegui ler level -> tentando novamente em 10s...")
            time.sleep(10)
            continue

        # >>> REGRA NOVA:
        # 1) Se >1 e <100 => PULA JIN e vai CEMITÉRIO
        # 2) Se ==1 => DISTRIBUI PONTOS e depois JIN
        # 3) Se >=100 => MOVE ARENA
        if 1 < level < 100:
            log("[FLOW] 1 < level < 100 -> pular Jin -> ir cemitério -> upar até 100 -> arena.")
            go_cemiterio_and_up_until_100_then_move_arena(bot)

        elif level == 1:
            log("[FLOW] level == 1 -> distribuição + Jin (depois segue o ciclo normal).")
            jin_interaction()

            # Depois do Jin, normalmente você vai upar (cemitério), então mantém o mesmo fluxo:
            go_cemiterio_and_up_until_100_then_move_arena(bot)

        else:  # level >= 100
            log("[FLOW] level >= 100 -> /move arena direto.")
            ok_arena = send_move_arena_and_verify(bot)
            if not ok_arena:
                log("[FLOW] não confirmou arena -> tentando novamente em 5s.")
                time.sleep(5)

        # Se entrou em Arena, roda o fluxo completo
        if ensure_arena(bot):
            arena_flow(bot)
        else:
            log("[FLOW] não estou em Arena, voltando loop em 5s...")
            time.sleep(5)

# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        log("\nEncerrado pelo usuário (Ctrl+C).")
