from botcity.core import DesktopBot
from botcity.maestro import *
import pyautogui
import time
import re
from collections import Counter
import random
import math

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

# Caminho por checkpoints até o cemitério (Lorencia)
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

# Botão a clicar ao chegar no spot futuro (HUD)
BUTTON_BOX = (223, 60, 26, 28)

# Tempo máximo total para chegar no cemitério (2 minutos)
FUTURO_TIMEOUT_S = 120

# Verificar level a cada 1 minuto até >= 100
LEVEL_CHECK_EVERY_S = 60
TARGET_LEVEL = 100

# Move arena
MOVE_CMD = "/move arena"
CHAT_OPEN_WAIT_S = 5
CHAT_AFTER_TYPE_WAIT_S = 5
AFTER_MOVE_WAIT_S = 10

# Movimento
MAX_STEPS = 320
MOVE_WAIT = 0.35
CENTER_Y_RATIO = 0.55
LINEAR_CLICK_SCALE = 0.62
LINEAR_CLICK_JITTER = 36
LINEAR_CLOSE_JITTER = 22
AXIS_LOCK_CROSS_JITTER = 3
RESCUE_CLICK_SCALE = 0.45
TURN_PENALTY_DEG = 18

# Limites coordenadas
X_MIN, X_MAX = 0, 255
Y_MIN, Y_MAX = 0, 255

# OCR anti-salto
MAX_JUMP = 30
_last_good = None

# Direções (mapeadas para seu client)
DIR_CLICKS = {
    "N":  (0,   200),
    "S":  (0,  -200),
    "E":  (240,   0),
    "W":  (-240,  0),
    "NE": (220,  80),
    "NW": (-220, 80),
    "SE": (220, -80),
    "SW": (-220, -80),
}

DIRECTION_VECTORS = {
    "N":  (0, -1),
    "S":  (0, 1),
    "E":  (1, 0),
    "W":  (-1, 0),
    "NE": (1, -1),
    "NW": (-1, -1),
    "SE": (1, 1),
    "SW": (-1, 1),
}

# “Nudges” horizontais (mais suaves) para corrigir X sem sair muito do caminho
NUDGE_E = (140, 0)
NUDGE_W = (-140, 0)

QUAD_PREFS = {
    "NW": ["NW", "W", "N", "SW", "NE"],
    "NE": ["NE", "E", "N", "SE", "NW"],
    "SW": ["SW", "W", "S", "NW", "SE"],
    "SE": ["SE", "E", "S", "NE", "SW"],
    "W":  ["W", "NW", "SW", "N", "S"],
    "E":  ["E", "NE", "SE", "N", "S"],
    "N":  ["N", "NW", "NE", "W", "E"],
    "S":  ["S", "SW", "SE", "W", "E"],
}

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
def click_relative_safe(dx: int, dy: int):
    w, h = pyautogui.size()
    cx = w // 2
    cy = int(h * CENTER_Y_RATIO)
    pyautogui.click(cx + dx, cy + dy)

def click_box_center(box):
    x, y, w, h = box
    px = x + (w // 2)
    py = y + (h // 2)
    pyautogui.click(px, py)

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

def get_current_xy_filtered(hud_box, samples=6, delay=0.08) -> tuple[int, int] | None:
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

def get_current_value_filtered(hud_box, samples=5, delay=0.10) -> int | None:
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
# MOVIMENTO
# =========================
def dist(cur: tuple[int, int], target: tuple[int, int]) -> float:
    cx, cy = cur
    tx, ty = target
    return math.hypot(tx - cx, ty - cy)

def primary_quadrant_xy(cur, target, tol_x, tol_y) -> str | None:
    cx, cy = cur
    tx, ty = target
    dx = tx - cx
    dy = ty - cy

    if abs(dx) <= tol_x and abs(dy) <= tol_y:
        return None

    if dx < -tol_x and dy < -tol_y: return "NW"
    if dx >  tol_x and dy < -tol_y: return "NE"
    if dx < -tol_x and dy >  tol_y: return "SW"
    if dx >  tol_x and dy >  tol_y: return "SE"
    if dx < -tol_x: return "W"
    if dx >  tol_x: return "E"
    if dy < -tol_y: return "N"
    if dy >  tol_y: return "S"
    return None

def click_any_direction(strength=RESCUE_CLICK_SCALE):
    name = random.choice(list(DIR_CLICKS.keys()))
    click_direction_linear(name, strength=strength, jitter=LINEAR_CLICK_JITTER)
    time.sleep(0.45)

def log_navigation_profile():
    log(
        "[NAV] perfil linear ativo -> "
        f"scale={LINEAR_CLICK_SCALE}, jitter={LINEAR_CLICK_JITTER}, close_jitter={LINEAR_CLOSE_JITTER}, "
        f"axis_cross_jitter={AXIS_LOCK_CROSS_JITTER}, rescue_scale={RESCUE_CLICK_SCALE}, "
        f"turn_penalty_deg={TURN_PENALTY_DEG}"
    )

def click_direction_linear(direction: str, strength=LINEAR_CLICK_SCALE, jitter=LINEAR_CLICK_JITTER, axis_lock: str | None = None):
    """
    Clica em um quadrado pequeno à frente do personagem para reduzir trajetos em arco.
    Isso deixa o movimento mais linear e com menos risco de bater em obstáculo lateral.
    """
    dx_base, dy_base = DIR_CLICKS[direction]

    if axis_lock == "X":
        # Em linha reta no eixo X: quase sem variação vertical.
        dx = int(dx_base * strength) + random.randint(-jitter, jitter)
        dy = int(dy_base * strength) + random.randint(-AXIS_LOCK_CROSS_JITTER, AXIS_LOCK_CROSS_JITTER)
    elif axis_lock == "Y":
        # Em linha reta no eixo Y: quase sem variação horizontal.
        dx = int(dx_base * strength) + random.randint(-AXIS_LOCK_CROSS_JITTER, AXIS_LOCK_CROSS_JITTER)
        dy = int(dy_base * strength) + random.randint(-jitter, jitter)
    else:
        dx = int(dx_base * strength) + random.randint(-jitter, jitter)
        dy = int(dy_base * strength) + random.randint(-jitter, jitter)

    lock_msg = f" axis_lock={axis_lock}" if axis_lock else ""
    log(f"[DIR] {direction} -> click linear ({dx},{dy}){lock_msg}")
    click_relative_safe(dx, dy)

def choose_linear_direction(dx: int, dy: int, last_direction: str | None = None) -> str:
    """Escolhe a direção que mais reduz a distância angular até o alvo, penalizando curvas bruscas."""
    target_len = math.hypot(dx, dy)
    if target_len == 0:
        return "E"

    best_direction = "E"
    best_score = float("inf")

    for direction, (vx, vy) in DIRECTION_VECTORS.items():
        vec_len = math.hypot(vx, vy)
        cos_theta = ((dx * vx) + (dy * vy)) / (target_len * vec_len)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        score = math.degrees(math.acos(cos_theta))

        if last_direction and direction != last_direction:
            score += TURN_PENALTY_DEG

        if score < best_score:
            best_score = score
            best_direction = direction

    return best_direction

def nudge_to_fix_x(cur, target_x):
    """Pequeno ajuste lateral para forçar X ir pro valor exato (ex: manter em 134)."""
    cx, _ = cur
    if cx < target_x:
        dx, dy = NUDGE_E
        log(f"[NUDGE] corrigindo X: {cx} -> {target_x} (E {dx},{dy})")
        click_relative_safe(dx, dy)
    elif cx > target_x:
        dx, dy = NUDGE_W
        log(f"[NUDGE] corrigindo X: {cx} -> {target_x} (W {dx},{dy})")
        click_relative_safe(dx, dy)
    time.sleep(MOVE_WAIT)

def walk_to(hud_box, target_xy: tuple[int, int], label="ALVO",
            tol=3, timeout_s=35, tol_x=None, tol_y=None, enforce_x=False) -> bool:
    """
    tol_x/tol_y: tolerâncias separadas.
    enforce_x=True: se estiver fora do X, PRIORIZA corrigir X (com nudges) antes de andar no Y.
    """
    if tol_x is None: tol_x = tol
    if tol_y is None: tol_y = tol

    log(f"Indo até {label} {target_xy} (tol_x={tol_x}, tol_y={tol_y}, timeout={timeout_s}s)...")
    start_t = time.time()

    last_xy = None
    stuck_count = 0
    ocr_fail = 0

    last_dist = None
    worse_streak = 0
    last_direction = None

    for step in range(1, MAX_STEPS + 1):
        if time.time() - start_t > timeout_s:
            log(f"[TIMEOUT] walk_to {label} excedeu {timeout_s}s")
            return False

        cur = get_current_xy_filtered(hud_box, samples=6, delay=0.08)
        if not cur:
            ocr_fail += 1
            log(f"[WARN] OCR não leu coordenadas... ({ocr_fail})")
            if ocr_fail >= 3:
                log("[ACTION] OCR falhando -> emergência")
                click_any_direction()
                ocr_fail = 0
            time.sleep(0.25)
            continue

        ocr_fail = 0
        d = dist(cur, target_xy)
        cx, cy = cur
        tx, ty = target_xy
        dx = tx - cx
        dy = ty - cy

        log(f"[step {step}/{MAX_STEPS}] [XY] atual=({cx},{cy}) alvo=({tx},{ty}) dist={d:.2f} (dx={dx}, dy={dy})")

        if abs(dx) <= tol_x and abs(dy) <= tol_y:
            log(f"Chegou em {label} (<=tol_x={tol_x}, tol_y={tol_y}).")
            return True

        # detecta stuck
        if last_xy is not None and cur == last_xy:
            stuck_count += 1
        else:
            stuck_count = 0
        last_xy = cur

        # se travou, tenta resolver de forma "inteligente" (corrigir X primeiro se necessário)
        if stuck_count >= 3:
            log(f"[STUCK] {stuck_count}x -> rescue")
            if enforce_x and abs(dx) > 0:
                nudge_to_fix_x(cur, tx)
            else:
                click_any_direction()
            stuck_count = 0
            continue

        # Se a ideia é "manter em X=134", priorize corrigir X mesmo com dy grande
        if enforce_x and abs(dx) > tol_x:
            nudge_to_fix_x(cur, tx)
            continue

        if last_dist is not None and d > last_dist + 0.5:
            worse_streak += 1
        else:
            worse_streak = 0
        last_dist = d

        quad = primary_quadrant_xy(cur, target_xy, tol_x=tol_x, tol_y=tol_y)
        if quad is None:
            time.sleep(0.18)
            continue

        axis_lock = None
        if abs(dy) <= tol_y and abs(dx) > tol_x:
            # Se já está no Y do destino, anda reto apenas no X (linha horizontal curta).
            direction = "E" if dx > 0 else "W"
            axis_lock = "X"
        elif abs(dx) <= tol_x and abs(dy) > tol_y:
            # Se já está no X do destino, anda reto apenas no Y (linha vertical curta).
            direction = "S" if dy > 0 else "N"
            axis_lock = "Y"
        else:
            direction = choose_linear_direction(dx, dy, last_direction=last_direction)
            if worse_streak >= 2 and direction == last_direction and quad in QUAD_PREFS:
                # fallback: tenta segunda melhor opção do quadrante para destravar sem perder linearidade.
                direction = QUAD_PREFS[quad][1]
                log(f"[ADAPT] piorando {worse_streak}x -> fallback {direction}")

        dynamic_jitter = LINEAR_CLICK_JITTER
        if abs(dx) <= 6 or abs(dy) <= 6:
            dynamic_jitter = LINEAR_CLOSE_JITTER

        click_direction_linear(direction, jitter=dynamic_jitter, axis_lock=axis_lock)
        last_direction = direction
        time.sleep(MOVE_WAIT)

    not_found(f"não chegou em {label}")
    return False

def go_exact(hud_box, xy: tuple[int,int], label: str) -> bool:
    if not walk_to(hud_box, xy, label=f"{label}_FAST", tol=3, timeout_s=25):
        return False
    if walk_to(hud_box, xy, label=f"{label}_TIGHT", tol=1, timeout_s=12):
        return True
    log(f"[EXACT] não conseguiu tol=1 em {label}, mas está próximo. Continuando mesmo assim.")
    return True

# =========================
# ROTA CHECKPOINTS (CEMITÉRIO) - FIXA X=134 nos CPs que pedem isso
# =========================
def walk_route_with_checkpoints(hud_box, checkpoints, total_timeout_s=120) -> bool:
    """
    Rota com timeout GLOBAL.
    1) tenta a rota mais curta (reta) até o último checkpoint.
    2) se falhar por obstáculo, cai para checkpoints com retry/rescue.
    """
    log(f"[ROUTE] Iniciando rota com {len(checkpoints)} checkpoints. Timeout total={total_timeout_s}s")
    start = time.time()

    final_target = checkpoints[-1]
    direct_timeout = max(18, int(total_timeout_s * 0.35))
    log(f"[ROUTE][SHORT] tentando rota direta até {final_target} (timeout={direct_timeout}s)")
    if walk_to(hud_box, final_target, label="DIRECT_FINAL", timeout_s=direct_timeout, tol_x=2, tol_y=2):
        log("[ROUTE][SHORT] rota direta concluída com sucesso.")
        return True

    log("[ROUTE][SHORT] rota direta falhou -> fallback para checkpoints.")

    for idx, cp in enumerate(checkpoints, 1):
        remaining = total_timeout_s - (time.time() - start)
        if remaining <= 0:
            log("[ROUTE][TIMEOUT] acabou o tempo total da rota")
            return False

        cur = get_current_xy_filtered(hud_box, samples=6, delay=0.08)
        d = dist(cur, cp) if cur else 60.0
        wanted = max(14, int(d * 0.45))
        slice_timeout = min(int(remaining * 0.90), wanted)
        slice_timeout = max(14, slice_timeout)

        # >>> REGRA: se checkpoint quer X=134, NÃO ACEITA “132 dentro do tol=3”
        enforce_x = (cp[0] == 134)
        tol_x = 0 if enforce_x else 3
        tol_y = 3

        log(f"[ROUTE] Checkpoint {idx}/{len(checkpoints)} -> {cp} dist~{d:.2f} timeout={slice_timeout}s enforce_x={enforce_x}")

        ok = walk_to(
            hud_box, cp,
            label=f"CP{idx}",
            timeout_s=slice_timeout,
            tol_x=tol_x, tol_y=tol_y,
            enforce_x=enforce_x
        )
        if ok:
            continue

        log(f"[ROUTE][RETRY] Falhou CP{idx}. Rescue e retry...")
        # rescue mais direcionado: se cp.x==134, tenta “puxar” X pra 134 antes
        if enforce_x and cur:
            nudge_to_fix_x(cur, cp[0])
        click_any_direction(strength=0.55)
        click_any_direction(strength=0.55)

        remaining2 = total_timeout_s - (time.time() - start)
        if remaining2 <= 0:
            log("[ROUTE][TIMEOUT] sem tempo para retry")
            return False

        retry_timeout = min(35, int(remaining2 * 0.55))
        retry_timeout = max(14, retry_timeout)

        ok2 = walk_to(
            hud_box, cp,
            label=f"CP{idx}_RETRY",
            timeout_s=retry_timeout,
            tol_x=tol_x, tol_y=tol_y,
            enforce_x=enforce_x
        )
        if not ok2:
            log(f"[ROUTE] Falhou no checkpoint {idx}: {cp}")
            return False

    log("[ROUTE] Rota concluída com sucesso.")
    return True

# =========================
# CLIQUE NO JIN (OFFSETS)
# =========================
def click_jin_by_offsets(hud_box, base_xy=(135,126)) -> bool:
    if not go_exact(hud_box, base_xy, "BASE_JIN"):
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
        walk_to(hud_box, base_xy, label="RET_BASE", tol=3, timeout_s=12)

        before = get_current_xy_filtered(hud_box, samples=6, delay=0.08)
        if not before:
            log("[JIN] OCR falhou antes do clique, tentando próximo...")
            continue

        log(f"[JIN] tentativa {i}/{len(offsets)} click({dx},{dy}) from={before}")
        click_relative_safe(dx, dy)
        time.sleep(0.60)

        after = get_current_xy_filtered(hud_box, samples=6, delay=0.08)
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
# CHAT COMMAND + VALIDACAO HUD
# =========================
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
            log("[VAL] OCR do valor falhou em alguma leitura -> tentando novamente")
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
    log(f"[UP] Indo para CEMITÉRIO via checkpoints (timeout total {FUTURO_TIMEOUT_S}s)...")

    ok = walk_route_with_checkpoints(
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

    # 1) Verificar Lorencia primeiro
    if not ensure_lorencia(bot):
        return

    log(f"HUD_BOX = {HUD_BOX}")
    log_navigation_profile()

    # 2) Ler level
    level = get_level_filtered(samples=7, delay=0.08)
    log(f"[LEVEL] level filtrado (início): {level}")

    # 3) Lógica:
    if level is not None and 1 < level < 100:
        log("[FLOW] level > 1 e < 100 -> ir para cemitério (checkpoints) e upar até 100 -> arena.")
        go_cemiterio_and_up_until_100_then_move_arena(bot)
        return

    if level is None:
        log("[FLOW] Não consegui ler level -> seguindo para Jin.")
    elif level >= 100:
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
    else:
        log("[FLOW] level == 1 -> verificar pontos antes de distribuir.")

    # =========================
    # FLUXO LEVEL == 1 => VERIFICA PONTOS
    # =========================
    valor = read_points_from_c(HUD_BOX_VALOR)
    if valor is None:
        log("[WARN] Não foi possível ler o valor/pontos. Indo cemitério por segurança.")
        go_cemiterio_and_up_until_100_then_move_arena(bot)
        return

    log(f"[PONTOS] valor lido: {valor}")

    if valor == 0:
        log("[FLOW] level==1 e pontos==0 -> pular Jin/distribuição e ir direto pro cemitério.")
        go_cemiterio_and_up_until_100_then_move_arena(bot)
        return

    # =========================
    # FLUXO JIN + DISTRIBUIR
    # =========================
    TARGET_FONTE = (136, 127)
    if not walk_to(HUD_BOX, TARGET_FONTE, label="FONTE", tol=3, timeout_s=35):
        return

    BASE_JIN = (135, 126)
    if not click_jin_by_offsets(HUD_BOX, base_xy=BASE_JIN):
        not_found("Clique no Jin por offsets falhou")
        return

    log("Jin clicado (provável) por offsets.")

    # recalcula pontos (garante leitura pós-Jin)
    valor = read_points_from_c(HUD_BOX_VALOR)
    if valor is None:
        log("[WARN] Não foi possível ler pontos após Jin. Indo cemitério.")
        go_cemiterio_and_up_until_100_then_move_arena(bot)
        return

    log(f"[PONTOS] valor lido pós-Jin: {valor}")

    if valor == 0:
        log("[FLOW] Pós-Jin: pontos==0 -> pular distribuição e ir cemitério.")
        go_cemiterio_and_up_until_100_then_move_arena(bot)
        return

    forca = int(round(valor * 0.55))
    agilidade = int(round(valor * 0.25))
    vitalidade = int(round(valor * 0.15))
    energia = int(round(valor * 0.05))

    log(f"Calculado -> Força: {forca}, Agilidade: {agilidade}, Vitalidade: {vitalidade}, Energia: {energia}")

    cmds = [('/f', forca), ('/a', agilidade), ('/v', vitalidade), ('/e', energia)]
    for cmd, val in cmds:
        ok = apply_stat_with_validation(
            cmd, val, HUD_BOX_VALOR,
            retries=2,
            open_delay=0.35,
            type_delay=0.15,
            send_delay=0.35
        )
        if not ok:
            log(f"[WARN] Não consegui confirmar {cmd}. Continuando mesmo assim...")

    final_points = read_points_from_c(HUD_BOX_VALOR)
    log(f"[FINAL] pontos após aplicar tudo: {final_points}")

    # Depois de distribuir, vai cemitério e upa até 100
    log("[FLOW] Distribuição concluída -> ir cemitério (checkpoints) e upar até 100 -> arena.")
    go_cemiterio_and_up_until_100_then_move_arena(bot)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\nEncerrado pelo usuário (Ctrl+C).")
