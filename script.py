import pyautogui, time
print("Coloque o mouse no CANTO SUPERIOR ESQUERDO do texto das coords e espere 5s...")
time.sleep(5)
x1, y1 = pyautogui.position()
print("Agora coloque o mouse no CANTO INFERIOR DIREITO do texto das coords e espere 5s...")
time.sleep(5)
x2, y2 = pyautogui.position()
print("HUD_BOX =", (x1, y1, x2-x1, y2-y1))