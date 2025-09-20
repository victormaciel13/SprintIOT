import cv2, numpy as np, pickle, os, time

# ---- Config ----
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
DB_LABELS = "labels.pkl"       # mapeia id->nome
DB_MODEL  = "lbph_model.yml"   # modelo treinado
SAMPLES_ENROLL = 12            # nº de amostras por cadastro
FACE_SIZE = (160, 160)         # tamanho do recorte
DETECT_SCALE = 1.2
DETECT_MIN_NEI = 5
COOLDOWN = 3.0                 # seg. entre sinais seriais

# ---- Serial (opcional) ----
ser = None
try:
    ser = serial.Serial(PORT, BAUD, timeout=0.5)
    time.sleep(2)
except Exception as e:
    print(f"[AVISO] Serial indisponível ({e}). Seguindo sem Arduino.")

# ---- Detector/Reconhecedor ----
detector = cv2.CascadeClassifier(CASCADE_PATH)
recognizer = cv2.face.LBPHFaceRecognizer_create()

labels, next_id = {}, 0
if os.path.exists(DB_LABELS):
    labels = pickle.load(open(DB_LABELS, "rb"))
    next_id = max(labels.keys(), default=-1) + 1

if os.path.exists(DB_MODEL):
    try:
        recognizer.read(DB_MODEL)
        print("[INFO] Modelo LBPH carregado.")
    except Exception as e:
        print(f"[AVISO] Falha ao carregar modelo: {e}")

def detect_faces_gray(gray):
    return detector.detectMultiScale(
        gray, scaleFactor=DETECT_SCALE, minNeighbors=DETECT_MIN_NEI, minSize=(80, 80))

def preprocess_face(gray, x, y, w, h):
    roi = gray[y:y+h, x:x+w]
    roi = cv2.resize(roi, FACE_SIZE)
    return roi

def train_or_update(recognizer, train_imgs, train_ids):
    if len(train_imgs) == 0:
        return
    try:
        if os.path.exists(DB_MODEL):
            recognizer.update(train_imgs, np.array(train_ids))
        else:
            recognizer.train(train_imgs, np.array(train_ids))
        recognizer.write(DB_MODEL)
        print("[OK] Modelo salvo:", DB_MODEL)
    except cv2.error:
        recognizer.train(train_imgs, np.array(train_ids))
        recognizer.write(DB_MODEL)

def enroll(cap, gray, face):
    global next_id, labels
    x, y, w, h = face
    nome = input("Nome: ").strip()
    if not nome:
        return
    if nome in labels.values():
        inv = {v:k for k,v in labels.items()}
        person_id = inv[nome]
    else:
        person_id = next_id
        labels[person_id] = nome
        next_id += 1

    print(f"[CADASTRO] Capturando {SAMPLES_ENROLL} amostras de {nome}...")
    imgs, ids = [], []
    collected = 0
    start = time.time()
    while collected < SAMPLES_ENROLL:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces_gray(g)
        if len(faces) != 1:
            cv2.putText(frame, "Precisa 1 rosto visivel", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv2.imshow("Faces", frame); cv2.waitKey(1)
            if time.time()-start > 8:
                break
            continue
        (x2,y2,w2,h2) = faces[0]
        roi = preprocess_face(g, x2,y2,w2,h2)
        imgs.append(roi)
        ids.append(person_id)
        collected += 1

        cv2.rectangle(frame, (x2,y2), (x2+w2,y2+h2), (0,255,255), 2)
        cv2.putText(frame, f"Amostra {collected}/{SAMPLES_ENROLL}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.imshow("Faces", frame); cv2.waitKey(1)

    if imgs:
        train_or_update(recognizer, imgs, ids)
        pickle.dump(labels, open(DB_LABELS, "wb"))
        print(f"[OK] Salvo cadastro/atualizacao de: {nome}")

# ---- Loop principal ----
cap = cv2.VideoCapture(0)
validando = False
ultimo_envio = 0.0

print("[E]=Cadastrar  [V]=Validar ON/OFF  [Q]=Sair")

while True:
    ok, frame = cap.read()
    if not ok: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces_gray(gray)

    for (x,y,w,h) in faces:
        roi = preprocess_face(gray, x,y,w,h)
        nome = "Desconhecido"

        if validando and os.path.exists(DB_MODEL) and len(labels) > 0:
            try:
                label_id, conf = recognizer.predict(roi)
                if label_id in labels:
                    nome = labels[label_id]
            except cv2.error:
                pass

        color = (0,255,0) if nome != "Desconhecido" else (0,0,255)
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame, nome, (x, max(20, y-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if ser and nome != "Desconhecido" and (time.time()-ultimo_envio) > COOLDOWN:
            try:
                ser.write(b'O')
                ultimo_envio = time.time()
            except Exception as e:
                print(f"[AVISO] Serial falhou: {e}")

    cv2.imshow("Faces", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'): break
    if k == ord('v'): validando = not validando
    if k == ord('e') and len(faces) == 1:
        enroll(cap, gray, faces[0])

cap.release()
cv2.destroyAllWindows()
if ser:
    try: ser.close()
    except: pass