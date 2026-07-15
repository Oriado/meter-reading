import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.GRU(128 * 4, 256, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512, 11)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
        x, _ = self.rnn(x)
        return self.fc(x).permute(1, 0, 2)


@st.cache_resource
def load_models():
    yolo = YOLO("yolo_pose_best.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN().to(device)
    model.load_state_dict(torch.load("ocr_model.pt", map_location=device))
    model.eval()
    return yolo, model, device


def get_strip(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    win = max(10, h // 7)
    margin = int(h * 0.15)
    best_y, best_val = margin, float("inf")
    for y in range(margin, h - margin - win):
        val = float(gray[y:y + win, :].mean())
        if val < best_val:
            best_val = val
            best_y = y
    pad = win // 3
    return img[max(0, best_y - pad):min(h, best_y + win + pad), :]


def detect(img, yolo):
    results = yolo(img, verbose=False)
    boxes = results[0].boxes
    kps = results[0].keypoints

    if boxes is None or len(boxes) == 0:
        return None, None

    i = int(boxes.conf.cpu().numpy().argmax())
    pts = kps.xy.cpu().numpy()[i]

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    tr = pts[np.argmin(d)]
    br = pts[np.argmax(s)]
    bl = pts[np.argmax(d)]
    corners = np.array([tl, tr, br, bl], dtype=np.float32)

    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))

    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, M, (w, h))

    return results[0].plot(), warped


def read_digits(img, model, device):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    if gray.mean() < 127:
        gray = cv2.bitwise_not(gray)
    pil = Image.fromarray(gray)
    tf = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    tensor = tf(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
    indices = out.argmax(2).squeeze(1)
    prev, result = 10, []
    for idx in indices:
        c = idx.item()
        if c != 10 and c != prev:
            result.append(str(c))
        prev = c
    return "".join(result) if result else "не розпізнано"


yolo, model, device = load_models()

st.title("Розпізнавання показників лічильників")
st.write("Завантажте фото лічильника")

uploaded = st.file_uploader("Оберіть фото", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("Завантажене фото")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    annotated, warped = detect(img, yolo)

    if warped is None:
        st.error("Область з цифрами не знайдена")
    else:
        st.subheader("Знайдена область")
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

        h_w, w_w = warped.shape[:2]
        strip = get_strip(warped) if h_w > w_w * 0.5 else warped

        gray_check = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        if gray_check.std() < 12:
            st.error("Цифри не знайдено")
        else:
            st.subheader("Вирізана область з цифрами")
            st.image(cv2.cvtColor(strip, cv2.COLOR_BGR2RGB))

            st.subheader("Результат розпізнавання")
            st.success(f"**{read_digits(strip, model, device)}**")
