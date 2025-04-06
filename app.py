from model import DigitCNN
from PIL import Image, ImageOps
import datetime
import numpy as np
import os
import pandas as pd
import psycopg2
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn.functional as F

def get_connection():
    return psycopg2.connect(
        host=os.environ["DB_HOST"],
        port=os.environ["DB_PORT"],
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"]
    )

def log_prediction(timestamp, pred, confidence, true_label):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO predictions (timestamp, predicted, confidence, true_label) VALUES (%s, %s, %s, %s)",
        (timestamp, pred, confidence, true_label)
    )
    conn.commit()
    cur.close()
    conn.close()

def get_prediction_logs(limit=50):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT timestamp, predicted, confidence, true_label FROM predictions ORDER BY timestamp DESC LIMIT %s;",
        (limit,)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

# Load model
model = DigitCNN()
model.load_state_dict(torch.load("models/mnist_cnn.pth", map_location=torch.device('cpu')))
model.eval()

st.title("🧠 Digit Recognizer")

# Drawing canvas
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=12,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data[:, :, 0:3]
    img = Image.fromarray((img * 255).astype(np.uint8)).convert('L')
    img = ImageOps.invert(img).resize((28, 28))
    img_tensor = torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

    true_label = st.number_input("True label:", min_value=0, max_value=9, step=1)

    if st.button("Submit"):
        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            pred = torch.argmax(probs).item()
            conf = torch.max(probs).item()

        st.write(f"**Prediction:** {pred}")
        st.write(f"**Confidence:** {int(conf*100)}%")

        log_prediction(str(datetime.datetime.now()), pred, conf, int(true_label))

# History from database
st.subheader("History (Prediction Logs)")
logs = get_prediction_logs()
if logs:
    df = pd.DataFrame(logs, columns=["Timestamp", "Predicted", "Confidence", "True Label"])
    st.dataframe(df)
else:
    st.info("No predictions logged yet.")
