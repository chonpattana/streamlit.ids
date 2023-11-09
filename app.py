import PIL
import streamlit as st
from ultralytics import YOLO

model_path = 'weights/best.pt'

st.set_page_config(
    page_title="Object Detection",
    page_icon=":shark:",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.header("Image Config")
    source_img = st.file_uploader(
        "Upload an image...", type=("jpg", "jpeg", "png", "bmp", "webp"))
    confidence =float(st.slider(
        "Select Model confidence", 25, 100, 40)) / 100

st.title("Object Dection")
st.caption('Updload a photo with this :blue[hand signals]: :+1:, :hand:, :i_love_you_hand_sign:, and :spock-hand:.')
st.caption('Then click the :blue[Detect Objects] button and check the result.')

col1, col2 = st.columns(2)

with col1:
    if source_img:
        uploaded_image = PIL.Image.open(source_img)
        st.image(source_img,
                 caption="Uploaded Image",
                 use_column_width=True
                 )
try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

if st.sidebar.button('Detect Object'):
    res = model.predict(uploaded_image,
                        conf = confidence
                        )
    boxes = res[0].boxes
    res_plotted = res[0].plot()[:, :, ::-1]
    with col2:
        st.image(res_plotted,
                 caption='Decteced Image'
                 )
        try:
            with st.expander("Dection Results"):
                for box in boxes:
                    st.write(box.xywh)
        except Exception as ex:
            st.write("No image is uploaded yet")