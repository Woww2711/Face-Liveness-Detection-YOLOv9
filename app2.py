import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image


modelPath = 'E:/Face Liveness Detector/best13042024.pt'

def main():
    # Page layout
    st.set_page_config(
        page_title="Face Liveness Dectection",  # Setting page title
        layout="wide",      # Setting layout to wide
        initial_sidebar_state="expanded",    # Expanding sidebar by default
    )

    # Sidebar
    with st.sidebar:
        # Adding header to sidebar
        st.header("Config files and parameters")

        # Adding file uploader to sidebar for selecting images
        source_img = st.file_uploader(
            "Upload an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
        source_video = st.file_uploader(
            "Upload a video...", type=('mp4','mov'))
        run = st.checkbox('Webcam')
        st.markdown('---')
        # Model Options
        confidence = float(st.slider(
            "Select Model Confidence", 25, 100, 40)) / 100
        st.markdown('---')

    # Creating main page heading
    st.title("Face Liveness Detection")
    st.header('The application to detect real or spoofed faces 😀👺💻')
    st.caption('Upload a photo or a video by selecting :blue[Browse files]')
    st.caption('Access webcam by selecting :blue[Webcam]')
    st.caption('Select the confidence value for model')
    st.caption('Then click the :blue[Detect] button and check the result.')

    # Check model
    try:
        model = YOLO(modelPath)
    except Exception as ex:
        st.error(
            f"Unable to load model. Check the specified path: {modelPath}")
        st.error(ex)

    if source_img:
        col1, col2 = st.columns(2,gap='large')
        with col1:
            uploaded_image = Image.open(source_img)
            image_width, _ = uploaded_image.size
            st.image(source_img,
                    caption="Uploaded Image",
                    width=image_width
                    )
        
    if st.sidebar.button('Detect Image'):
        if source_img:
            res = model.predict(uploaded_image,
                                conf=confidence,
                                line_width=1, 
                                show_labels=False, 
                                show_conf=False
                                )
            res_plotted = res[0].plot(labels=True, line_width=2)[:, :, ::-1]
            with col2:
                st.image(res_plotted,
                        caption='Detected Image',
                        width=image_width                 
                        )
    
    if source_video is not None:
        col1, col2 = st.columns([0.3,0.7],gap='large')
        vid = source_video.name
        with open(vid, mode='wb') as f:
            f.write(source_video.getbuffer())
        with col1:
            st.video(vid)
    
    if st.sidebar.button('Detect Video'):
        with col2:
            cap = cv2.VideoCapture(vid)
            holder = st.empty()
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                results = model.predict(frame,conf=confidence)
                annotated_frame = results[0].plot()
                holder.image(annotated_frame,channels='BGR',width=cap.get(3))
    
    if run:
        FRAME_WINDOW = st.empty()
        camera = cv2.VideoCapture(0)
        while run:
            _, frame = camera.read()
            results = model.predict(frame,conf=confidence)
            annotated_frame = results[0].plot()
            FRAME_WINDOW.image(annotated_frame,channels='BGR')
         
if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass

