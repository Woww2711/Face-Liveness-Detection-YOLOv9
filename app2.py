import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
# import torch
# from streamlit_webrtc import webrtc_streamer
# import av

modelPath = 'yolov9_openvino_model_best/'

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
        webcam = st.checkbox('Webcam')
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
        model = YOLO(modelPath, task='detect')
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
                    width=480
                    )
        
    if st.sidebar.button('Detect Image'):
        if source_img:
            res = model.predict(uploaded_image,
                                conf=confidence,
                                line_width=1, 
                                show_labels=False, 
                                show_conf=False
                                )
            res_plotted = res[0].plot(labels=True, line_width=int(image_width/250))[:, :, ::-1]
            with col2:
                st.image(res_plotted,
                        caption='Detected Image',
                        width=480              
                        )
                for box in res[0].boxes:
                    class_id = int(box.cls)
                    class_label = res[0].names[class_id]
                    st.caption(f'Detected class: {class_label}')        
    
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
            holder1 = st.empty()
            holder2 = st.empty()
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                results = model.predict(frame,conf=confidence)
                annotated_frame = results[0].plot()
                holder1.image(annotated_frame,channels='BGR',width=640)
                for box in results[0].boxes:
                    class_id = int(box.cls)
                    class_label = results[0].names[class_id]
                    holder2.caption(f'Detected class: {class_label}')
    
    #### To run inference locally (frame by frame) ####
    # def frame_process(frame):
    #     frame = frame.to_ndarray(format="bgr24")
    #     results = model.predict(frame,conf=confidence)
    #     annotated_frame = results[0].plot()
    #     return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
    # if webcam:
    #     webrtc_streamer(key="example", 
    #                     video_frame_callback=frame_process,
    #                     media_stream_constraints={"video": True, "audio": False},
    #                     sendback_audio = False)
        
    #### To run inference on app (picture) ####
    if webcam:
        col1, col2 = st.columns(2)
        with col1:
            picture = st.camera_input("Take a picture")
            if picture is not None:
                picture = Image.open(picture)
                results = model.predict(picture,conf=confidence)
                with col2:
                    predicted_picture = results[0].plot()
                    st.image(predicted_picture,channels='BGR',caption="Detected face(s)")
                    for box in results[0].boxes:
                        class_id = int(box.cls)
                        class_label = results[0].names[class_id]
                        st.caption(f'Detected class: {class_label}')

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass

