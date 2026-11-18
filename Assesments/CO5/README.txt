Assignment 5 - code and results
Environment: python -m venv venv; pip install "opencv-contrib-python-headless==4.10.0.84" numpy psutil scikit-learn matplotlib onnxruntime pytesseract rapidocr-onnxruntime --no-deps(rapidocr) ; TensorFlow-CPU only for code/t5_cnn.py (separate venv).
Data/models NOT included (download separately, paths in code headers): ORL faces, JAFFE, OpenCV sample backgrounds, res10 SSD, insightface buffalo_sc (det_500m.onnx, w600k_mbf.onnx), EfficientNet-B0 emotion ONNX.
Run order: make_gate_video.py, t1_video.py, t1_image_demo.py, t2_preproc.py, t3_data.py, t3_detect.py, t4_recog.py, t5_data.py, t5_cnn.py, t5_main.py, t6_ocr.py, t7_motion.py, t8_video_recog.py, t9_integrated.py, t10_adapt.py
Timings come from a 1-core sandbox; re-run on your PC and update the report numbers.
