from roboflow import Roboflow

rf = Roboflow(api_key="71SjhPOPiAGvQLxirrwi")
project = rf.workspace().project("escape-frisbee-game-3xwgq")
model = project.version("1").model

job_id, signed_url, expire_time = model.predict_video(
    "/Users/jameselsner/Desktop/Projects/ddc-vision/Videos/Augie_Highlight_Volley.mp4",
    fps=5,
    prediction_type="batch-video",
)

results = model.poll_until_video_results(job_id)

print(results)
