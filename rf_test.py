from inference_sdk import InferenceHTTPClient

# create an inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="wNNtaVXlX79MsMdYXtEc"
)

# run inference on a local image
print(CLIENT.infer(
    "Pictures/test_image.jpg", 
    model_id="ddc-discs-5vdna/7"
))
