from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
import urllib.request as urllib2
from pydantic import BaseModel
from enum import Enum
import traceback
import uvicorn
import uuid
import sys
import os
import re

from inference import run_inference


# The root is the absolute path of the __init_.py under the source
ROOT = os.path.abspath(__file__)[:os.path.abspath(__file__).rfind(os.path.sep)]
ROOT_DOWNLOAD_URL = os.path.join(ROOT, ".data_cache")

app = FastAPI(title="Custom Model Inference")

# load models here
model1 = "facenet"
model2 = "mobilenetv3_large"
models_list = [model1, model2]


class InputModel(BaseModel):
    threshold: float = 0.55
    model_name: str
    image_file_path: str


class model_name(str, Enum):
    facenet = model1
    mobilenetv3_large = model2


def remove_file(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


def download_url_file(download_url: str, download_path: str) -> None:
    response = urllib2.urlopen(download_url)
    with open(download_path, 'wb') as f:
        f.write(response.read())


async def cache_file(name: str, data: bytes) -> str:
    image_file_path = os.path.join(ROOT_DOWNLOAD_URL, name)
    print(f"Caching image in server at {image_file_path}")
    os.makedirs(ROOT_DOWNLOAD_URL, exist_ok=True)
    with open(image_file_path, 'wb') as img_file_ptr:
        img_file_ptr.write(data)
    return image_file_path


class InferenceProcessTask():
    def __init__(self, func, input_data):
        super(InferenceProcessTask, self).__init__()
        self.func = func
        self.input_data = input_data
        self.response_data = dict()

    def run(self):
        # check if the input image_file_path is an existing path or a url
        is_local_dir = os.path.exists(
            self.input_data.image_file_path)
        is_url = re.match(
            r'^https?:/{2}\w.+$', self.input_data.image_file_path)

        if is_local_dir:
            input_image_path = self.input_data.image_file_path
        elif is_url:
            try:
                os.makedirs(ROOT_DOWNLOAD_URL, exist_ok=True)
                input_image_path = os.path.join(
                    ROOT_DOWNLOAD_URL, str(uuid.uuid4()) + '.jpg')
                download_url_file(
                    self.input_data.image_file_path, input_image_path)
            except Exception as e:
                print(e)
                self.response_data["code"] = "failed"
                self.response_data['msg'] = "Can not download image from \'%s\'. Not a valid link." % (
                    self.input_data.image_file_path)
                return

        # run the inference function
        self.results = self.func(
            input_image_path, self.input_data.model_name, self.input_data.threshold)
        self.response_data["code"] = "success"
        self.response_data['msg'] = "Prediction successful"
        self.response_data["prediction"] = self.results


@app.post("/inference_model_file/{inputModel}_model")
async def inference_model_file(input_model: model_name,
                               background_tasks: BackgroundTasks,
                               file: UploadFile = File(...),
                               display_image_only: bool = Form(False),
                               threshold: float = Form(0.55)):
    response_data = dict()
    image_file_path = ""
    try:
        # Save this image to the temp file
        file_name = str(uuid.uuid4()) + '.jpg'
        file_bytes_content = file.file.read()
        image_file_path = await cache_file(file_name, file_bytes_content)
        # add cached file removal to list of bg tasks to exec after sending response
        background_tasks.add_task(remove_file, image_file_path)

        input_data = InputModel(
            model_name=input_model.value,
            image_file_path=image_file_path,
            threshold=threshold)
        task = InferenceProcessTask(run_inference,
                                    input_data=input_data)
        task.run()
        response_data = task.response_data
        if display_image_only:
            return FileResponse(path=image_file_path,
                                media_type=file.content_type)
    except Exception as e:
        print(e)
        print(traceback.print_exc())
        response_data["code"] = "failed"
        response_data["msg"] = "failed to run inference on file"

    return response_data


@app.post("/inference_model_url/{inputModel}_model")
async def inference_model_url(*, input_model: model_name,
                              input_data: InputModel,
                              background_tasks: BackgroundTasks):
    response_data = dict()
    try:
        task = InferenceProcessTask(run_inference, input_data=input_data)
        task.run()
        response_data = task.response_data
    except Exception as e:
        print(e)
        print(traceback.print_exc())
        response_data["code"] = "failed"
        response_data["msg"] = "failed to run inference on file from url"

    return response_data


@app.get("/")
def index():
    return {"Welcome to Model Inference Server Web Service": "Please visit /docs"}


if __name__ == '__main__':
    if len(sys.argv) == 1:    # if port is not specified
        print("Using default port: 8080")
        uvicorn.run(app, host='0.0.0.0', port=8080, workers=1)
    elif len(sys.argv) == 2:  # port specified
        print("Using port: " + sys.argv[1])
        uvicorn.run(app, host='0.0.0.0', port=int(sys.argv[1]), workers=1)
