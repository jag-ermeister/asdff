!pip install git+https://github.com/jag-ermeister/asdff.git@sdxl
!pip install ultralytics
!pip uninstall asdff -y
- Restart Kernel after reinstalling


```
from asdff import __version__


from functools import partial
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, DPMSolverMultistepScheduler
from asdff import AdStableDiffusionXlPipeline, yolo_detector
from huggingface_hub import hf_hub_download
from PIL import Image
from datetime import datetime
from diffusers.utils import load_image

pipe = AdStableDiffusionXlPipeline.from_single_file("24GB_Best.safetensors", torch_dtype=torch.float16)
pipe.safety_checker = None
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "portrait of ((ohwx woman))"
common = {"prompt": prompt, "num_inference_steps": 30}
inpaint_only = {"prompt": prompt, "num_inference_steps": 30, "strength": 0.7}

person_model_path = hf_hub_download("Bingsu/adetailer", "face_yolov8n.pt")
person_detector = partial(yolo_detector, model_path=person_model_path)

image_to_detail = load_image('inference_results/im_20240107183247_000_1024665048.png')

common = {
    "prompt": "portrait of ((ohwx woman))", 
    "num_inference_steps": 28,
}
result = pipe(common=common, images=[image_to_detail], detectors=[person_detector, pipe.default_detector])
image = result[0][0]
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
filename = f"ad_{timestamp}.jpg"
image.save(filename)

```


Optionally use this to download a model I trained (ohwx man):
!pip install boto3
```
import boto3

boto3.setup_default_session(
    aws_access_key_id='YOUR_ACCESS_KEY',
    aws_secret_access_key='YOUR_SECRET_KEY',
    region_name='YOUR_REGION'
)

s3_client = boto3.client('s3')

bucket_name = 'photo-packs-jag-order-images-dev'
s3_file_key = "<model_id>/weights/24GB_Best.safetensors"
local_file_path = '24GB_Best.safetensors'

s3_client.download_file(bucket_name, s3_file_key, local_file_path)
```