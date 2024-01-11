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

pipe = AdStableDiffusionXlPipeline.from_pretrained("diffusers_model", torch_dtype=torch.float16)
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
    "image": image_to_detail
}
result = pipe(common=common, images=[image_to_detail], detectors=[person_detector, pipe.default_detector])
image = result[0][0]
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
filename = f"ad_{timestamp}.jpg"
image.save(filename)

```