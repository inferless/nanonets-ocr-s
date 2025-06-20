import os, io, requests
from PIL import Image
import torch
from typing import Optional
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForImageTextToText,
)
import inferless
from pydantic import BaseModel, Field

@inferless.request
class RequestObjects(BaseModel):
    image_url: str = Field(default="https://example.com/sample.jpg")
    prompt: str = Field(default="""Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes.""")
    temperature: Optional[float] = 0.7
    do_sample: Optional[bool] = False
    max_new_tokens: Optional[int] = 15000

@inferless.response
class ResponseObjects(BaseModel):
    extracted_text: str = Field(default="")

class InferlessPythonModel:
    def initialize(self):
        model_id = "nanonets/Nanonets-OCR-s"
        self.model = AutoModelForImageTextToText.from_pretrained(model_id,torch_dtype="auto",device_map="cuda",).eval()
        self.tokenizer  = AutoTokenizer.from_pretrained(model_id)
        self.processor  = AutoProcessor.from_pretrained(model_id)

    def infer(self, request: RequestObjects) -> ResponseObjects:
        image = self._fetch_image(request.image_url)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": request.prompt},
                ],
            },
        ]

        text_inputs = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text_inputs], images=[image], padding=True, return_tensors="pt").to(self.model.device)
      
        with torch.inference_mode():
            out_ids  = self.model.generate(**inputs, max_new_tokens=request.max_new_tokens, do_sample=request.do_sample)
            gen_ids  = out_ids[:, inputs["input_ids"].shape[-1] :]
            decoded  = self.processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

        return ResponseObjects(extracted_text=decoded)

    def finalize(self):
        self.model = self.processor = self.tokenizer = None

    @staticmethod
    def _fetch_image(url: str) -> Image.Image:
        resp = requests.get(url, timeout=300)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
