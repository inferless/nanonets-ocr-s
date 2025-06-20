# Tutorial - Deploy Nanonets-OCR-s using Inferless
Nanonets-OCR-s is an open-source, **3-billion-parameter** vision-language model that turns scanned pages and PDFs directly into richly structured Markdown instead of flat text. It preserves tables as HTML, renders equations in LaTeX, tags check-boxes with ☐/☑, wraps page numbers and watermarks in explicit tags, and even inserts image captions or auto-generated descriptions inside `<img>` elements—producing outputs that are ready for downstream LLM or RAG pipelines.

Under the hood, Nanonets-OCR-s is **fine-tuned from the Qwen 2.5-VL-3B-Instruct backbone**, inheriting that model’s strong multimodal reasoning and layout-aware capabilities. ([huggingface.co][2], [huggingface.co][3])  This choice gives the OCR system a compact size that still fits on a single consumer GPU while reaching state-of-the-art accuracy on complex documents. Community posts and the official announcement highlight that the entire 3 B stack is released under the Apache-2.0 licence, making it free to self-host, fine-tune or embed in commercial workflows.

## TL;DR:
- Deployment of Nanonets-OCR-s model using [transformers](https://github.com/huggingface/transformers).
- Dependencies defined in `inferless-runtime-config.yaml`.
- GitHub/GitLab template creation with `app.py`, `inferless-runtime-config.yaml` and `inferless.yaml`.
- Model class in `app.py` with `initialize`, `infer`, and `finalize` functions.
- Custom runtime creation with necessary system and Python packages.
- Recommended GPU: NVIDIA A100 for optimal performance.
- Custom runtime selection in advanced configuration.
- Final review and deployment on the Inferless platform.

### Fork the Repository
Get started by forking the repository. You can do this by clicking on the fork button in the top right corner of the repository page.

This will create a copy of the repository in your own GitHub account, allowing you to make changes and customize it according to your needs.

### Create a Custom Runtime in Inferless
To access the custom runtime window in Inferless, simply navigate to the sidebar and click on the Create new Runtime button. A pop-up will appear.

Next, provide a suitable name for your custom runtime and proceed by uploading the **inferless-runtime-config.yaml** file given above. Finally, ensure you save your changes by clicking on the save button.

### Import the Model in Inferless
Log in to your inferless account, select the workspace you want the model to be imported into and click the `Add a custom model` button.

- Select `Github` as the method of upload from the Provider list and then select your Github Repository and the branch.
- Choose the type of machine, and specify the minimum and maximum number of replicas for deploying your model.
- Configure Custom Runtime ( If you have pip or apt packages), choose Volume, Secrets and set Environment variables like Inference Timeout / Container Concurrency / Scale Down Timeout
- Once you click “Continue,” click Deploy to start the model import process.

Enter all the required details to Import your model. Refer [this link](https://docs.inferless.com/integrations/git-custom-code/git--custom-code) for more information on model import.

---
## Curl Command
Following is an example of the curl command you can use to make inference. You can find the exact curl command in the Model's API page in Inferless.
```bash
curl --location '<your_inference_url>' \
    --header 'Content-Type: application/json' \
    --header 'Authorization: Bearer <your_api_key>' \
    --data '{
      "inputs": [
                    {
                      "name": "image_url",
                      "shape": [1],
                      "datatype": "BYTES",
                      "data": [
                        "https://github.com/NanoNets/docext/raw/main/assets/invoice_test.jpeg"
                      ]
                    },
                    {
                      "name": "prompt",
                      "shape": [1],
                      "datatype": "BYTES",
                      "data": [
                              "Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using \u2610 and \u2611 for check boxes."
                        ]
                    },
                    {
                      "name": "temperature",
                      "shape": [1],
                      "datatype": "FP64",
                      "data": [0.7]
                    },
                    {
                      "name": "do_sample",
                      "shape": [1],
                      "datatype": "BOOL",
                      "data": [true]
                    },
                    {
                      "name": "max_new_tokens",
                      "shape": [1],
                      "datatype": "INT32",
                      "data": [15000]
                    }
                ]
}'
```

---
## Customizing the Code
Open the `app.py` file. This contains the main code for inference. The `InferlessPythonModel` has three main functions, initialize, infer and finalize.

**Initialize** -  This function is executed during the cold start and is used to initialize the model. If you have any custom configurations or settings that need to be applied during the initialization, make sure to add them in this function.

**Infer** - This function is where the inference happens. The infer function leverages both RequestObjects and ResponseObjects to handle inputs and outputs in a structured and maintainable way.
- RequestObjects: Defines the input schema, validating and parsing the input data.
- ResponseObjects: Encapsulates the output data, ensuring consistent and structured API responses.

**Finalize** - This function is used to perform any cleanup activity for example you can unload the model from the gpu by setting to `None`.

For more information refer to the [Inferless docs](https://docs.inferless.com/).
