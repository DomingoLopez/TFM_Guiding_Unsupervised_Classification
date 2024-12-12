from pathlib import Path
import pickle
import shutil
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from transformers import LlavaProcessor, LlavaForConditionalGeneration, LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import time
import os
import sys
import matplotlib.pyplot as plt
from scipy.stats import entropy


#from src.utils.image_loader import ImageLoader
#from loguru import logger


class LlavaInference():
    """
    LlavaInference allows us to deploy selected Llava model (locally or in NGPU - UGR, but without automation yet)
    We start with Llava1.5-7b params. It can download model, and do some inference given some images and text prompt as inputs.
    """
    def __init__(self, 
                 images: list,
                 classification_lvl: str,
                 n_prompt:int,
                 model:str,
                 cache: bool = True, 
                 verbose: bool = False):
        """
        Loads images from every cluster in order to do some inference on llava on ugr gpus
        Args:
            images_cluster_dict (dict)
            classification_lvl (str): Classification level to be used
            experiment_name (str): Name of the experiment for organizing results
        """

        if(model not in ("llava1-5_7b", "llava1-6_7b","llava1-6_13b")):
            raise ValueError("type must be one of followin: [llava1-5_7b, llava1-6_7b,llava1-6_13b]")
        
        # Adjust model from huggint face, but anyway, we need 2 different methods
        # depending on llava or llava-next
        if model == "llava1-5_7b":
            self.model_hf = "llava-hf/llava-1.5-7b-hf"
        elif model == "llava1-6_7b":
            self.model_hf = "llava-hf/llava-v1.6-mistral-7b-hf"
        elif model == "llava1-6_13b":
            self.model_hf = "liuhaotian/llava-v1.6-vicuna-13b"
        else:
            self.model_hf = "llava-hf/llava-v1.6-mistral-7b-hf"

        self.images = images
        self.classification_lvl = classification_lvl
        self.model = model
        self.n_prompt = n_prompt
        self.cache = cache
        self.verbose = verbose
        # Base dirs
        self.results_dir = Path(__file__).resolve().parent / f"results/classification_lvl_{self.classification_lvl}/{self.model}/prompt_{self.n_prompt}"
        self.results_csv = self.results_dir / f"inference_results.csv"
        self.classification_lvls_dir = Path(__file__).resolve().parent / "classification_lvls/"
        # Ensure directories exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load categories based on classification level
        self.categories = pd.read_csv(os.path.join(self.classification_lvls_dir, f"classification_level_{self.classification_lvl}.csv"), header=None, sep=";").iloc[:, 0].tolist()
        categories_joins = ", ".join([category.upper() for category in self.categories])

        self.prompt_1 = (
            "You are an Image Classification Assistant specialized in identifying cultural ecosystem services and the cultural contributions of nature to people. "
            f"Your task is to classify images into one of the following {len(self.categories)} categories: {categories_joins}. "
            "Please adhere to the following rules:"
            "1. You must not assign a category that is not listed above."
            "2. If the image does not belong to any of the listed categories, classify it as 'NOT VALID'."
            "3. Provide your response exclusively as the classification, without any additional explanation or commentary."
            )
        
        self.prompt_2 = (
            "You are an Image Classification Assistant specialized in identifying cultural ecosystem services and the cultural contributions of nature to people. "
            f"Your task is to classify images into one of the following {len(self.categories)} categories: {categories_joins}. "
            "Please adhere to the following rules:"
            "1. You must not assign a category that is not listed above."
            "2. If the image does not clearly belong to any of the listed categories, classify it as the most similar category from the list."
            "3. If the image is not clear enough or blurry, classify it as 'NOT VALID'."
            "4. Provide your response EXCLUSIVELY as the classification, without any additional explanation or commentary."
            )
        
        
        if n_prompt > 2 or n_prompt < 1:
                raise ValueError("n_prompt must be 1 or 2")
            
        self.prompt = self.prompt_1 if n_prompt == 1 else self.prompt_2



    def show_prompts(self):
        print(self.prompt_1)
        print(self.prompt_2)


    def run(self):
        self.__run_llava() if self.model == "llava1-5_7b" else self.__run_llava_next()

    

    # TODO: TAKE IMAGES AND INFERENCE THEM
    def __run_llava(self):
        """
        Run Llava inference for every image in each subfolder of the base path.
        Store results.
        """
        if os.path.isfile(self.results_csv) and self.cache:
            print("Recovering results from cache")
            self.result_df = pd.read_csv(self.results_csv, sep=";", header=0) 
        else:
            processor = LlavaProcessor.from_pretrained(self.model_hf)
            model = LlavaForConditionalGeneration.from_pretrained(self.model_hf, 
                                                                  torch_dtype=torch.float16, 
                                                                  low_cpu_mem_usage=True)
            model.to("cuda:0")

            results = []
            print(f"Launching llava: {self.model_hf}")
            
            for image_path in self.images:
                image = Image.open(image_path)
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.prompt},
                            {"type": "image", "image": image},  
                        ],
                    },
                ]
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")
                
                start_time = time.time()
                output = model.generate(**inputs, max_new_tokens=500)
                classification_result = processor.decode(output[0], skip_special_tokens=True)
                classification_category = classification_result.split(":")[-1].strip()
                inference_time = time.time() - start_time

                results.append({
                    "img": image_path,
                    "category_llava": classification_category,
                    "output": classification_result,
                    "inference_time": inference_time
                })

            results_df = pd.DataFrame(results)
            results_df.to_csv(self.results_csv, index=False, sep=";")
            self.result_df = results_df



    # TODO: TAKE IMAGES FROM DATA, AND INFERENCE THEM
    def __run_llava_next(self):
        """
        Run Llava-Next inference for every image in each subfolder of the base path.
        Store results.
        """
        if os.path.isfile(self.results_csv) and self.cache:
            print("Recovering results from cache")
            self.result_df = pd.read_csv(self.results_csv, sep=";", header=0) 
        else:
            processor = LlavaNextProcessor.from_pretrained(self.model_hf)
            model = LlavaNextForConditionalGeneration.from_pretrained(self.model_hf, 
                                                                      torch_dtype=torch.float16, 
                                                                      low_cpu_mem_usage=True)
            model.to("cuda:0")
            model.config.pad_token_id = model.config.eos_token_id

            results = []
            print(f"Launching llava: {self.model_hf}")
            
            for image_path in self.images:
                try:
                    image = Image.open(image_path).convert("RGB")  # Ensure compatibility with the model
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.prompt},
                                {"type": "image", "image": image},  
                            ],
                        },
                    ]
                    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

                    start_time = time.time()
                    output = model.generate(**inputs, max_new_tokens=500)

                    classification_result = processor.decode(output[0], skip_special_tokens=True)
                    
                    if "[/INST]" in classification_result:
                        classification_category = classification_result.split("[/INST]")[-1].strip()
                    else:
                        classification_category = "Unknown"  # Handle unexpected output format

                    inference_time = time.time() - start_time

                    results.append({
                        "img": image_path,
                        "category_llava": classification_category,
                        "output": classification_result,
                        "inference_time": inference_time
                    })
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")

            results_df = pd.DataFrame(results)
            results_df.to_csv(self.results_csv, index=False, sep=";")
            self.result_df = results_df




    def get_results(self,model_name):
        """
        Returns inference results for given model name 
        (on classification_lvl where it was created, and for given prompt)
        """
        results = None
        try:
            results = pd.read_csv(Path(__file__).resolve().parent / f"results/classification_lvl_{self.classification_lvl}/{model_name}/prompt_{self.n_prompt}/inference_results.csv",
                                  sep=";",
                                  header=0)
        except:
            ValueError("File not found")

        return results
    


    def get_categories(self, classsification_lvl):
        """
        Returns categories from classification_lvl
        """
        categories = None
        try:
            categories = pd.read_csv(Path(__file__).resolve().parent / f"classification_lvls/classification_level_{classsification_lvl}.csv",
                                  sep=";",
                                  header=None)
        except:
            ValueError("File not found")

        data_list = categories[0].to_list()
        uppercase_list = [item.upper() for item in data_list]
        return uppercase_list







if __name__ == "__main__":
    pass
