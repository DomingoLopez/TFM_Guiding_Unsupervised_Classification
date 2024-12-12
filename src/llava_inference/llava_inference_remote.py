from pathlib import Path
import pickle
import pandas as pd
from transformers import LlavaProcessor, LlavaForConditionalGeneration, LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import time
import os
import matplotlib.pyplot as plt

#from src.utils.image_loader import ImageLoader
#from loguru import logger


class LlavaInferenceRemote():
    """
    LlavaInferenceRemote for execution on remote, with only archives needed
    """
    def __init__(self,
                 classification_lvl: str,
                 experiment:int,
                 name:str,
                 n_prompt:int,
                 type:str,
                 cache: bool = False,
                 verbose: bool = False):
        """
        Loads images from every cluster in order to do some inference on llava on ugr gpus
        Args:
            images_cluster_dict (dict)
            classification_lvl (str): Classification level to be used
            experiment_name (str): Name of the experiment for organizing results
        """
        self.experiment = experiment
        self.name = name
        self.type=type
        self.base_dir = Path(__file__).resolve().parent / f"cluster_images/experiment_{experiment}" / f"{name}"
        self.results_dir = Path(__file__).resolve().parent / f"results/classification_lvl_{classification_lvl}/experiment_{experiment}" / f"{name}" / f"prompt_{n_prompt}"
        self.results_object = self.results_dir / f"result_{self.type}.pkl"
        self.results_csv = self.results_dir / f"inference_results_{self.type}.csv"
        self.classification_lvls_dir = Path(__file__).resolve().parent / "classification_lvls/"
 
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # Load categories based on classification level
        self.classification_lvl = classification_lvl
        self.categories = pd.read_csv(os.path.join(self.classification_lvls_dir, f"classification_level_{self.classification_lvl}.csv"), header=None, sep=";").iloc[:, 0].tolist()
        self.cache = cache

        # Initialize images_dict_format
        self.images_dict_format = self.get_cluster_images_dict()

        # Results
        self.result_df = None
        self.result_stats_df = None


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
        
            
        self.prompt = self.prompt_1 if n_prompt == 1 else self.prompt_2




    def show_prompts(self):
        print(self.prompt_1)
        print(self.prompt_2)



    def get_cluster_images_dict(self, knn=None):
        
        cluster_images_dict = {}
        for cluster_dir in self.base_dir.iterdir():
            if cluster_dir.is_dir():
                cluster_id = int(cluster_dir.name)
                cluster_images_dict[cluster_id] = [str(img_path) for img_path in cluster_dir.iterdir() if img_path.is_file()]
        return dict(sorted(cluster_images_dict.items()))





    def run(self):
        if self.type == "llava":
            self.__run_llava()
        elif self.type == "llava_next":
            self.__run_llava_next()
        else:
            self.__run_llava_next_2()



    def __run_llava(self):
        """
        Run Llava inference for every image in each subfolder of the base path.
        Store results.
        """
        if os.path.isfile(self.results_object) and self.cache:
            print("Recovering results from cache")
            self.result_df = pickle.load(open(str(self.results_object), "rb"))
        else:
            processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
            model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
            model.to("cuda:0")

            results = []
            print("Launching llava")
            
            for cluster_name, image_paths in self.images_dict_format.items():
                print(f"Cluster {cluster_name}. Imágenes: {len(image_paths)}")
                for image_path in image_paths:
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
                        "cluster": cluster_name,
                        "img": image_path,
                        "category_llava": classification_category,
                        "output": classification_result,
                        "inference_time": inference_time
                    })

            results_df = pd.DataFrame(results)
            results_df.to_csv(self.results_csv, index=False, sep=";")
            pickle.dump(results_df, open(self.results_object, "wb"))
            self.result_df = results_df
            #logger.info(f"Results saved to {results_path}")


    def __run_llava_next(self):
        """
        Run Llava-Next inference for every image in each subfolder of the base path.
        Store results.
        """
        if os.path.isfile(self.results_object) and self.cache:
            print("Recovering results from cache")
            self.result_df = pickle.load(open(str(self.results_object), "rb"))
        else:
            # "llava-hf/llava-v1.6-mistral-7b-hf"
            processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
            model = LlavaNextForConditionalGeneration.from_pretrained(
                "llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True
            )
            model.to("cuda:0")
            model.config.pad_token_id = model.config.eos_token_id

            results = []
            print("Launching llava-next")
            for cluster_name, image_paths in self.images_dict_format.items():
                print(f"Cluster {cluster_name}. Images: {len(image_paths)}")
                for image_path in image_paths:
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
                            "cluster": cluster_name,
                            "img": image_path,
                            "category_llava": classification_category,
                            "output": classification_result,
                            "inference_time": inference_time
                        })
                    except Exception as e:
                        print(f"Error processing image {image_path}: {e}")

            results_df = pd.DataFrame(results)
            results_df.to_csv(self.results_csv, index=False, sep=";")
            pickle.dump(results_df, open(self.results_object, "wb"))
            self.result_df = results_df



    def __run_llava_next_2(self):
        """
        Run Llava-Next inference for every image in each subfolder of the base path.
        Store results.
        """
        if os.path.isfile(self.results_object) and self.cache:
            print("Recovering results from cache")
            self.result_df = pickle.load(open(str(self.results_object), "rb"))
        else:
            # "llava-hf/llava-v1.6-mistral-7b-hf"
            processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf")
            model = LlavaNextForConditionalGeneration.from_pretrained(
                "llava-hf/llava-v1.6-vicuna-13b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True
            )
            model.to("cuda:0")
            model.config.pad_token_id = model.config.eos_token_id

            results = []
            print("Launching llava-next-vicuna-13b")
            for cluster_name, image_paths in self.images_dict_format.items():
                print(f"Cluster {cluster_name}. Images: {len(image_paths)}")
                for image_path in image_paths:
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
                        
                        if "ASSISTANT" in classification_result:
                            classification_category = classification_result.split("ASSISTANT:",1)[-1].strip()
                        else:
                            classification_category = "Unknown"  # Handle unexpected output format

                        inference_time = time.time() - start_time

                        
                        results.append({
                            "cluster": cluster_name,
                            "img": image_path,
                            "category_llava": classification_category,
                            "output": classification_result,
                            "inference_time": inference_time
                        })
                    except Exception as e:
                        print(f"Error processing image {image_path}: {e}")

            results_df = pd.DataFrame(results)
            results_df.to_csv(self.results_csv, index=False, sep=";")
            pickle.dump(results_df, open(self.results_object, "wb"))
            self.result_df = results_df





if __name__ == "__main__":
    llava = LlavaInferenceRemote(3,1,"index_18_silhouette_0.755",1,"llava",False,False)
    llava.run()
    llava = LlavaInferenceRemote(3,1,"index_18_silhouette_0.755",2,"llava",False,False)
    llava.run()
    llava = LlavaInferenceRemote(3,1,"index_18_silhouette_0.755",1,"llava_next",False,False)
    llava.run()
    llava = LlavaInferenceRemote(3,1,"index_18_silhouette_0.755",2,"llava_next",False,False)
    llava.run()
    llava = LlavaInferenceRemote(3,1,"index_18_silhouette_0.755",1,"llava_next_13b",False,False)
    llava.run()
    llava = LlavaInferenceRemote(3,1,"index_18_silhouette_0.755",2,"llava_next_13b",False,False)
    llava.run()
    # classification lvl 2#######################################################################
    llava = LlavaInferenceRemote(2,1,"index_18_silhouette_0.755",1,"llava",False,False)
    llava.run()
    llava = LlavaInferenceRemote(2,1,"index_18_silhouette_0.755",2,"llava",False,False)
    llava.run()
    llava = LlavaInferenceRemote(2,1,"index_18_silhouette_0.755",1,"llava_next",False,False)
    llava.run()
    llava = LlavaInferenceRemote(2,1,"index_18_silhouette_0.755",2,"llava_next",False,False)
    llava.run()
    llava = LlavaInferenceRemote(2,1,"index_18_silhouette_0.755",1,"llava_next_13b",False,False)
    llava.run()
    llava = LlavaInferenceRemote(2,1,"index_18_silhouette_0.755",2,"llava_next_13b",False,False)
    llava.run()
# classification lvl 1#######################################################################
    llava = LlavaInferenceRemote(1,1,"index_18_silhouette_0.755",1,"llava",False,False)
    llava.run()
    llava = LlavaInferenceRemote(1,1,"index_18_silhouette_0.755",2,"llava",False,False)
    llava.run()
    llava = LlavaInferenceRemote(1,1,"index_18_silhouette_0.755",1,"llava_next",False,False)
    llava.run()
    llava = LlavaInferenceRemote(1,1,"index_18_silhouette_0.755",2,"llava_next",False,False)
    llava.run()
    llava = LlavaInferenceRemote(1,1,"index_18_silhouette_0.755",1,"llava_next_13b",False,False)
    llava.run()
    llava = LlavaInferenceRemote(1,1,"index_18_silhouette_0.755",2,"llava_next_13b",False,False)
    llava.run()

