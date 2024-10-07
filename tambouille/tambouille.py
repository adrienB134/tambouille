import outlines
from outlines.models import transformers_vision

from transformers import Qwen2VLForConditionalGeneration
from .utils import tear_down_torch

# from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from typing import Union

from datasets import load_dataset, Dataset, DatasetDict

from qwen_vl_utils import process_vision_info
from .prompt import Prompt

from pathlib import Path
from typing import cast

import torch
from colpali_engine.collators.visual_retriever_collator import VisualRetrieverCollator
from colpali_engine.loss import ColbertPairwiseCELoss
from colpali_engine.models import ColPali, ColPaliProcessor, ColQwen2, ColQwen2Processor
from colpali_engine.trainer.contrastive_trainer import ContrastiveTrainer
from colpali_engine.utils.torch_utils import get_torch_device
from datasets import DatasetDict, load_dataset
from transformers import BitsAndBytesConfig, TrainerCallback, TrainingArguments
from typing import Union, Literal
from .utils import (
    EvaluateFirstStepCallback,
    print_trainable_parameters,
    tear_down_torch,
)
import wandb


class Tambouille:
    """
    A class for managing and training visual retrieval models.

    This class provides functionality to initialize, generate datasets for, and train
    ColQwen2 or ColPali models for visual retrieval tasks.
    """

    def __init__(
        self,
        model_name: str = "vidore/colqwen2-v0.1",
        model_type: Literal["colqwen2", "colpali"] = "colqwen2",
    ):
        """
        Initialize a Tambouille instance.

        Args:
            model_name (str): The name or path of the pre-trained model to use.
                Defaults to "vidore/colqwen2-v0.1".
            model_type (Literal["colqwen2", "colpali"]): The type of model to use.
                Must be either "colqwen2" or "colpali". Defaults to "colqwen2".

        Raises:
            ValueError: If an unsupported model_type is provided.
        """
        self.model_name = model_name
        self.model_type = model_type
        if self.model_type == "colqwen2":
            self.processor = ColQwen2Processor
            self.model = ColQwen2
        elif self.model_type == "colpali":
            self.processor = ColPaliProcessor
            self.model = ColPali
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def generate_dataset(
        self,
        qwen_vl_model: str,
        prompt: Prompt,
        dataset: Union[Dataset, str],
        img_column_name: str = "image",
        test_size: float = 0.1,
        max_pixels: int = 1280 * 28 * 28,
        num_samples: Union[int, None] = None,
    ) -> DatasetDict:
        """
        Generate a dataset using a Qwen vision-language model.

        Args:
            qwen_vl_model (str): The name of the Qwen vision-language model to use.
            prompts (Prompt): The prompt to use for generating queries.
            dataset (Union[Dataset, str]): An instance of Dataset or a HuggingFace dataset name.
            img_column_name (str, optional): The name of the image column. Defaults to "image".
            test_size (float, optional): The proportion of the dataset to use for the test set. Defaults to 0.2.
            max_pixels (int, optional): The maximum number of pixels in the image. Defaults to 1280 * 28 * 28 (maximum for Qwen2-VL).
            num_samples (Union[int, None], optional): The number of images to use if you don't want to use the whole dataset (e.g. for testing prompts). Defaults to None.

        Returns:
            DatasetDict: A dataset dictionary containing the generated dataset split into train and test sets.
        """

        model = transformers_vision(
            qwen_vl_model,
            model_kwargs=dict(
                torch_dtype="auto", attn_implementation="flash_attention_2"
            ),
            processor_kwargs=dict(torch_dtype="auto", max_pixels=max_pixels),
            model_class=Qwen2VLForConditionalGeneration,
            device="cuda",
        )

        description_generator = outlines.generate.json(model, prompt.schema)

        if isinstance(dataset, str):
            dataset = load_dataset(dataset)["train"]

        if num_samples is not None:
            dataset = dataset.take(num_samples)

        if img_column_name != "image":
            dataset = dataset.rename_column(img_column_name, "image")
            dataset = dataset.remove_columns(
                [key for key in dataset.column_names if key not in ["image"]]
            )

        results = []
        for i in tqdm(range(len(dataset))):
            image = dataset[i]["image"]
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {"type": "text", "text": prompt.prompt},
                    ],
                }
            ]
            text = model.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            image_input, _ = process_vision_info(conversation)
            try:
                desc = description_generator(
                    text,
                    image_input,
                    max_tokens=2048,  # to avoid infinite generation hang-up
                )
                results.append(desc.dict())
            except Exception as e:
                print(f"Generation Error {e}")
                results.append({field: None for field in prompt.schema.model_fields})
                pass
        dataset = dataset.map(lambda x: results.pop(0))
        dataset = dataset.filter(lambda x: x["simple_query"] is not None)

        # clear GPU memory
        del model
        del description_generator
        tear_down_torch()

        return dataset.train_test_split(test_size=test_size)

    # ? Need to implement some kind of query quality filtering method, qwen2-vl reverts too often to chinese or just plain gibberish

    def train(
        self,
        dataset: Union[str, DatasetDict],
        query_column_name: str = "query",
        image_column_name: str = "image",
        training_args: TrainingArguments = None,
        bnb_config: BitsAndBytesConfig = None,
        callbacks: TrainerCallback = EvaluateFirstStepCallback(),
        push_to_hub: bool = False,
        repo_id: Union[str, None] = None,
        local_save: bool = False,
        wandb_logging: bool = False,
        wandb_project: Union[str, None] = None,
        wandb_experiment_name: Union[str, None] = None,
    ) -> None:
        """
        Train a ColPali or ColQwen2 model on a given dataset.

        Args:
            dataset (Union[str, DatasetDict]): The dataset to train on. Either HuggingFace dataset name or a DatasetDict.
            query_column_name (str, optional): The name of the column containing the query text. Defaults to "query".
            image_column_name (str, optional): The name of the column containing the image data. Defaults to "image".
            training_args (TrainingArguments, optional): Custom training arguments. Defaults to None.
            bnb_config (BitsAndBytesConfig, optional): Configuration for quantization. Defaults to None.
            callbacks (TrainerCallback, optional): Custom callbacks for the trainer. Defaults to EvaluateFirstStepCallback().
            push_to_hub (bool, optional): Whether to push the trained model to the HuggingFace Hub. Defaults to False.
            repo_id (Union[str, None], optional): The name of the repo to use when pushing to the HuggingFace Hub. Defaults to None.
            local_save (bool, optional): Whether to save the model locally. Defaults to False.
            wandb_logging (bool, optional): Whether to use Weights & Biases logging. Defaults to False.
            wandb_project (Union[str, None], optional): The name of the Weights & Biases project. Defaults to None.
            wandb_experiment_name (Union[str, None], optional): The name of the Weights & Biases experiment. Defaults to None.

        Returns:
            None
        """

        bnb_config = bnb_config or BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        device = get_torch_device("auto")

        # Load the model with the loaded pre-trained adapter
        model = cast(
            self.model,
            self.model.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map=device,
            ),
        )
        processor = cast(
            self.processor,
            self.processor.from_pretrained(self.model_name),
        )
        collator = VisualRetrieverCollator(processor=processor)

        if not model.active_adapters():
            raise ValueError("No adapter found in the model.")

        # The LoRA weights are frozen by default. We need to unfreeze them to fine-tune the model.
        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True

        print_trainable_parameters(model)

        # Load the dataset

        if isinstance(dataset, str):
            ds = cast(DatasetDict, load_dataset(dataset))
        else:
            ds = dataset
        # Rename the columns to match the trainer's requirements
        if image_column_name != "image":
            ds = ds.rename_column(image_column_name, "image")
        if query_column_name != "query":
            ds = ds.rename_column(query_column_name, "query")

        ds["train"] = ds["train"].shuffle(seed=42)

        checkpoints_dir = Path("checkpoints")
        checkpoints_dir.mkdir(exist_ok=True, parents=True)

        training_args = training_args or TrainingArguments(
            output_dir=str(checkpoints_dir),
            hub_model_id=None,
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            eval_strategy="steps",
            save_steps=200,
            logging_steps=2,
            eval_steps=100,
            warmup_steps=100,
            learning_rate=5e-5,
            save_total_limit=1,
            report_to=["wandb"] if wandb_logging else [],
            run_name=wandb_experiment_name,
            push_to_hub=push_to_hub,
            push_to_hub_model_id=repo_id,
        )

        trainer = ContrastiveTrainer(
            model=model,
            train_dataset=ds["train"],
            eval_dataset=ds["test"],
            args=training_args,
            data_collator=collator,
            loss_func=ColbertPairwiseCELoss(),
            is_vision_model=True,  # I think this is not needed
        )

        trainer.args.remove_unused_columns = False
        if callbacks is not None:
            trainer.add_callback(callbacks)

        if wandb_logging:
            wandb_tags = ["finetuning", "colpali"]

            if bnb_config:
                wandb_tags.append("quantization")

            run = wandb.init(
                project=wandb_project,
                name=wandb_experiment_name,
                job_type="finetuning",
                tags=wandb_tags,
                config={
                    "model_name": self.model_name,
                    "bitsandbytes_config": bnb_config.to_dict() if bnb_config else None,
                },
            )

        if callbacks is not None:
            eval_results_before = trainer.evaluate()

        # Train the model
        trainer.create_model_card(
            model_name=repo_id, tags=["ColPali", "🍲 Tambouille"], license="mit"
        )
        trainer.train()

        eval_results = trainer.evaluate()

        if callbacks is not None:
            print(f"Performance before training: {eval_results_before}")
        print(f"Performance after training: {eval_results}")

        if wandb_logging:
            run.finish()

        if local_save:
            trainer.save_model("./model")

        # Clean up the GPU memory
        del model
        del trainer
        tear_down_torch()
