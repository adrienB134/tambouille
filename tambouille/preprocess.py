import os
import tempfile
from typing import List, Union
from PIL import Image
from pdf2image import convert_from_path


def encode_image(
    input_data: Union[str, Image.Image, List[Union[str, Image.Image]]],
) -> None:
    if not isinstance(input_data, list):
        input_data = [input_data]

    images = []
    for item in input_data:
        if isinstance(item, Image.Image):
            images.append(item)
        elif isinstance(item, str):
            if os.path.isdir(item):
                # Process folder
                for file in os.listdir(item):
                    if file.lower().endswith(
                        (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
                    ):
                        images.append(Image.open(os.path.join(item, file)))
            elif item.lower().endswith(".pdf"):
                # Process PDF
                with tempfile.TemporaryDirectory() as path:
                    pdf_images = convert_from_path(
                        item, thread_count=os.cpu_count() - 1, output_folder=path
                    )
                    images.extend(pdf_images)
            elif item.lower().endswith(
                (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
            ):
                # Process image file
                images.append(Image.open(item))
            else:
                raise ValueError(f"Unsupported file type: {item}")
        else:
            raise ValueError(f"Unsupported input type: {type(item)}")
