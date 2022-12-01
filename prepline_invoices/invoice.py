from __future__ import annotations
from abc import ABC, abstractclassmethod
from dataclasses import dataclass
from pathlib import Path
import torch
from typing import Final, List, Optional, Tuple, Union

from donut.model import DonutModel

from transformers import (
    LayoutLMModel,
    LayoutLMForTokenClassification,
)

from PIL import Image


DONUT_MODEL_PATH: Final = "unstructuredio/donut-invoices-fake"
#DONUT_MODEL_PATH: Final = "/Users/ajimeno/Documents/git/donut/result.prev"
LAYOUTLM_MODEL_PATH: Final = "microsoft/layoutlm-base-uncased"


class DonutModelMain(DonutModel):
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, bytes, os.PathLike],
        *model_args,
        **kwargs,
    ):
        r"""
        Instantiate a pretrained donut model from a pre-trained model configuration

        Args:
            pretrained_model_name_or_path:
                Name of a pretrained model name either registered in huggingface.co. or saved in local,
                e.g., `naver-clova-ix/donut-base`, or `naver-clova-ix/donut-base-finetuned-rvlcdip`
        """
        model = super(DonutModel, cls).from_pretrained(pretrained_model_name_or_path, revision="main", *model_args, **kwargs)

        # truncate or interplolate position embeddings of donut decoder
        max_length = kwargs.get("max_length", model.config.max_position_embeddings)
        if (
            max_length != model.config.max_position_embeddings
        ):  # if max_length of trained model differs max_length you want to train
            model.decoder.model.model.decoder.embed_positions.weight = torch.nn.Parameter(
                model.decoder.resize_bart_abs_pos_emb(
                    model.decoder.model.model.decoder.embed_positions.weight,
                    max_length
                    + 2,  # https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/mbart/modeling_mbart.py#L118-L119
                )
            )
            model.config.max_position_embeddings = max_length

        return model

def donut_load_model(
    model_path: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
) -> DonutModel:
    """Loads the donut model using the specified parameters"""
    if model_path is None:
        model_path = DONUT_MODEL_PATH

    model = DonutModelMain.from_pretrained(model_path, use_auth_token=True)

    if device is not None:
        model.to(device)

    return model


def layoutlm_load_model(
    model_path: Optional[Union[str, Path]] = None,
    num_labels: Optional[int] = None,
    device: Optional[str] = None,
) -> LayoutLMModel:
    """Loads the LayoutLMModel model using the specified parameters"""

    model = LayoutLMForTokenClassification.from_pretrained(
        LAYOUTLM_MODEL_PATH, num_labels=num_labels
    )

    model.load_state_dict(torch.load(model_path))

    if device is not None:
        model.to(device)

    return model


@dataclass
class InvoiceElement:
    # Left, top, width, height
    coordinates: List[Tuple[float, float, float, float]]
    fieldName: Optional[str] = None
    text: Optional[str] = None

    def __str__(self):
        return self.fieldName + ": " + self.text

    def to_dict(self):
        return self.__dict__


class InvoiceModel(ABC):
    @abstractclassmethod
    def get_elements(
        image: Image,
    ) -> Optional[
        Tuple[Optional[List[InvoiceElement]], Optional[List[List[InvoiceElement]]]]
    ]:
        pass


class InvoiceModelLayoutLM(InvoiceModel):
    pass
    # def __init__(self, model_file_name: str, device: Optional[str] = None):
    #    layoutlmv1.load_model(model_file_name, device)


class InvoiceModelDonut(InvoiceModel):
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model = donut_load_model(model_path, device)

    def get_elements(
        self, image: Image
    ) -> Optional[
        Tuple[Optional[List[InvoiceElement]], Optional[List[List[InvoiceElement]]]]
    ]:
        prediction = self.model.inference(image, prompt="<s_dataset-donut-generated>",)[
            "predictions"
        ][0]

        return [
            InvoiceElement(coordinates=None, fieldName=k, text=v)
            for k, v in prediction.items()
            if k != "items"
        ], [
            [
                InvoiceElement(coordinates=None, fieldName=k, text=v)
                for k, v in itemLine.items()
            ]
            for itemLine in (
                prediction["items"]
                if type(prediction["items"]) is list
                else [prediction["items"]]
            )
        ]


class DocumentInvoice:
    """Class for handling invoices that are saved as image files."""

    def __init__(self):
        self._pages = None

    def __str__(self) -> str:
        return "\n\n".join([str(page) for page in self.pages])

    @property
    def pages(self) -> List[PageInvoice]:
        """Gets all elements from pages in sequential order."""
        return self._pages

    @classmethod
    def from_pages(cls, pages: List[PageInvoice]) -> DocumentInvoice:
        """Generates a new instance of the class from a list of `PageInvoices`s"""
        invoice = cls()
        invoice._pages = pages
        return invoice

    @classmethod
    def from_file(cls, filename: str, model: Optional[InvoiceModel] = None):
        # logger.info(f"Reading invoice image for file: {filename} ...")

        page = PageInvoice(image=Image.open(filename), model=model)
        page.get_elements()
        return cls.from_pages([page])


class PageInvoice:
    def __init__(self, image: Image, model: InvoiceModel):
        self.image = image
        self.elements: List[InvoiceElement]
        self.itemLists = List[List[InvoiceElement]]
        self.model = model

    def __str__(self):
        return "\n\n".join(
            [str(element) for element in self.elements]
            + [
                " ".join([str(element) for element in itemList])
                for itemList in self.itemLists
            ]
        )

    def get_elements(
        self, inplace=True
    ) -> Optional[
        Tuple[Optional[List[InvoiceElement]], Optional[List[List[InvoiceElement]]]]
    ]:
        elements, itemLists = self.model.get_elements(self.image)

        if inplace:
            self.elements = elements
            self.itemLists = itemLists
            return None

        return elements, itemLists
