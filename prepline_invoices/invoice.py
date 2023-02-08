from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Final, List, Optional, Tuple, Union, BinaryIO

from transformers import DonutProcessor, VisionEncoderDecoderModel

from PIL import Image


DONUT_MODEL_PATH: Final = "unstructuredio/donut-invoices"


def donut_load_model(
    model_path: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
) -> Tuple[VisionEncoderDecoderModel, DonutProcessor]:
    """Loads the donut model using the specified parameters"""
    if model_path is None:
        model_path = DONUT_MODEL_PATH

    processor = DonutProcessor.from_pretrained(
        "unstructuredio/donut-invoices",
        max_length=1200,
    )
    model = VisionEncoderDecoderModel.from_pretrained(
        "unstructuredio/donut-invoices",
        max_length=1200,
    )

    if device is not None:
        model.to(device)

    return model, processor


@dataclass
class InvoiceElement:
    fieldName: Optional[str] = None
    text: Optional[str] = None

    def __str__(self):
        return self.fieldName + ": " + self.text

    def to_dict(self):
        return self.__dict__


class InvoiceModel:
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model, self.processor = donut_load_model(model_path, device)

    def get_elements(
        self, image: Image
    ) -> Tuple[Optional[List[InvoiceElement]], Optional[List[List[InvoiceElement]]]]:

        pixel_values = self.processor(image.convert("RGB"), return_tensors="pt").pixel_values

        decoder_input_ids = self.processor.tokenizer(
            "<s_unstructured-invoices>", add_special_tokens=False, return_tensors="pt"
        ).input_ids

        outputs = self.model.generate(
            pixel_values.to(self.model.device),
            decoder_input_ids=decoder_input_ids.to(self.model.device),
        )

        # process output
        prediction = self.processor.token2json(self.processor.batch_decode(outputs)[0])
        print(prediction)

        return [
            InvoiceElement(fieldName=k, text=v) for k, v in prediction.items() if k != "ItemLines"
        ], [
            [InvoiceElement(fieldName=k, text=v) for k, v in itemLine.items()]
            for itemLine in (
                prediction["ItemLines"]
                if type(prediction["ItemLines"]) is list
                else [prediction["ItemLines"]]
            )
        ] if "ItemLines" in prediction else []


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
    def from_file(cls, file: BinaryIO, filename: str, model: InvoiceModel):
        page = PageInvoice(image=Image.open(file), model=model)
        page.get_elements()
        return cls.from_pages([page])


class PageInvoice:
    def __init__(self, image: Image, model: InvoiceModel):
        self.image = image
        self.elements: Optional[List[InvoiceElement]]
        self.itemLists = Optional[List[List[InvoiceElement]]]
        self.model = model

    def __str__(self):
        return "\n\n".join(
            [str(element) for element in self.elements]
            + [" ".join([str(element) for element in itemList]) for itemList in self.itemLists]
        )

    def get_elements(
        self, inplace=True
    ) -> Optional[Tuple[Optional[List[InvoiceElement]], Optional[List[List[InvoiceElement]]]]]:
        elements, itemLists = self.model.get_elements(self.image)

        if inplace:
            self.elements = elements
            self.itemLists = itemLists
            return None

        return elements, itemLists
