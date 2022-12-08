from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Final, List, Optional, Tuple, Union, BinaryIO, Dict, Any

import cv2
import numpy as np
import pytesseract
import torch

from donut.model import DonutModel

from transformers import (
    LayoutLMModel,
    LayoutLMTokenizer,
    LayoutLMForTokenClassification,
)

from PIL import Image


DONUT_MODEL_PATH: Final = "unstructuredio/donut-invoices-fake"

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
                Name of a pretrained model name either registered in huggingface.co. or saved in
                local, e.g., `naver-clova-ix/donut-base`, or
                `naver-clova-ix/donut-base-finetuned-rvlcdip`
        """
        model = super(DonutModel, cls).from_pretrained(
            pretrained_model_name_or_path, revision="main", *model_args, **kwargs
        )

        # truncate or interplolate position embeddings of donut decoder
        max_length = kwargs.get("max_length", model.config.max_position_embeddings)
        if (
            max_length != model.config.max_position_embeddings
        ):  # if max_length of trained model differs max_length you want to train
            model.decoder.model.model.decoder.embed_positions.weight = torch.nn.Parameter(
                model.decoder.resize_bart_abs_pos_emb(
                    model.decoder.model.model.decoder.embed_positions.weight,
                    max_length + 2,
                    # https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/mbart/modeling_mbart.py#L118-L119
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

    model = DonutModelMain.from_pretrained(model_path)

    if device is not None:
        model.to(device)

    return model


def layoutlm_load_model(
    model_path: Optional[Union[str, Path]] = None,
    num_labels: Optional[int] = None,
    device: Optional[str] = None,
) -> LayoutLMModel:
    """Loads the LayoutLMModel model using the specified parameters"""
    tokenizer = LayoutLMTokenizer.from_pretrained(LAYOUTLM_MODEL_PATH)

    # num_labels = len(labels)
    model = LayoutLMForTokenClassification.from_pretrained(
        # LAYOUTLM_MODEL_PATH, num_labels=num_labels
        "unstructuredio/layoutlmv1-invoices-fake"
    )
    # model_path = "/Users/ajimeno/Documents/git/doc-layout-exploration"
    # "/invoices/invoice-training/dataset-layoutlm-generated/processed/model.torch"
    # model.load_state_dict(torch.load(model_path))

    if device is not None:
        model.to(device)

    return tokenizer, model, labels


@dataclass
class InvoiceElement:
    # Left, top, width, height
    coordinates: Optional[List[Tuple[float, float, float, float]]]
    fieldName: Optional[str] = None
    text: Optional[str] = None

    def __str__(self):
        return self.fieldName + ": " + self.text

    def to_dict(self):
        return self.__dict__


class InvoiceModel(ABC):
    @abstractmethod
    def get_elements(
        self, image: Image
    ) -> Tuple[Optional[List[InvoiceElement]], Optional[List[List[InvoiceElement]]]]:
        pass


def get_bboxes(ocr_df, width, height):
    actual_boxes = []
    coordinates = ocr_df[["left", "top", "width", "height"]]
    for idx, row in coordinates.iterrows():
        x, y, w, h = tuple(row)  # the row comes in (left, top, width, height) format
        actual_box = [
            x,
            y,
            x + w,
            y + h,
        ]  # we turn it into (left, top, left+widght, top+height) to get the actual box
        actual_boxes.append(actual_box)

    def normalize_box(box, width, height):
        return [
            int(1000 * (box[0] / width)),
            int(1000 * (box[1] / height)),
            int(1000 * (box[2] / width)),
            int(1000 * (box[3] / height)),
        ]

    boxes = []

    for box in actual_boxes:
        boxes.append(normalize_box(box, width, height))

    return actual_boxes, boxes


def convert_example_to_features(
    image,
    words,
    boxes,
    actual_boxes,
    tokenizer,
    max_sequence_length,
    cls_token_box=[0, 0, 0, 0],
    sep_token_box=[1000, 1000, 1000, 1000],
    pad_token_box=[0, 0, 0, 0],
):
    width, height = image.size

    tokens = []
    token_boxes = []
    actual_bboxes = []  # we use an extra b because actual_boxes is already used
    token_actual_boxes = []
    for word, box, actual_bbox in zip(words, boxes, actual_boxes):
        word_tokens = tokenizer.tokenize(word)
        tokens.extend(word_tokens)
        token_boxes.extend([box] * len(word_tokens))
        actual_bboxes.extend([actual_bbox] * len(word_tokens))
        token_actual_boxes.extend([actual_bbox] * len(word_tokens))

    # Truncation: account for [CLS] and [SEP] with "- 2".
    special_tokens_count = 2
    if len(tokens) > max_sequence_length - special_tokens_count:
        tokens = tokens[: (max_sequence_length - special_tokens_count)]
        token_boxes = token_boxes[: (max_sequence_length - special_tokens_count)]
        actual_bboxes = actual_bboxes[: (max_sequence_length - special_tokens_count)]
        token_actual_boxes = token_actual_boxes[: (max_sequence_length - special_tokens_count)]

    # add [SEP] token, with corresponding token boxes and actual boxes
    tokens += [tokenizer.sep_token]
    token_boxes += [sep_token_box]
    actual_bboxes += [[0, 0, width, height]]
    token_actual_boxes += [[0, 0, width, height]]

    segment_ids = [0] * len(tokens)

    # next: [CLS] token
    tokens = [tokenizer.cls_token] + tokens
    token_boxes = [cls_token_box] + token_boxes
    actual_bboxes = [[0, 0, width, height]] + actual_bboxes
    token_actual_boxes = [[0, 0, width, height]] + token_actual_boxes
    segment_ids = [1] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_sequence_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    input_mask += [0] * padding_length
    segment_ids += [tokenizer.pad_token_id] * padding_length
    token_boxes += [pad_token_box] * padding_length
    token_actual_boxes += [pad_token_box] * padding_length

    assert len(input_ids) == max_sequence_length
    assert len(input_mask) == max_sequence_length
    assert len(segment_ids) == max_sequence_length
    # assert len(label_ids) == max_sequence_lengthh
    assert len(token_boxes) == max_sequence_length
    assert len(token_actual_boxes) == max_sequence_length

    return input_ids, input_mask, segment_ids, token_boxes, token_actual_boxes


labels = [
    "B-AMOUNT",
    "B-AMOUNTDUE",
    "B-CUSTOMERADDRESS",
    "B-CUSTOMERNAME",
    "B-DESCRIPTION",
    "B-INVOICEID",
    "B-QUANTITY",
    "B-UNITPRICE",
    "B-VENDORADDRESS",
    "B-VENDORNAME",
    "E-AMOUNT",
    "E-AMOUNTDUE",
    "E-CUSTOMERADDRESS",
    "E-CUSTOMERNAME",
    "E-DESCRIPTION",
    "E-INVOICEID",
    "E-QUANTITY",
    "E-UNITPRICE",
    "E-VENDORADDRESS",
    "E-VENDORNAME",
    "I-AMOUNT",
    "I-CUSTOMERADDRESS",
    "I-CUSTOMERNAME",
    "I-DESCRIPTION",
    "I-QUANTITY",
    "I-UNITPRICE",
    "I-VENDORADDRESS",
    "I-VENDORNAME",
    "S-AMOUNT",
    "S-CUSTOMERNAME",
    "S-DESCRIPTION",
    "S-INVOICEDATE",
    "S-INVOICETOTAL",
    "S-OTHER",
    "S-QUANTITY",
    "S-UNITPRICE",
    "S-VENDORNAME",
]


class InvoiceModelLayoutLM(InvoiceModel):
    def __init__(self, model_file_name: Optional[str] = None, device: Optional[str] = None):
        self.tokenizer, self.model, self.labels = layoutlm_load_model(
            model_file_name, device=device
        )
        self.device = device

    def get_elements(
        self, image: Image
    ) -> Tuple[Optional[List[InvoiceElement]], Optional[List[List[InvoiceElement]]]]:
        # process image with tesseract/paddleocr
        zoom = 6
        img = cv2.resize(
            cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR),
            None,
            fx=zoom,
            fy=zoom,
            interpolation=cv2.INTER_CUBIC,
        )

        width, height = image.size
        width *= zoom
        height *= zoom
        w_scale = 1000 / width
        h_scale = 1000 / height

        kernel = np.ones((1, 1), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)

        ocr_df = pytesseract.image_to_data(Image.fromarray(img), output_type="data.frame")

        ocr_df = ocr_df.dropna().assign(
            left_scaled=ocr_df.left * w_scale,
            width_scaled=ocr_df.width * w_scale,
            top_scaled=ocr_df.top * h_scale,
            height_scaled=ocr_df.height * h_scale,
            right_scaled=lambda x: x.left_scaled + x.width_scaled,
            bottom_scaled=lambda x: x.top_scaled + x.height_scaled,
        )

        float_cols = ocr_df.select_dtypes("float").columns
        ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
        ocr_df = ocr_df.replace(r"^\s*$", np.nan, regex=True)
        ocr_df = ocr_df.dropna().reset_index(drop=True)

        words = list(ocr_df.text)
        actual_boxes, boxes = get_bboxes(ocr_df, width, height)

        # prepare data for layoutlm
        (
            input_ids,
            input_mask,
            segment_ids,
            token_boxes,
            token_actual_boxes,
        ) = convert_example_to_features(
            image=image,
            words=words,
            boxes=boxes,
            actual_boxes=actual_boxes,
            tokenizer=self.tokenizer,
            max_sequence_length=512,
        )

        # run layoutlm
        input_ids = torch.tensor(input_ids, device=self.device).unsqueeze(0)
        attention_mask = torch.tensor(input_mask, device=self.device).unsqueeze(0)
        token_type_ids = torch.tensor(segment_ids, device=self.device).unsqueeze(0)
        bbox = torch.tensor(token_boxes, device=self.device).unsqueeze(0)
        print(bbox)
        outputs = self.model(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Translating output into invoice elements
        token_predictions = outputs.logits.argmax(-1).squeeze().tolist()

        word_level_predictions = []  # let's turn them into word level predictions
        final_boxes = []
        tokens: List[str] = []
        for id, token_pred, box in zip(
            input_ids[0].squeeze().tolist(),
            token_predictions,
            token_actual_boxes,
        ):
            if (self.tokenizer.decode([id]).startswith("##")) or (
                id
                in [
                    self.tokenizer.cls_token_id,
                    self.tokenizer.sep_token_id,
                    self.tokenizer.pad_token_id,
                ]
            ):
                if self.tokenizer.decode([id]).startswith("##"):
                    tokens[-1] = tokens[-1] + self.tokenizer.decode([id])[2:]
            else:
                word_level_predictions.append(token_pred)
                final_boxes.append(box)
                tokens.append(self.tokenizer.decode([id]))

        plabels = [self.labels[label] for label in word_level_predictions]

        invoice_components = []

        alabel = None
        atext = None
        abox = None

        for t, l, b in zip(tokens, plabels, final_boxes):
            tlabel = l.split("-")[-1]
            if alabel is None:
                alabel = tlabel
                atext = t
                abox = b
            else:
                if l[0] == "E":
                    if tlabel == alabel:
                        atext += " " + t
                        abox = [
                            min(abox[0], b[0]),
                            min(abox[1], b[1]),
                            max(abox[2], b[2]),
                            max(abox[3], b[3]),
                        ]

                        invoice_components.append({"label": alabel, "text": atext, "bbox": abox})
                    else:
                        invoice_components.append({"label": alabel, "text": atext, "bbox": abox})

                        invoice_components.append({"label": tlabel, "text": t, "bbox": b})

                    alabel = None
                    atext = None
                    abox = None
                elif l[0] == "S":
                    if alabel is not None:
                        if alabel == tlabel:
                            atext += " " + t
                            abox = [
                                min(abox[0], b[0]),
                                min(abox[1], b[1]),
                                max(abox[2], b[2]),
                                max(abox[3], b[3]),
                            ]
                        else:
                            invoice_components.append(
                                {"label": alabel, "text": atext, "bbox": abox}
                            )

                            alabel = tlabel
                            atext = t
                            abox = b
                    else:
                        alabel = tlabel
                        atext = t
                        abox = b
                else:
                    if tlabel == alabel:
                        atext += " " + t
                        abox = [
                            min(abox[0], b[0]),
                            min(abox[1], b[1]),
                            max(abox[2], b[2]),
                            max(abox[3], b[3]),
                        ]
                    else:
                        invoice_components.append({"label": alabel, "text": atext, "bbox": abox})

                        alabel = tlabel
                        atext = t
                        abox = b

        if alabel is not None:
            invoice_components.append({"label": alabel, "text": atext, "bbox": abox})

        # NOTE(alan): Narrow this type
        dfields: Dict[str, Any] = {}

        position = None
        # NOTE(alan): Narrow this type
        items = []
        item: Optional[Dict[str, Any]] = None
        for d in invoice_components:
            if d["label"] != "OTHER":
                if d["label"] in [
                    "ITEMS",
                    "DESCRIPTION",
                    "UNITPRICE",
                    "QUANTITY",
                    "AMOUNT",
                ]:
                    if position is None:
                        position = d["bbox"][1]
                        item = {}
                        items.append(item)

                    if position >= d["bbox"][1] - 6 and position <= d["bbox"][1] + 6:
                        pass
                    else:
                        item = {}
                        items.append(item)
                        position = d["bbox"][1]

                    if d["label"] in item:
                        item[d["label"]] += " " + d["text"]
                    else:
                        item[d["label"]] = d["text"]
                else:
                    if d["label"] in dfields:
                        dfields[d["label"]] += " " + d["text"]
                    else:
                        dfields[d["label"]] = d["text"]

        dfields["items"] = items

        print(dfields)

        prediction = {c[0]: c[1] for c in sorted(json.loads(json.dumps(dfields).lower()).items())}

        return [
            InvoiceElement(coordinates=None, fieldName=k, text=v)
            for k, v in prediction.items()
            if k != "items"
        ], [
            [InvoiceElement(coordinates=None, fieldName=k, text=v) for k, v in itemLine.items()]
            for itemLine in (
                prediction["items"] if type(prediction["items"]) is list else [prediction["items"]]
            )
        ]


class InvoiceModelDonut(InvoiceModel):
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model = donut_load_model(model_path, device)

    def get_elements(
        self, image: Image
    ) -> Tuple[Optional[List[InvoiceElement]], Optional[List[List[InvoiceElement]]]]:
        prediction = self.model.inference(image, prompt="<s_dataset-donut-generated>",)[
            "predictions"
        ][0]

        return [
            InvoiceElement(coordinates=None, fieldName=k, text=v)
            for k, v in prediction.items()
            if k != "items"
        ], [
            [InvoiceElement(coordinates=None, fieldName=k, text=v) for k, v in itemLine.items()]
            for itemLine in (
                prediction["items"] if type(prediction["items"]) is list else [prediction["items"]]
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
    def from_file(cls, file: BinaryIO, filename: str, model: InvoiceModel):
        # logger.info(f"Reading invoice image for file: {filename} ...")

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
