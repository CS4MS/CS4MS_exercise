from __future__ import annotations

import math
import tempfile
import urllib.request
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, unique
from functools import partial
from pathlib import Path
from typing import Any, Callable, Generator, List, Optional, Sequence, Set, Tuple, Type, Union
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from timm.models.layers import DropPath, Mlp, trunc_normal_
from torchvision.datasets.utils import download_url
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor
from transformers import AutoModel, AutoTokenizer


def get_module_device(module: torch.nn.Module) -> torch.device:
    device = next(module.parameters()).device  # type: ignore[arg-type]
    assert isinstance(device, torch.device)
    return device


@dataclass
class ImageModelOutput:
    img_embedding: torch.Tensor
    patch_embeddings: torch.Tensor
    projected_global_embedding: torch.Tensor
    class_logits: torch.Tensor
    projected_patch_embeddings: torch.Tensor


@unique
class ImageEncoderType(str, Enum):
    RESNET18 = "resnet18"
    RESNET50 = "resnet50"
    RESNET18_MULTI_IMAGE = "resnet18_multi_image"
    RESNET50_MULTI_IMAGE = "resnet50_multi_image"

    @classmethod
    def get_members(cls, multi_image_encoders_only: bool) -> List["ImageEncoderType"]:
        if multi_image_encoders_only:
            return [cls.RESNET18_MULTI_IMAGE, cls.RESNET50_MULTI_IMAGE]
        return [member for member in cls]


@unique
class ImageEncoderWeightTypes(str, Enum):
    RANDOM = "random"
    IMAGENET = "imagenet"
    BIOVIL = "biovil"
    BIOVIL_T = "biovil_t"


class MLP(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None, use_1x1_convs: bool = False
    ) -> None:
        super().__init__()

        if use_1x1_convs:
            linear_proj_1_args = {
                "in_channels": input_dim,
                "out_channels": hidden_dim,
                "kernel_size": 1,
                "bias": False,
            }
            linear_proj_2_args = {
                "in_channels": hidden_dim,
                "out_channels": output_dim,
                "kernel_size": 1,
                "bias": True,
            }
            normalisation_layer: Callable[..., nn.Module] = nn.BatchNorm2d
            projection_layer: Callable[..., nn.Module] = nn.Conv2d
        else:
            linear_proj_1_args = {"in_features": input_dim, "out_features": hidden_dim, "bias": False}
            linear_proj_2_args = {"in_features": hidden_dim, "out_features": output_dim, "bias": True}
            normalisation_layer = nn.BatchNorm1d
            projection_layer = nn.Linear

        self.output_dim = output_dim
        self.input_dim = input_dim
        if hidden_dim is not None:
            self.model = nn.Sequential(
                projection_layer(**linear_proj_1_args),
                normalisation_layer(hidden_dim),
                nn.ReLU(inplace=True),
                projection_layer(**linear_proj_2_args),
            )
        else:
            self.model = nn.Linear(input_dim, output_dim)  # type: ignore[arg-type]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MultiTaskModel(nn.Module):
    def __init__(self, input_dim: int, classifier_hidden_dim: Optional[int], num_classes: int, num_tasks: int):
        super().__init__()
        self.num_classes = num_classes
        self.num_tasks = num_tasks

        for task in range(num_tasks):
            setattr(self, "fc_" + str(task), MLP(input_dim, output_dim=num_classes, hidden_dim=classifier_hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        out = torch.zeros((batch_size, self.num_classes, self.num_tasks), dtype=x.dtype, device=x.device)
        for task in range(self.num_tasks):
            classifier = getattr(self, "fc_" + str(task))
            out[:, :, task] = classifier(x)
        return out


TypeSkipConnections = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class ResNetHIML(ResNet):
    def forward(
        self, x: torch.Tensor, return_intermediate_layers: bool = False
    ) -> Union[torch.Tensor, TypeSkipConnections]:
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)

        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        if return_intermediate_layers:
            return x0, x1, x2, x3, x4
        return x4


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> ResNetHIML:
    del arch, progress
    model = ResNetHIML(block=block, layers=layers, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetHIML:
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetHIML:
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


@dataclass
class MultiHeadAttentionOutput:
    mha_output: torch.Tensor
    attention: Optional[torch.Tensor] = None


class VisionTransformerPooler(nn.Module):
    def __init__(
        self,
        input_dim: int,
        grid_shape: Tuple[int, int],
        num_heads: int = 8,
        num_blocks: int = 3,
        norm_layer: Any = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()

        block_kwargs = dict(
            dim=input_dim,
            num_heads=num_heads,
            mlp_ratio=1.0,
            drop=0.10,
            attn_drop=0.10,
            drop_path=0.25,
            act_layer=nn.GELU,
            norm_layer=norm_layer,
        )
        self.blocks = nn.ModuleList([Block(**block_kwargs) for _ in range(num_blocks)])
        self.norm_post = norm_layer(input_dim)
        self.grid_shape = grid_shape
        self.num_patches = grid_shape[0] * grid_shape[1]

        self.type_embed = nn.Parameter(torch.zeros(2, 1, input_dim))
        trunc_normal_(self.type_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=0.10)
        pos_embed_class = SinePositionEmbedding(embedding_dim=input_dim // 2, normalize=True)
        pos_embed = pos_embed_class(mask=torch.ones([1, grid_shape[0], grid_shape[1]]))
        self.register_buffer("pos_embed", pos_embed, persistent=False)

        self.apply(self._init_weights)

    def no_weight_decay(self) -> Set[str]:
        return {"type_embed"}

    def forward(self, current_image: torch.Tensor, previous_image: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, channels, height, width = current_image.shape
        assert height == self.grid_shape[0] and width == self.grid_shape[1], "Input and grid shapes do not match"

        if previous_image is not None:
            assert previous_image.shape == current_image.shape, "current_image and previous_image shapes do not match"
            previous_image = previous_image.view(batch_size, channels, height * width).transpose(1, 2)
        current_image = current_image.view(batch_size, channels, height * width).transpose(1, 2)
        pos_embed = self.pos_embed.repeat(batch_size, 1, 1)  # type: ignore[union-attr]

        token_features = self.forward_after_reshape(x=current_image, pos_embed=pos_embed, x_previous=previous_image)
        current_patch_features = token_features[:, : self.num_patches].transpose(1, 2).view(batch_size, channels, height, width)
        return current_patch_features

    def forward_after_reshape(
        self, x: torch.Tensor, pos_embed: torch.Tensor, x_previous: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        type_embed = self.type_embed[0].expand(batch_size, sequence_length, -1)
        if x_previous is not None:
            x = torch.cat((x, x_previous), dim=1)
            pos_embed = torch.cat((pos_embed, pos_embed), dim=1)
            prev_type_embed = self.type_embed[1].expand(batch_size, sequence_length, -1)
            type_embed = torch.cat((type_embed, prev_type_embed), dim=1)

        pos_and_type_embed = pos_embed + type_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x=x, pos_and_type_embed=pos_and_type_embed)
        return self.norm_post(x)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)


class MultiHeadAttentionLayer(nn.Module):
    def __init__(
        self, dim: int, num_heads: int = 8, qkv_bias: bool = False, attn_drop: float = 0.0, proj_drop: float = 0.0
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0, f"The embedding dim ({dim}) must be divisible by the number of heads ({num_heads})"
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.return_attention = False

        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, k: torch.Tensor, q: torch.Tensor, v: torch.Tensor) -> MultiHeadAttentionOutput:
        batch_size, sequence_length, channels = v.shape
        assert channels % self.num_heads == 0

        w_q = self.proj_q(q).reshape(batch_size, sequence_length, self.num_heads, channels // self.num_heads).permute(0, 2, 1, 3)
        w_k = self.proj_k(k).reshape(batch_size, sequence_length, self.num_heads, channels // self.num_heads).permute(0, 2, 1, 3)
        w_v = self.proj_v(v).reshape(batch_size, sequence_length, self.num_heads, channels // self.num_heads).permute(0, 2, 1, 3)

        attn = (w_q @ w_k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))

        output = (attn @ w_v).transpose(1, 2).reshape(batch_size, sequence_length, channels)
        output = self.proj(output)
        output = self.proj_drop(output)

        attention_output = attn if self.return_attention else None
        return MultiHeadAttentionOutput(mha_output=output, attention=attention_output)

    def forward_as_mhsa(self, input_tensor: torch.Tensor) -> MultiHeadAttentionOutput:
        return self(k=input_tensor, q=input_tensor, v=input_tensor)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 1.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttentionLayer(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def with_pos_and_type_embed(self, tensor: torch.Tensor, emb: Optional[torch.Tensor]) -> torch.Tensor:
        return tensor if emb is None else tensor + emb

    def forward(self, x: torch.Tensor, pos_and_type_embed: Optional[torch.Tensor]) -> torch.Tensor:
        x_with_emb = self.with_pos_and_type_embed(self.norm1(x), emb=pos_and_type_embed)
        x = x + self.drop_path(self.attn.forward_as_mhsa(x_with_emb).mha_output)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SinePositionEmbedding:
    def __init__(
        self, embedding_dim: int = 64, temperature: int = 10000, normalize: bool = False, scale: Optional[float] = None
    ) -> None:
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = 2 * math.pi if scale is None else scale

    def __call__(self, mask: torch.Tensor) -> torch.Tensor:
        batch_size, height, width = mask.shape
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale

        dim_t = torch.arange(self.embedding_dim, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.embedding_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).view(batch_size, height * width, self.embedding_dim * 2)


DEFAULT_DILATION_VALUES_FOR_RESNET = (False, False, True)
ImageEncoderOutputType = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


class ImageEncoder(nn.Module):
    def __init__(self, img_encoder_type: str):
        super().__init__()
        self.img_encoder_type = img_encoder_type
        self.encoder = self._create_encoder()

    def _create_encoder(self, **kwargs: Any) -> nn.Module:
        if self.img_encoder_type in [ImageEncoderType.RESNET18, ImageEncoderType.RESNET18_MULTI_IMAGE]:
            encoder_class = resnet18
        elif self.img_encoder_type in [ImageEncoderType.RESNET50, ImageEncoderType.RESNET50_MULTI_IMAGE]:
            encoder_class = resnet50
        else:
            supported = ImageEncoderType.get_members(multi_image_encoders_only=False)
            raise NotImplementedError(f'Image encoder type "{self.img_encoder_type}" must be in {supported}')

        return encoder_class(pretrained=False, **kwargs)

    def forward(self, current_image: torch.Tensor, return_patch_embeddings: bool = False) -> ImageEncoderOutputType:
        patch_emb = self.encoder(current_image)
        avg_pooled_emb = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(patch_emb, (1, 1)), 1)
        if return_patch_embeddings:
            return patch_emb, avg_pooled_emb
        return avg_pooled_emb

    def reload_encoder_with_dilation(self, replace_stride_with_dilation: Optional[Sequence[bool]] = None) -> None:
        if self.img_encoder_type == ImageEncoderType.RESNET18:
            raise NotImplementedError("resnet18 does not support dilated convolutions")

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = DEFAULT_DILATION_VALUES_FOR_RESNET

        device = next(self.encoder.parameters()).device
        new_encoder = self._create_encoder(replace_stride_with_dilation=replace_stride_with_dilation).to(device)
        new_encoder.train(self.encoder.training)
        new_encoder.load_state_dict(self.encoder.state_dict())
        self.encoder = new_encoder


class MultiImageEncoder(ImageEncoder):
    def __init__(self, img_encoder_type: str):
        super().__init__(img_encoder_type)

        output_dim = 256
        grid_shape = (14, 14)

        backbone_output_feature_dim = get_encoder_output_dim(self.encoder, device=get_module_device(self))
        self.backbone_to_vit = nn.Conv2d(
            in_channels=backbone_output_feature_dim,
            out_channels=output_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.vit_pooler = VisionTransformerPooler(input_dim=output_dim, grid_shape=grid_shape)
        self.missing_previous_emb = nn.Parameter(torch.zeros(1, output_dim, 1, 1))
        trunc_normal_(self.missing_previous_emb, std=0.02)

    def forward(
        self,
        current_image: torch.Tensor,
        previous_image: Optional[torch.Tensor] = None,
        return_patch_embeddings: bool = False,
    ) -> ImageEncoderOutputType:
        batch_size = current_image.shape[0]

        if previous_image is not None:
            assert current_image.shape == previous_image.shape
            x = torch.cat([current_image, previous_image], dim=0)
            x = super().forward(x, return_patch_embeddings=True)[0]
            x = self.backbone_to_vit(x)
            patch_x, patch_x_previous = x[:batch_size], x[batch_size:]
            diff_x = self.vit_pooler(current_image=patch_x, previous_image=patch_x_previous)
        else:
            x = super().forward(current_image, return_patch_embeddings=True)[0]
            patch_x = self.backbone_to_vit(x)
            _, _, width, height = patch_x.shape
            diff_x = self.missing_previous_emb.repeat(batch_size, 1, width, height)

        patch_fused = torch.cat([patch_x, diff_x], dim=1)
        avg_pooled_emb = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(patch_fused, (1, 1)), 1)

        if return_patch_embeddings:
            return patch_fused, avg_pooled_emb
        return avg_pooled_emb

    def reload_encoder_with_dilation(self, replace_stride_with_dilation: Optional[Sequence[bool]] = None) -> None:
        del replace_stride_with_dilation
        raise NotImplementedError


@torch.no_grad()
def get_encoder_output_dim(module: torch.nn.Module, device: torch.device) -> int:
    x = torch.rand((1, 3, 448, 448)).to(device)
    with restore_training_mode(module):
        module.eval()
        representations = module(x)
    return representations.shape[1]


@contextmanager
def restore_training_mode(module: nn.Module) -> Generator[None, None, None]:
    training_mode = module.training
    yield
    module.train(mode=training_mode)


def get_encoder_from_type(img_encoder_type: str) -> ImageEncoder:
    if img_encoder_type in ImageEncoderType.get_members(multi_image_encoders_only=True):
        return MultiImageEncoder(img_encoder_type=img_encoder_type)
    return ImageEncoder(img_encoder_type=img_encoder_type)


class BaseImageModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> ImageModelOutput:
        raise NotImplementedError

    @abstractmethod
    def get_patchwise_projected_embeddings(self, input_img: torch.Tensor, normalize: bool) -> torch.Tensor:
        raise NotImplementedError


class ImageModel(BaseImageModel):
    def __init__(
        self,
        img_encoder_type: str,
        joint_feature_size: int,
        freeze_encoder: bool = False,
        pretrained_model_path: Optional[Union[str, Path]] = None,
        **downstream_classifier_kwargs: Any,
    ):
        super().__init__()
        self.encoder = get_encoder_from_type(img_encoder_type)
        self.feature_size = get_encoder_output_dim(self.encoder, device=get_module_device(self.encoder))
        self.projector = MLP(
            input_dim=self.feature_size,
            output_dim=joint_feature_size,
            hidden_dim=joint_feature_size,
            use_1x1_convs=True,
        )
        self.downstream_classifier_kwargs = downstream_classifier_kwargs
        self.classifier = self.create_downstream_classifier() if downstream_classifier_kwargs else None
        self.freeze_encoder = freeze_encoder
        self.train()

        if pretrained_model_path is not None:
            if not isinstance(pretrained_model_path, (str, Path)):
                raise TypeError(f"Expected a string or Path, got {type(pretrained_model_path)}")
            state_dict = torch.load(pretrained_model_path, map_location="cpu")
            self.load_state_dict(state_dict)

    def train(self, mode: bool = True) -> Any:
        super().train(mode=mode)
        if self.freeze_encoder:
            self.encoder.train(mode=False)
            self.projector.train(mode=False)
        return self

    def forward(self, x: torch.Tensor) -> ImageModelOutput:
        with torch.set_grad_enabled(not self.freeze_encoder):
            patch_x, pooled_x = self.encoder(x, return_patch_embeddings=True)
        return self.forward_post_encoder(patch_x, pooled_x)

    def forward_post_encoder(self, patch_x: torch.Tensor, pooled_x: torch.Tensor) -> ImageModelOutput:
        with torch.set_grad_enabled(not self.freeze_encoder):
            projected_patch_embeddings = self.projector(patch_x)
            projected_global_embedding = torch.mean(projected_patch_embeddings, dim=(2, 3))

        logits = self.classifier(pooled_x) if self.classifier else None
        return ImageModelOutput(
            img_embedding=pooled_x,
            patch_embeddings=patch_x,
            class_logits=logits,
            projected_patch_embeddings=projected_patch_embeddings,
            projected_global_embedding=projected_global_embedding,
        )

    def create_downstream_classifier(self, **kwargs: Any) -> MultiTaskModel:
        downstream_classifier_kwargs = kwargs if kwargs else self.downstream_classifier_kwargs
        return MultiTaskModel(self.feature_size, **downstream_classifier_kwargs)

    @torch.no_grad()
    def get_patchwise_projected_embeddings(self, input_img: torch.Tensor, normalize: bool) -> torch.Tensor:
        assert not self.training, "This function is only implemented for evaluation mode"
        outputs = self.forward(input_img)
        projected_embeddings = outputs.projected_patch_embeddings.detach()
        if normalize:
            projected_embeddings = F.normalize(projected_embeddings, dim=1)
        return projected_embeddings.permute([0, 2, 3, 1])


class MultiImageModel(ImageModel):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert isinstance(self.encoder, MultiImageEncoder), "MultiImageModel only supports MultiImageEncoder"

    def forward(self, current_image: torch.Tensor, previous_image: Optional[torch.Tensor] = None) -> ImageModelOutput:
        with torch.set_grad_enabled(not self.freeze_encoder):
            patch_x, pooled_x = self.encoder(
                current_image=current_image, previous_image=previous_image, return_patch_embeddings=True
            )
        return self.forward_post_encoder(patch_x, pooled_x)


JOINT_FEATURE_SIZE = 128
BIOMED_VLP_CXR_BERT_SPECIALIZED = "microsoft/BiomedVLP-CXR-BERT-specialized"
BIOMED_VLP_BIOVIL_T = "microsoft/BiomedVLP-BioViL-T"
HF_URL = "https://huggingface.co"
CXR_BERT_COMMIT_TAG = "v1.1"
BIOVIL_T_COMMIT_TAG = "v1.0"
BIOVIL_IMAGE_WEIGHTS_NAME = "biovil_image_resnet50_proj_size_128.pt"
BIOVIL_IMAGE_WEIGHTS_URL = (
    f"{HF_URL}/{BIOMED_VLP_CXR_BERT_SPECIALIZED}/resolve/{CXR_BERT_COMMIT_TAG}/{BIOVIL_IMAGE_WEIGHTS_NAME}"
)
BIOVIL_IMAGE_WEIGHTS_MD5 = "02ce6ee460f72efd599295f440dbb453"
BIOVIL_T_IMAGE_WEIGHTS_NAME = "biovil_t_image_model_proj_size_128.pt"
BIOVIL_T_IMAGE_WEIGHTS_URL = (
    f"{HF_URL}/{BIOMED_VLP_BIOVIL_T}/resolve/{BIOVIL_T_COMMIT_TAG}/{BIOVIL_T_IMAGE_WEIGHTS_NAME}"
)
BIOVIL_T_IMAGE_WEIGHTS_MD5 = "a83080e2f23aa584a4f2b24c39b1bb64"


def _download_biovil_image_model_weights() -> Path:
    root_dir = tempfile.gettempdir()
    download_url(
        BIOVIL_IMAGE_WEIGHTS_URL,
        root=root_dir,
        filename=BIOVIL_IMAGE_WEIGHTS_NAME,
        md5=BIOVIL_IMAGE_WEIGHTS_MD5,
    )
    return Path(root_dir, BIOVIL_IMAGE_WEIGHTS_NAME)


def _download_biovil_t_image_model_weights() -> Path:
    root_dir = tempfile.gettempdir()
    download_url(
        BIOVIL_T_IMAGE_WEIGHTS_URL,
        root=root_dir,
        filename=BIOVIL_T_IMAGE_WEIGHTS_NAME,
        md5=BIOVIL_T_IMAGE_WEIGHTS_MD5,
    )
    return Path(root_dir, BIOVIL_T_IMAGE_WEIGHTS_NAME)


def get_biovil_image_encoder(pretrained: bool = True) -> ImageModel:
    resnet_checkpoint_path = _download_biovil_image_model_weights() if pretrained else None
    return ImageModel(
        img_encoder_type=ImageEncoderType.RESNET50,
        joint_feature_size=JOINT_FEATURE_SIZE,
        pretrained_model_path=resnet_checkpoint_path,
    )


def get_biovil_t_image_encoder() -> ImageModel:
    biovilt_checkpoint_path = _download_biovil_t_image_model_weights()
    return ImageModel(
        img_encoder_type=ImageEncoderType.RESNET50_MULTI_IMAGE,
        joint_feature_size=JOINT_FEATURE_SIZE,
        pretrained_model_path=biovilt_checkpoint_path,
    )


class ImageEncoderBioVil(nn.Module):
    def __init__(self, backbone: str = "biovil"):
        super().__init__()
        if backbone == ImageEncoderWeightTypes.BIOVIL:
            self.backbone = get_biovil_image_encoder()
        elif backbone == ImageEncoderWeightTypes.BIOVIL_T:
            self.backbone = get_biovil_t_image_encoder()
        else:
            raise ValueError(f"Weights option not found: {backbone}")

    def forward(self, image: torch.Tensor) -> ImageModelOutput:
        return self.backbone(image)


def print_red(text: str) -> None:
    print("\033[91m" + text + "\033[0m")


def print_green(text: str) -> None:
    print("\033[92m" + text + "\033[0m")


def load_biovil_image_components(backbone: str = "biovil_t") -> Tuple[ImageEncoderBioVil, Compose, torch.device, Callable[[List[Image.Image]], torch.Tensor]]:
    image_transforms = Compose([Resize(512, antialias=True), CenterCrop(512), ToTensor()])
    image_encoder = ImageEncoderBioVil(backbone)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print_red("No GPU accelerator found, falling back to CPU - model inference will be slow")
    else:
        print_green("Running on GPU! (fast inference)")

    image_encoder.eval()
    image_encoder.to(device)

    def get_image_embeddings(images: List[Image.Image]) -> torch.Tensor:
        with torch.no_grad():
            transformed_images = torch.stack([image_transforms(image) for image in images])
            image_model_output = image_encoder(transformed_images.to(device))
            return image_model_output.projected_global_embedding

    return image_encoder, image_transforms, device, get_image_embeddings


def load_biovil_text_components(
    device: torch.device, url: str = "microsoft/BiomedVLP-BioViL-T"
) -> Tuple[Any, Any, Callable[[List[str]], torch.Tensor]]:
    tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True)
    text_encoder = AutoModel.from_pretrained(
        url,
        trust_remote_code=True,
        device_map=device
    )
    text_encoder.eval()

    def get_text_embeddings(text_prompts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            tokenizer_output = tokenizer.batch_encode_plus(
                batch_text_or_text_pairs=text_prompts,
                add_special_tokens=True,
                padding="longest",
                return_tensors="pt",
            ).to(device)
            text_embeddings = text_encoder(
                input_ids=tokenizer_output.input_ids,
                attention_mask=tokenizer_output.attention_mask,
                output_cls_projected_embedding=True,
            )
            return text_embeddings.cls_projected_embedding

    return tokenizer, text_encoder, get_text_embeddings


def calculate_cosine_similarity(embedding_1: torch.Tensor, embedding_2: torch.Tensor) -> torch.Tensor:
    return F.normalize(embedding_1) @ F.normalize(embedding_2).T


def load_image(url: str, cache_directory: str = "images") -> Image.Image:
    parsed_url = urlparse(url)
    image_name = Path(parsed_url.path).name
    cache_path = Path(cache_directory)
    cache_path.mkdir(exist_ok=True)
    image_path = cache_path / image_name
    if not image_path.exists():
        with urllib.request.urlopen(url) as response, image_path.open("wb") as image_file:
            image_file.write(response.read())
    return Image.open(image_path)


def plot_similarities(
    similarities: torch.Tensor,
    text_prompts: List[str],
    images: List[Image.Image],
    image_descriptions: List[str],
    plot_probabilities: bool = False,
    size_unit: float = 1,
) -> None:
    if plot_probabilities:
        title = "Softmax probabilities"
        matrix = similarities.softmax(dim=-1)
        fmt = ".0%"
    else:
        title = "Cosine similarities"
        matrix = similarities
        fmt = ".2f"

    display_prompts = [prompt.replace(". ", ".\n") if len(prompt) > 40 else prompt for prompt in text_prompts]
    dataframe = pd.DataFrame(matrix.T.cpu().detach().numpy(), columns=image_descriptions, index=display_prompts)
    fig, ax = plt.subplots(
        figsize=(size_unit + (size_unit * len(images)), size_unit + size_unit * len(text_prompts))
    )
    heatmap = sns.heatmap(
        dataframe,
        annot=True,
        cbar=False,
        cmap="RdYlGn",
        vmin=0 if plot_probabilities else -1,
        vmax=1,
        linewidths=0.5,
        linecolor="black",
        square=True,
        annot_kws={"fontsize": 12},
        fmt=fmt,
    )
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
    heatmap.set_xlabel("Patient images", weight="bold")
    heatmap.set_ylabel("Text prompts", weight="bold")

    x_positions = ax.get_xticks()
    for i, img in enumerate(images):
        imagebox = OffsetImage(img, zoom=0.11 * size_unit)
        annotation = AnnotationBbox(imagebox, (x_positions[i], 0), box_alignment=(0.5, -0.05), frameon=False)
        ax.add_artist(annotation)

    fig.text(0.5, 1.2, title, fontsize=12, weight="bold", ha="center")
    plt.show()


class Patient:
    def __init__(self, image_url: str, report: str):
        self.image = load_image(image_url)
        parsed_url = urlparse(image_url)
        self.patient_id = Path(parsed_url.path).stem
        self.keywords = parsed_url.query.split("=")[1].replace("%20", " ")
        self.url = "https://openi.nlm.nih.gov/detailedresult?img=" + self.patient_id
        self.report = report

    def __str__(self) -> str:
        return f"""
Patient ID: {self.patient_id}
Patient URL: {self.url}

RADIOLOGY REPORT
-------------------------------------------
{self.report}
-------------------------------------------
\n\n
"""

    def __repr__(self) -> str:
        display(self.image)
        return self.__str__()


def build_similarity_function(
    get_text_embeddings: Callable[[List[str]], torch.Tensor],
    get_image_embeddings: Callable[[List[Image.Image]], torch.Tensor],
) -> Callable[[List[str], List[Image.Image]], torch.Tensor]:
    def get_similarities_from_text_and_images(text_prompts: List[str], images: List[Image.Image]) -> torch.Tensor:
        text_embeddings = get_text_embeddings(text_prompts)
        image_embeddings = get_image_embeddings(images)
        return calculate_cosine_similarity(image_embeddings, text_embeddings)

    return get_similarities_from_text_and_images


def generate_report(
    image: Image.Image,
    reporting_template: Union[dict, list],
    prompts: dict,
    similarity_fn: Callable[[List[str], List[Image.Image]], torch.Tensor],
) -> List[str]:
    report_sentences: List[str] = []
    if isinstance(reporting_template, dict):
        choices = list(reporting_template.keys())
        text_prompts = [prompts[key] for key in choices]
        similarities = similarity_fn(text_prompts, [image])
        decision = choices[similarities.argmax()]
        decision_content = reporting_template[decision]
        if isinstance(decision_content, str):
            report_sentences.append(decision_content)
        else:
            report_sentences.extend(generate_report(image, decision_content, prompts, similarity_fn))
    elif isinstance(reporting_template, list):
        for sub_report in reporting_template:
            report_sentences.extend(generate_report(image, sub_report, prompts, similarity_fn))
    return report_sentences
