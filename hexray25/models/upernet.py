import torch
import pydoc
import torch.nn as nn
import lightning as pl
from transformers.models.upernet import UperNetConfig
from transformers.models.upernet.modeling_upernet import UperNetPreTrainedModel, UperNetHead, SemanticSegmenterOutput, UperNetConvModule
from transformers import AutoBackbone
from typing import Union, Tuple, Optional, Dict
from hexray25.utils import object_from_dict, state_dict_from_disk
from hexray25.metrics import binary_mean_iou

class HuggingFaceModel(torch.nn.Module):
    def __init__(self, network, backbone, model_config):
        super().__init__()
        backbone = object_from_dict(backbone)
        model_config["backbone_config"] = backbone
        self.config = object_from_dict(model_config)
        self.model = pydoc.locate(network)(config=self.config)

    def forward(self, images):
        x = self.model(images)
        return x.logits


class UperNetConfigCustom(UperNetConfig):
    def __init__(self,
                 backbone_config=None,
                 hidden_size=512,
                 initializer_range=0.02,
                 pool_scales=[1, 2, 3, 6],
                 use_auxiliary_head=True,
                 auxiliary_loss_weight=0.4,
                 auxiliary_in_channels=384,
                 auxiliary_channels=256,
                 auxiliary_num_convs=1,
                 auxiliary_concat_input=False,
                 loss_ignore_index=255,
                 temperature=False, 
                 **kwargs,):
        super().__init__(backbone_config=backbone_config,
                         hidden_size=hidden_size,
                         initializer_range=initializer_range,
                         pool_scales=pool_scales,
                         use_auxiliary_head=use_auxiliary_head,
                         auxiliary_loss_weight=auxiliary_loss_weight,
                         auxiliary_in_channels=auxiliary_in_channels,
                         auxiliary_channels=auxiliary_channels,
                         auxiliary_num_convs=auxiliary_num_convs,
                         auxiliary_concat_input=auxiliary_concat_input,
                         loss_ignore_index=loss_ignore_index,
                         **kwargs)
        self.temperature = temperature

class UperNetFCNHeadModified(nn.Module):
    """
    Fully Convolution Networks for Semantic Segmentation. This head is the implementation of
    [FCNNet](https://arxiv.org/abs/1411.4038>).

    Args:
        config:
            Configuration.
        in_channels (int):
            Number of input channels.
        kernel_size (int):
            The kernel size for convs in the head. Default: 3.
        dilation (int):
            The dilation rate for convs in the head. Default: 1.
    """

    def __init__(
        self, config, kernel_size: int = 3, dilation: Union[int, Tuple[int, int]] = 1
    ) -> None:
        super().__init__()

        self.config = config
        self.in_channels = config.auxiliary_in_channels
        self.channels = config.auxiliary_channels
        self.num_convs = config.auxiliary_num_convs
        self.concat_input = config.auxiliary_concat_input
        self.in_index = config.auxiliary_in_index if isinstance(config.auxiliary_in_index, list) else [config.auxiliary_in_index]
        
        
        conv_padding = (kernel_size // 2) * dilation
        if not isinstance(self.in_channels, list):
            self.in_channels = [self.in_channels]
            
        
        for n, in_channels in enumerate(self.in_channels):
            convs = []
            convs.append(
                UperNetConvModule(
                    in_channels, self.channels, kernel_size=kernel_size, padding=conv_padding, dilation=dilation
                )
            )
            for i in range(self.num_convs - 1):
                convs.append(
                    UperNetConvModule(
                        self.channels, self.channels, kernel_size=kernel_size, padding=conv_padding, dilation=dilation
                    )
                )
            if self.num_convs == 0:
                setattr(self, f"head{n}_convs", nn.Identity())
            else:
                setattr(self, f"head{n}_convs", nn.Sequential(*convs))
            if self.concat_input:
                setattr(self, f"head{n}_conv_cat", UperNetConvModule(
                    in_channels + self.channels, self.channels, kernel_size=kernel_size, padding=kernel_size // 2
                ))

        self.classifier = nn.Conv2d(self.channels, config.num_labels, kernel_size=1)

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        # just take the relevant feature maps
        outputs = []
        for n, index in enumerate(self.in_index):
            hidden_states = encoder_hidden_states[index]
            output = getattr(self, f"head{n}_convs")(hidden_states)
            if self.concat_input:
                output = getattr(self, f"head{n}_conv_cat")(torch.cat([hidden_states, output], dim=1))
            outputs.append(output)
        outputs = [self.classifier(output) for output in outputs]
        return outputs

class UperNetForSemanticSegmentation(UperNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.backbone = AutoBackbone.from_config(config.backbone_config)

        # Semantic segmentation head(s)
        self.decode_head = UperNetHead(config, in_channels=self.backbone.channels)
        self.auxiliary_head = UperNetFCNHeadModified(config) if config.use_auxiliary_head else None
        self.temperature = None 
        if hasattr(config, "temperature"):
            if config.temperature:
                self.temperature = nn.Parameter(torch.ones(1) * 1.5)

        # Initialize weights and apply final processing
        self.post_init()
    
    def temp_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.shape)
        return logits / temperature
    
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> tuple:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        outputs = self.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions
        )
        features = outputs.feature_maps

        logits = self.decode_head(features)
        logits = nn.functional.interpolate(logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False)
        
        if self.temperature:
            logits = self.temp_scale(logits)

        auxiliary_logits = []
        if self.auxiliary_head is not None:
            auxiliary_out = self.auxiliary_head(features)
            for aux in auxiliary_out:
                auxiliary_logit = nn.functional.interpolate(
                    aux, size=pixel_values.shape[2:], mode="bilinear", align_corners=False
                )
                auxiliary_logits.append(auxiliary_logit)

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return output

        return SemanticSegmenterOutput(
            logits=[logits]+ auxiliary_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
class UperNetSegmenter(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)
        self.model = object_from_dict(self.hparams["model"])
        if "pretrained_weight_loc" in self.hparams.keys():
            from pretrained_hub import load_pretrained_weights
            load_pretrained_weights(self.hparams["pretrained_weight_loc"], self.model)
        
        if "infer_type" in self.hparams.keys():
            from monai.inferers import AvgMerger, PatchInferer, SlidingWindowSplitter
            self.inferer = PatchInferer(splitter=SlidingWindowSplitter(patch_size=self.hparams["infer_type"]["patch_size"], overlap=0, pad_mode=None),
                                        merger_cls=AvgMerger,
                                        match_spatial_shape=True, 
                                        batch_size=self.hparams["infer_type"]["batch_size"],
                                    )
        else:
            self.inferer=None
        
        if "resume_from_checkpoint" in self.hparams:
            corrections: Dict[str, str] = {"model.": ""}
            state_dict = state_dict_from_disk(
                file_path=self.hparams["resume_from_checkpoint"],
                rename_in_layers=corrections,
            )
            self.model.load_state_dict(state_dict)

        if "loss" in self.hparams.keys():
            self.losses = []
            for key, value in self.hparams["loss"].items():
                self.losses.append([key, value["weights"], object_from_dict(value["type"])])
        else:
            import warnings
            from pytorch_toolbelt.losses import JaccardLoss, BinaryFocalLoss
            warnings.warn("deprecated. we need loss to be defined in the yaml itself. calling default values now")
            self.losses = [
                ("jaccard", 0.1, JaccardLoss(mode="binary", from_logits=True)),
                ("focal", 0.9, BinaryFocalLoss()),
                ]
        
        self.aux_loss = True if "auxiliary_loss_weight" in self.hparams.keys() else False 
        if self.aux_loss:
            self.aux_loss_weight = self.hparams["auxiliary_loss_weight"]
        
        self.train_step_outputs = []
        self.val_step_outputs = []

    def forward(self, batch: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.model(batch)
    
    def configure_optimizers(self):
        optimizer = object_from_dict(
            self.hparams["optimizer"],
            params=[x for x in self.model.parameters() if x.requires_grad],
        )
        self.optimizers = [optimizer]
        if "scheduler" not in self.hparams.keys():
            return self.optimizers
            
        scheduler = object_from_dict(self.hparams["scheduler"], optimizer=optimizer)
        return self.optimizers, [scheduler]

    def training_step(self, batch, batch_idx):
        features, masks = batch
        #features = batch["image"]
        #masks = batch["mask"]
        #with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = self.forward(features)
        result = self._calculate_loss(logits, masks, "train")

        result["lr"] = self._get_current_lr().to(self.device)
        for k, v in result.items():
            self.log(name=k, value=v, prog_bar=True, batch_size=features.shape[0], sync_dist=True)

        self.train_step_outputs.append(result["train_loss"])
        return {"loss": result["train_loss"], "log": result}

    
    def _calculate_loss(self, logits, masks, loss_type):
        if self.aux_loss:
            assert isinstance(logits, list), "aux_outputs are not present to caluclate aux_loss."
            aux_logits = logits[1:] 
            logits = logits[0]
        
        total_loss = 0 
        result = {}
        if isinstance(logits, list):
            if len(logits)>1: raise ValueError("only logits of single instance working")
            logits = logits[0]

        for loss_name, weight, loss in self.losses:
            if logits.shape[1] == 2:
                ls_mask = loss(logits, masks.squeeze(1).long())
            else:
                ls_mask = loss(logits, masks)
            if self.aux_loss:
                logits_aux_loss = [loss(aux_logit, masks) for aux_logit in aux_logits]
                logits_aux_loss = sum(logits_aux_loss)/len(logits_aux_loss)
                ls_mask += self.aux_loss_weight*logits_aux_loss
            total_loss += weight * ls_mask 
            result[f"{loss_type}_mask_{loss_name}"] = ls_mask 
        result[f"{loss_type}_loss"] = total_loss 
        return result 

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_step_outputs).mean()
        self.log("train_epoch_loss", avg_loss, sync_dist=True)
        self.train_step_outputs.clear()

    def _get_current_lr(self) -> torch.Tensor:
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]  # type: ignore
        return torch.Tensor([lr])[0]

    def validation_step(self, batch, batch_id):
        features, masks = batch
        #features = batch["image"]
        #masks = batch["mask"]
        if self.inferer is not None:
            self.inferer.merger_kwargs = {"device": self.device}
            logits = self.inferer(features, self.model)
        else:
            logits = self.forward(features)
        result = self._calculate_loss(logits, masks, "val")
        
        for k, v in result.items():
            self.log(name=k, value=v, prog_bar=True, batch_size=features.shape[0], sync_dist=True)
        
        val_iou = self._metrics(logits, masks)
        self.val_step_outputs.extend(val_iou)
        return {"loss": result["val_loss"]}

    def _metrics(self, logits, masks):
        iou = []
        if isinstance(logits, list):
            logits = logits[0]
        for i, j in zip(logits, masks):
            if i.shape[0] == 2:
                i = i.argmax(dim=0).unsqueeze(0)
            _iou = binary_mean_iou(i[None], j[None])
            iou.append(_iou)
        return iou
    
    def on_validation_epoch_end(self):
        all_preds = torch.stack(self.val_step_outputs)
        self.log(name="val_epoch_iou", value=all_preds.mean(), prog_bar=True, sync_dist=True)
        self.val_step_outputs.clear()