import numpy as np
import torch
import torch.nn as nn

from model import YOLOv3
from utils import extract_layers


class LoadYOLOWeights:

    def __init__(self, config_file_path: str, weights_file_path: str):
        self.config_file_path = config_file_path
        self.weights_file_path = weights_file_path

    def _parse_cfg(self):
        """
        Takes a configuration file

        Returns a list of blocks. Each block describes a block in the neural
        network to be built. Block is represented as a dictionary in the list

        """
        file = open(self.config_file_path, 'r')
        lines = file.read().split('\n')  # store the lines in a list
        lines = [x for x in lines if len(x) > 0]  # get read of the empty lines
        lines = [x for x in lines if x[0] != '#']
        lines = [x.rstrip().lstrip() for x in lines]

        block = {}
        blocks = []

        for line in lines:
            if line[0] == "[":  # This marks the start of a new block
                if len(block) != 0:
                    blocks.append(block)
                    block = {}
                block["type"] = line[1:-1].rstrip()
            else:
                key, value = line.split("=")
                block[key.rstrip()] = value.lstrip()
        blocks.append(block)
        return blocks

    def load(self, model: nn.Module):
        fp = open(self.weights_file_path, "rb")
        header = torch.from_numpy(np.fromfile(fp, dtype=np.int32, count=5))

        # The rest of the values are the weights
        weights = np.fromfile(fp, dtype=np.float32)
        weights_length = len(weights)
        darknet_module_list = self._parse_cfg()
        module_list = extract_layers(model)

        # Number of weights to be skipped (predicition layer)
        skip_idx = 0
        skip_weights = [261120, 255, 130560, 255, 65280, 255]

        ptr = 0
        model_idx = 0
        darknet_idx = 1

        while model_idx < len(module_list) and darknet_idx < len(darknet_module_list):

            # Load weights only for conv, conv-bias and batch-norm layers
            darknet_module_type = darknet_module_list[darknet_idx]["type"]
            if darknet_module_type != "convolutional":
                darknet_idx += 1
                continue

            if module_list[model_idx][0] != "conv":
                model_idx += 1
                continue

            # if batch norm present, it should follow the conv layer
            try:
                batch_normalize = int(darknet_module_list[darknet_idx]["batch_normalize"])
            except KeyError:
                batch_normalize = 0
                # Do not load weights if the layer does not have a batch norm layer -> Because
                # it is that block which makes the predictions.

                ptr += skip_weights[skip_idx]
                skip_idx += 1
                ptr += skip_weights[skip_idx]
                skip_idx += 1

                model_idx += 1
                darknet_idx += 1
                continue

            conv = module_list[model_idx][1]

            if batch_normalize:
                # Batch Norm Layer, if present, is always placed after the conv layer
                bn = module_list[model_idx + 1][1]

                # Get the number of weights of Batch Norm Layer
                num_bn_biases = bn.bias.numel()

                # Load the weights
                bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                ptr += num_bn_biases

                bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                ptr += num_bn_biases

                bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                ptr += num_bn_biases

                bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                ptr += num_bn_biases

                # Cast the loaded weights into dims of model weights.
                bn_biases = bn_biases.view_as(bn.bias.data)
                bn_weights = bn_weights.view_as(bn.weight.data)
                bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                bn_running_var = bn_running_var.view_as(bn.running_var)

                # Copy the data to model
                bn.bias.data.copy_(bn_biases)
                bn.weight.data.copy_(bn_weights)
                bn.running_mean.copy_(bn_running_mean)
                bn.running_var.copy_(bn_running_var)

            else:
                # Number of biases
                num_biases = conv.bias.numel()

                # Load the weights
                conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                ptr = ptr + num_biases

                # reshape the loaded weights according to the dims of the model weights
                conv_biases = conv_biases.view_as(conv.bias.data)

                # Finally copy the data
                conv.bias.data.copy_(conv_biases)

            # Let us load the weights for the Convolutional layers
            num_weights = conv.weight.numel()

            # Do the same as above for weights
            conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
            ptr = ptr + num_weights

            conv_weights = conv_weights.view_as(conv.weight.data)
            conv.weight.data.copy_(conv_weights)

            darknet_idx += 1
            model_idx += 1

        assert ptr == weights_length, f"{weights_length - ptr} weights were not read."


if __name__ == "__main__":
    num_classes = 19
    config_file_path = "./weights/yolov3.cfg"
    weights_file_path = "./weights/yolov3.weights"

    model = YOLOv3(num_classes=num_classes)
    # summary(model, (3, 416, 416))
    weight_loader = LoadYOLOWeights(config_file_path, weights_file_path)
    weight_loader.load(model)
    model.freeze_backbone_weights()
