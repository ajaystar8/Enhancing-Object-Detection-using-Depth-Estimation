import numpy as np
import torch


################################################### IGNORE FOR NOW ###################################################
def parse_cfg(fp):
    """
    Parses the darknet config file

    return a list of dicts, each of which represents a block
    """

    with open(fp) as cfg:
        lines = [x.strip() for x in cfg.readlines()]  # Filter out all '\n's
        lines = [x for x in lines if len(x) > 0 and x[0] != '#']  # Filter out the comments and blank lines
        # print(lines)
        assert lines[0][0] == '['

        blocks = []
        block = {}
        for line in lines:
            if line[0] == '[':
                if len(block) != 0:
                    blocks.append(block)
                block = {'type': line[1:-1]}
            else:
                key_, value_ = line.split('=')
                block[key_.strip()] = value_.strip()

        blocks.append(block)
        return blocks


def load_darknet_weights(model, fp_weights, fp_config):
    # Thanks to
    # https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-3/
    blocks = parse_cfg(fp_config)
    with open(fp_weights, 'rb') as weight_file:
        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(weight_file, dtype=np.int32, count=5)
        header = torch.from_numpy(header)
        seen = header[3]
        module_list = list(model.modules())

        weights = np.fromfile(weight_file, dtype=np.float32)
        weights_length = len(weights)
        ptr = 0
        for i in range(len(module_list)):
            module_type = blocks[i + 1]["type"]

            # If module_type is convolutional load weights, Otherwise ignore.
            if module_type == "convolutional":
                model = module_list[i]
                conv = model[0]
                batch_normalize = int(blocks[i + 1]["batch_normalize"]) \
                    if "batch_normalize" in blocks[i + 1] else 0

                if batch_normalize:
                    bn = model[1]

                    # Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    # Load the weights
                    bn_biases = torch.from_numpy(
                        weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_weights = torch.from_numpy(
                        weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_mean = torch.from_numpy(
                        weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_var = torch.from_numpy(
                        weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(
                        bn.running_mean)
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
                    conv_biases = torch.from_numpy(
                        weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                # Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                # Do the same as above for weights
                conv_weights = torch.from_numpy(
                    weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

        assert ptr == weights_length
        print(f'{ptr} of {weights_length} read')
