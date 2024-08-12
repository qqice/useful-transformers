import sys

import torch
import numpy as np


# Download a whisper checkpoint and run this script to convert it to npz
# The URLs for the whisper checkpoints can be found here:
# https://github.com/openai/whisper/blob/main/whisper/__init__.py#L18-L28
#
# For example, convert the base multilingual model to npz:
#
# $ wget "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt"
# $ python torch_state_dict_to_npz.py base.pt base.npz
#
def main():
    assert len(sys.argv) == 3, f'Run as python {sys.argv[0]} <input_file.pt> <output_filename.npz>'
    pt_file = sys.argv[1]
    model = torch.load(pt_file)
    dims = {f'dims/{k}': v for k, v in model['dims'].items()}
    #print size of model['dims']
    # print("size of model['dims'].items():", len(model['dims'].items()))
    #print('dims.n_vocab:', model['dims']['n_vocab'])
    #print(dims)       
    params = {f'params/{k}': v for k, v in model['model_state_dict'].items()}
    # 打印model['model_state_dict'] 中的decoder.token_embedding.weight 的 dtype print(
    # 'params/decoder.token_embedding.weight dtype:', model['model_state_dict'][
    # 'decoder.token_embedding.weight'].dtype) print(params.keys()) print('params/encoder.positional_embedding
    # shape:', model['model_state_dict']['encoder.positional_embedding'].shape)
    np.savez(sys.argv[2], **dims, **params)


if __name__ == '__main__':
    main()
