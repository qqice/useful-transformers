import os
import sys
import time
import numpy as np
from dataclasses import dataclass
from .tokenizer import get_tokenizer
# import torch.nn.functional as F
# import torch

import ctypes
# Load this here explicitly into the process, so importing pybind_whisper
# doesn't have to search for librknnrt.so.
_ = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), 'librknnrt.so'))

from .pybind_whisper import WhisperModel as CWhisperModel

@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


def set_encoder_params(model, p, dims):
    Wconv0, conv0_bias = p['encoder.conv1.weight'], p['encoder.conv1.bias']
    Wconv1, conv1_bias = p['encoder.conv2.weight'], p['encoder.conv2.bias']
    # [output_depth, channels, input_depth]
    model.set_conv0_weights(Wconv0.transpose([0, 2, 1]))
    model.set_conv1_weights(Wconv1.transpose([0, 2, 1]))
    model.set_conv0_bias(conv0_bias)
    model.set_conv1_bias(conv1_bias)

    model.set_encoder_positional_embedding(p['encoder.positional_embedding'])

    for i in range(dims.n_audio_layer):
        prefix = f'encoder.blocks.{i}'
        model.set_encoder_attn_ln_gamma(i, p[f'{prefix}.attn_ln.weight'])
        model.set_encoder_attn_ln_beta(i, p[f'{prefix}.attn_ln.bias'])
        model.set_encoder_attn_Wq(i, p[f'{prefix}.attn.query.weight'].T)
        model.set_encoder_attn_q_bias(i, p[f'{prefix}.attn.query.bias'])
        model.set_encoder_attn_Wk(i, p[f'{prefix}.attn.key.weight'].T)
        model.set_encoder_attn_Wv(i, p[f'{prefix}.attn.value.weight'].T)
        model.set_encoder_attn_v_bias(i, p[f'{prefix}.attn.value.bias'])
        model.set_encoder_attn_Wout(i, p[f'{prefix}.attn.out.weight'].T)
        model.set_encoder_attn_out_bias(i, p[f'{prefix}.attn.out.bias'])

        model.set_encoder_mlp_ln_gamma(i, p[f'{prefix}.mlp_ln.weight'])
        model.set_encoder_mlp_ln_beta(i, p[f'{prefix}.mlp_ln.bias'])
        model.set_encoder_Wfc1(i, p[f'{prefix}.mlp.0.weight'].T)
        model.set_encoder_fc1_bias(i, p[f'{prefix}.mlp.0.bias'])
        model.set_encoder_Wfc2(i, p[f'{prefix}.mlp.2.weight'].T)
        model.set_encoder_fc2_bias(i, p[f'{prefix}.mlp.2.bias'])

    model.set_encoder_ln_post_gamma(p[f'encoder.ln_post.weight'])
    model.set_encoder_ln_post_beta(p[f'encoder.ln_post.bias'])


def next_multiple_of_3(x):
    return int(3 * ((x + 2) // 3))


def set_decoder_params(model, p, dims):
    model.set_decoder_positional_embedding(p['decoder.positional_embedding'])

    for i in range(dims.n_text_layer):
        prefix = f'decoder.blocks.{i}'
        model.set_decoder_attn_ln_gamma(i, p[f'{prefix}.attn_ln.weight'])
        model.set_decoder_attn_ln_beta(i, p[f'{prefix}.attn_ln.bias'])
        model.set_decoder_attn_Wq(i, p[f'{prefix}.attn.query.weight'].T)
        model.set_decoder_attn_q_bias(i, p[f'{prefix}.attn.query.bias'])
        model.set_decoder_attn_Wk(i, p[f'{prefix}.attn.key.weight'].T)
        model.set_decoder_attn_Wv(i, p[f'{prefix}.attn.value.weight'].T)
        model.set_decoder_attn_v_bias(i, p[f'{prefix}.attn.value.bias'])
        model.set_decoder_attn_Wout(i, p[f'{prefix}.attn.out.weight'].T)
        model.set_decoder_attn_out_bias(i, p[f'{prefix}.attn.out.bias'])

        model.set_decoder_cross_attn_ln_gamma(i, p[f'{prefix}.cross_attn_ln.weight'])
        model.set_decoder_cross_attn_ln_beta(i, p[f'{prefix}.cross_attn_ln.bias'])
        model.set_decoder_cross_attn_Wq(i, p[f'{prefix}.cross_attn.query.weight'].T)
        model.set_decoder_cross_attn_q_bias(i, p[f'{prefix}.cross_attn.query.bias'])
        model.set_decoder_cross_attn_Wk(i, p[f'{prefix}.cross_attn.key.weight'].T)
        model.set_decoder_cross_attn_Wv(i, p[f'{prefix}.cross_attn.value.weight'].T)
        model.set_decoder_cross_attn_v_bias(i, p[f'{prefix}.cross_attn.value.bias'])
        model.set_decoder_cross_attn_Wout(i, p[f'{prefix}.cross_attn.out.weight'].T)
        model.set_decoder_cross_attn_out_bias(i, p[f'{prefix}.cross_attn.out.bias'])

        model.set_decoder_mlp_ln_gamma(i, p[f'{prefix}.mlp_ln.weight'])
        model.set_decoder_mlp_ln_beta(i, p[f'{prefix}.mlp_ln.bias'])
        model.set_decoder_Wfc1(i, p[f'{prefix}.mlp.0.weight'].T)
        model.set_decoder_fc1_bias(i, p[f'{prefix}.mlp.0.bias'])
        model.set_decoder_Wfc2(i, p[f'{prefix}.mlp.2.weight'].T)
        model.set_decoder_fc2_bias(i, p[f'{prefix}.mlp.2.bias'])

    model.set_decoder_ln_gamma(p[f'decoder.ln.weight'])
    model.set_decoder_ln_beta(p[f'decoder.ln.bias'])
    Wdetokenizer = p[f'decoder.token_embedding.weight'].T
    n_vocab = next_multiple_of_3(dims.n_vocab-1)
    slice_len = n_vocab // 3
    model.set_detokenizer0(Wdetokenizer[:, :slice_len])
    model.set_detokenizer1(Wdetokenizer[:, slice_len:2*slice_len])
    extra_columns = n_vocab - dims.n_vocab + 1
    third_slice = Wdetokenizer[:, 2*slice_len:-1]
    if extra_columns:
        third_slice = np.concatenate([third_slice, np.zeros_like(third_slice[:, :extra_columns])], -1)
    model.set_detokenizer2(third_slice)


class WhisperModel(object):
    def __init__(self, model='tiny.en', verbose=False):
        params_file = os.path.join(os.path.dirname(__file__), "weights", f"{model}.npz")
        params_file = np.load(params_file)
        dims = {k.split('/')[-1]:v for k, v in params_file.items() if k.startswith('dims/')}
        params = {k.split('/')[-1]:v for k, v in params_file.items() if k.startswith('params/')}

        dims = ModelDimensions(**dims)
        n_vocab = next_multiple_of_3(dims.n_vocab-1)
        self.model = CWhisperModel(dims.n_mels,
                                   dims.n_audio_ctx,
                                   dims.n_audio_state,
                                   dims.n_audio_head,
                                   dims.n_audio_layer,
                                   dims.n_text_ctx,
                                   dims.n_text_state,
                                   dims.n_text_head,
                                   dims.n_text_layer,
                                   n_vocab)
        self.dims = dims

        set_encoder_params(self.model, params, dims)
        set_decoder_params(self.model, params, dims)
        self.multilingual = not model.endswith('.en')
        self.tokenizer = get_tokenizer(multilingual=self.multilingual)
        self.lang_dict = dict(zip(self.tokenizer.all_language_codes, self.tokenizer.all_language_tokens))
        assert os.sched_getaffinity(os.getpid()) == set([4, 5, 6, 7]), (
            f'Should be run with taskset -c4-7')

        assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        self.N_FFT = 400
        self.HOP_LENGTH = 160
        self.mel_filters = np.load(os.path.join(assets_dir, 'mel_filters.npz'))['mel_80'].T
        fft_matrix_file = np.load(os.path.join(assets_dir, 'fft_params.npz'))
        self.fft_matrix_real = fft_matrix_file['fft_matrix_real']
        self.fft_matrix_imag = fft_matrix_file['fft_matrix_imag']

        self.verbose = verbose

    def mel_spectrogram(self, audio):
        audio = audio.squeeze()
        audio = np.pad(audio, [self.N_FFT // 2, self.N_FFT // 2], mode='reflect')
        audio = np.lib.stride_tricks.sliding_window_view(audio, self.N_FFT)[::self.HOP_LENGTH]
        stft_real = audio @ self.fft_matrix_real.squeeze()
        stft_imag = audio @ self.fft_matrix_imag.squeeze()
        magnitudes = stft_real ** 2 + stft_imag ** 2
        mel_spec = magnitudes @ self.mel_filters
        log_spec = np.clip(mel_spec, 1e-10, np.finfo(np.float32).max)
        log_spec = np.log10(log_spec)
        log_spec = np.maximum(log_spec, np.max(log_spec) - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        log_spec = log_spec[:-1, :][np.newaxis, ...]
        return log_spec

    def decode_no_timestamps(self, mel, task='transcribe', src_lang='en'):
        tokenizer = self.tokenizer
        suppress_tokens_sans_no_speech = list(tokenizer.non_speech_tokens) + [
                self.tokenizer.transcribe,
                self.tokenizer.translate,
                self.tokenizer.sot,
                self.tokenizer.sot_prev,
                self.tokenizer.sot_lm,
            ]
        # Suppress padding tokens.
        for i in range(self.dims.n_vocab-1, next_multiple_of_3(self.dims.n_vocab-1)):
          suppress_tokens_sans_no_speech += [i]

        suppress_tokens = suppress_tokens_sans_no_speech + [tokenizer.no_speech]
        suppress_tokens = list(sorted(set(suppress_tokens)))
        initial_suppress_tokens = suppress_tokens + tokenizer.encode(' ') + [tokenizer.eot]

        self.model.reset(mel)
        #audio_features = self.model.get_audio_features().view(np.float32)
        #print(f"audio_features shape {audio_features.shape}")
        #save_path = 'audio_features_wrong.npy'
        #np.save(save_path, audio_features)
        
        # load_path = 'audio_features_ok.npy'
        # audio_features_load = np.load(load_path)
        # audio_features = self.model.set_audio_features(audio_features_load)

        initial_prompt = list(tokenizer.sot_sequence_including_notimestamps)
        #initial_prompt = list(tokenizer.sot_sequence)

        if self.multilingual:
            assert src_lang in self.lang_dict, f'{src_lang} is not a supported language'
            initial_prompt[1] = self.lang_dict[src_lang]
            if task == 'translate': initial_prompt[2] = self.tokenizer.translate
            
            if src_lang == 'zh' and task == 'transcribe': 
                initial_prompt2 = initial_prompt.copy()
                initial_prompt = [self.tokenizer.sot_prev] + list(tuple(self.tokenizer.encode(' ' + 'Hello,你是谁,赤兔机器人，开始充电，暂停任务，开始导航，停止充电，继续任务，停止导航。'.strip())))
                                      
                initial_prompt.extend(initial_prompt2)
                # initial_prompt.extend(tuple(self.tokenizer.encode('以下的是普通话的句子，'.strip())))
            
        if self.verbose:
            print(f'{initial_prompt} {self.tokenizer.decode(initial_prompt)}')
        for p in initial_prompt:
            self.model.call_no_copy(p)

        # logprobs = self.model.log_softmax(initial_suppress_tokens).view(np.float16)
        logprobs = self.model.get_logits32(initial_suppress_tokens).view(np.float32)
        decoded_tokens = [np.argmax(logprobs)]
        # decoded_tokens = [50364, 25597,   338,  1546, 37960, 34386,  1654]
        # print(f'timestamp_begin  {self.tokenizer.timestamp_begin}')
        # test_tokens = [50364, 25597,   338,  1546, 37960, 34386]
        # for p in test_tokens:
        #     self.model.call_no_copy(p)
        # print(f'test_tokens  {self.tokenizer.decode(test_tokens)}')
        
        # logits = self.model.get_logits32(initial_suppress_tokens).view(np.float32)
        # #将logits转换为tensor
        # logits = torch.tensor(logits)
        # # print(logits.shape)
        # logprobs = F.log_softmax(logits, dim=-1)
        # decoded_tokens = [torch.argmax(logprobs).item()]
               
        #print(decoded_tokens)
        # print("{:.6f}".format(logprobs[decoded_tokens[0]].item()))
        # print(len(logprobs))
        # print(logprobs[51865])
        # print(logprobs[51866])
        # print(logprobs[tokenizer.eot])
        # print(self.tokenizer.decode(decoded_tokens))
        # print(self.tokenizer.decode(initial_suppress_tokens))
        
        while len(decoded_tokens) < 120:
            self.model.call_no_copy(decoded_tokens[-1])
            
            # logprobs = self.model.log_softmax(suppress_tokens).view(np.float16)
            logprobs = self.model.get_logits32(suppress_tokens).view(np.float32)
            
            speech_token = np.argmax(logprobs[:tokenizer.eot])
            speech_logprob = logprobs[speech_token]
            eot_logprob = logprobs[tokenizer.eot]
            
            # logits = self.model.get_logits32(suppress_tokens).view(np.float32)
            # #将logits转换为tensor
            # logits = torch.tensor(logits)
            # logprobs = F.log_softmax(logits, dim=-1)
            # speech_token = torch.argmax(logprobs[:tokenizer.eot]).item()
            # speech_logprob = logprobs[speech_token].item()
            # eot_logprob = logprobs[tokenizer.eot].item()

            if eot_logprob > speech_logprob:
                break
            
            #print(speech_token)
            decoded_tokens.append(speech_token)
            
            if len(decoded_tokens) >= 4 and decoded_tokens[-1] == decoded_tokens[-2] == decoded_tokens[-3] == decoded_tokens[-4]:
                print(f'Breaking because of repetition1')
                break
            elif len(decoded_tokens) >= 8 and decoded_tokens[-1] == decoded_tokens[-3] == decoded_tokens[-5] == decoded_tokens[-7]:
                print(f'Breaking because of repetition2')
                break
            elif len(decoded_tokens) >= 11 and decoded_tokens[-1] == decoded_tokens[-4] == decoded_tokens[-7] == decoded_tokens[-10]:
                print(f'Breaking because of repetition3')
                break
            elif len(decoded_tokens) >= 14 and decoded_tokens[-1] == decoded_tokens[-5] == decoded_tokens[-9] == decoded_tokens[-13]:
                print(f'Breaking because of repetition4')
                break
            
        return decoded_tokens


def decode_wav_file(filename, model='tiny.en', task='transcribe', src_lang='en'):
    import wave
    w = wave.open(filename)
    assert w.getnchannels() == 1, f'Only one channel supported'
    assert w.getsampwidth() == 2, f'Datatype should be int16'
    assert w.getframerate() == 16000, f'Only 16kHz supported'
    frames = w.readframes(w.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16)
    model = WhisperModel(model)
    return decode_pcm(audio, model, task, src_lang)

def decode_pcm(audio, model, task='transcribe', src_lang='en'):
    import tqdm
    assert type(audio) == np.ndarray, f'audio should be a numpy array'
    if audio.dtype in (np.int16, np.int32, np.int8):
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    elif audio.dtype == np.float64:
        audio = audio.astype(np.float32)
    if type(model) is str:
        model = WhisperModel(model)
    assert type(model) is WhisperModel, f'model should be a WhisperModel or a string'
    segments = np.split(audio, np.arange(0, audio.shape[0], 480000)[1:])
    decoded = []
    for segment in tqdm.tqdm(segments):
        remainder = 480000 - segment.shape[0]
        segment = np.concatenate([segment, np.zeros([remainder]).astype(np.float32)])
        mel = model.mel_spectrogram(segment[np.newaxis])
        ##将mel保存为mel_wrong.npy文件
        #print(f'mel.shape {mel.shape}')
        #save_path = 'mel_wrong.npy'
        #np.save(save_path, mel)
        
        # load_path = 'mel_ok.npy'
        # mel_load = np.load(load_path)
        # mel = np.reshape(mel_load, [1, mel_load.shape[0], mel_load.shape[1]])
        
        
        start_time = time.time()

        tokens = model.decode_no_timestamps(mel, task, src_lang)

        end_time = time.time()

        print(f"Model Execution time: {end_time - start_time} seconds")
        decoded += tokens
    return model.tokenizer.decode(decoded)
