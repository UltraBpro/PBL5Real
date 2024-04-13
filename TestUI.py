import os  # doc file
import traceback  # duyet loi
import logging  # log DUHHHHH
import gradio as gr  # gradio
import numpy as np  # tinh toan co ban
import librosa  # doc file am thanh
import torch  # doc model
import glob  # tim file
import json  # doc json

from datetime import datetime  # self explanatory
from sympy import true, false  # neu muon dung share

logging.basicConfig(filename='myapp.log', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info('Started')

from lib.config.config import Config
from lib.vc.vc_infer_pipeline import VC
from lib.vc.settings import change_audio_mode
from lib.vc.audio import load_audio
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from lib.vc.utils import (
    combine_vocal_and_inst,
    cut_vocal_and_inst,
    download_audio,
    load_hubert
)

# auto load stuffs
config = Config()
spaces = os.getenv("SYSTEM") == "spaces"
force_support = None
if config.unsupported is False:
    if config.device == "mps" or config.device == "cpu":
        force_support = False
else:
    force_support = True
audio_mode = []
f0method_mode = []
f0method_info = ""
hubert_model = load_hubert(config)
model_name = ""


def run_convert(
        vc_upload,
        f0_up_key,
        f0_method,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
):
    try:
        logs = []
        logger.info(f"Converting ...")
        logs.append(f"Converting ...")
        yield "\n".join(logs), None
        if vc_upload is None:
            return "You need to upload an audio", None
        sampling_rate, audio = vc_upload
        duration = audio.shape[0] / sampling_rate
        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        if sampling_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
        times = [0, 0, 0]
        f0_up_key = int(f0_up_key)
        vc_input = os.path.join("output", "tts", "tts.mp3")
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            0,
            audio,
            vc_input,
            times,
            f0_up_key,
            f0_method,
            file_index,
            # file_big_npy,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            f0_file=None,
        )
        info = f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}]: npy: {times[0]}, f0: {times[1]}s, infer: {times[2]}s"
        logger.info(f" | {info}")
        logs.append(f"Successfully Convert \n{info}")
        yield "\n".join(logs), (tgt_sr, audio_opt)
    except Exception as err:
        info = traceback.format_exc()
        logger.error(info)
        logger.error(f"Error when using .\n{str(err)}")
        yield info, None
    return run_convert


def load_model():
    global tgt_sr, net_g, vc, hubert_model, version, if_f0, file_index
    path = os.path.join("models", model_name)
    for pth_file in glob.glob(f"{path}/*.pth"):
        file_index = glob.glob(f"{path}/*.index")[0]
        if not file_index:
            logger.warning("No Index file detected!")
        cpt = torch.load(pth_file, map_location=torch.device('cpu'))
        tgt_sr = cpt["config"][-1]
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        if_f0 = cpt.get("f0", 1)
        version = cpt.get("version", "v1")
        if version == "v1":
            if if_f0 == 1:
                net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
            else:
                net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
        elif version == "v2":
            if if_f0 == 1:
                net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
            else:
                net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
        del net_g.enc_q
        logger.info(net_g.load_state_dict(cpt["weight"], strict=False))
        net_g.eval().to(config.device)
        if config.is_half:
            net_g = net_g.half()
        else:
            net_g = net_g.float()
        vc = VC(tgt_sr, config)
    print(model_name)
    return model_name


def change_model_name(tobechangeto):
    global model_name
    model_name = tobechangeto
    load_model()
    return tobechangeto


if __name__ == '__main__':
    model_dirs = [d for d in os.listdir('models') if os.path.isdir(os.path.join('models', d))]
    with gr.Blocks() as main_app:
        models_dropdown = gr.Dropdown(choices=model_dirs, label="Chọn model:")
        models_dropdown.change(fn=change_model_name, inputs=models_dropdown)
        vc_upload = gr.Audio(label="Upload audio file", interactive=True)
        f0_up_key = gr.Number(label="Transpose(f0_up_key)", interactive=True)
        f0_method = gr.Radio(label="Pitch extraction algorithm(f0_method)", choices=["rmvpe", "pm"], value="rmvpe",
                             interactive=True)
        index_rate = gr.Slider(minimum=0, maximum=1, value=0.7, label="Retrieval feature ratio(index_rate)",
                               interactive=True)
        filter_radius = gr.Slider(minimum=0, maximum=7, value=3, step=1, label="Apply Median Filtering(filter_radius)",
                                  interactive=True)
        resample_sr = gr.Slider(minimum=0, maximum=48000, value=0, label="Resampling(resample_sr)", interactive=True)
        rms_mix_rate = gr.Slider(minimum=0, maximum=1, value=1, label="Volume Envelope(rms_mix_rate)", interactive=True)
        protect = gr.Slider(minimum=0, maximum=0.5, value=0.5, label="Voice Protection(protect)", interactive=True)
        Convertbtn = gr.Button("Convert",variant="primary")
        vc_log = gr.Textbox(label="Output Information", visible=False, interactive=False)
        vc_output = gr.Audio(label="Output Audio", interactive=False)
        Convertbtn.click(fn=run_convert,
                         inputs=[vc_upload, f0_up_key, f0_method, index_rate, filter_radius, resample_sr, rms_mix_rate,
                                 protect],
                         outputs=[vc_log ,vc_output]
                         )
    main_app.queue(
        max_size=20,
        api_open=config.api,
    ).launch(
        share=false,
        max_threads=1,
        allowed_paths=["models"]
    )
