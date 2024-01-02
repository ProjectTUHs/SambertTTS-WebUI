import subprocess
import random
import os
from pathlib import Path
import librosa
from scipy.io import wavfile
import numpy as np
import torch
import csv
import whisper
import gradio as gr
import soundfile as sf

os.system("pip install --upgrade Cython==0.29.35")
os.system("pip install pysptk --no-build-isolation")
os.system("pip install kantts -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html")
os.system("pip install tts-autolabel -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html")

import sox

def split_long_audio(model, filepaths, save_dir="data_dir", out_sr=44100):
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    for file_idx, filepath in enumerate(filepaths):

        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)

        print(f"Transcribing file {file_idx}: '{filepath}' to segments...")
        result = model.transcribe(filepath, word_timestamps=True, task="transcribe", beam_size=5, best_of=5)
        segments = result['segments']

        wav, sr = librosa.load(filepath, sr=None, offset=0, duration=None, mono=True)
        wav, _ = librosa.effects.trim(wav, top_db=20)
        peak = np.abs(wav).max()
        if peak > 1.0:
            wav = 0.98 * wav / peak
        wav2 = librosa.resample(wav, orig_sr=sr, target_sr=out_sr)
        wav2 /= max(wav2.max(), -wav2.min())

        for i, seg in enumerate(segments):
            start_time = seg['start']
            end_time = seg['end']
            wav_seg = wav2[int(start_time * out_sr):int(end_time * out_sr)]
            wav_seg_name = f"{file_idx}_{i}.wav"
            out_fpath = save_path / wav_seg_name
            wavfile.write(out_fpath, rate=out_sr, data=(wav_seg * np.iinfo(np.int16).max).astype(np.int16))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
whisper_size = "medium"
whisper_model = whisper.load_model(whisper_size).to(device)

from modelscope.tools import run_auto_label

from modelscope.models.audio.tts import SambertHifigan
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.utils.audio.audio_utils import TtsTrainType

pretrained_model_id = 'damo/speech_personal_sambert-hifigan_nsf_tts_zh-cn_pretrain_16k'

dataset_id = "/home/user/app/output_training_data/"
pretrain_work_dir = "/home/user/app/pretrain_work_dir/"


def auto_label(Voicetoclone, VoiceMicrophone):
    if VoiceMicrophone is not None:
        audio = VoiceMicrophone
    else:
        audio = Voicetoclone
        
    try:
        split_long_audio(whisper_model, audio, "/home/user/app/test_wavs/")
        input_wav = "/home/user/app/test_wavs/"
        output_data = "/home/user/app/output_training_data/"
        ret, report = run_auto_label(input_wav=input_wav, work_dir=output_data, resource_revision="v1.0.7")
    
    except Exception as e:
        print(e)
    return "标注成功"



def train(train_step):
    try:

        train_info = {
            TtsTrainType.TRAIN_TYPE_SAMBERT: {  # 配置训练AM（sambert）模型
                'train_steps': int(train_step / 20) * 20 + 2,               # 训练多少个step
                'save_interval_steps': int(train_step / 20) * 20,           # 每训练多少个step保存一次checkpoint
                'log_interval': int(train_step / 20) * 20                   # 每训练多少个step打印一次训练日志
            }
        }

        kwargs = dict(
            model=pretrained_model_id,                  # 指定要finetune的模型
            model_revision = "v1.0.6",
            work_dir=pretrain_work_dir,                 # 指定临时工作目录
            train_dataset=dataset_id,                   # 指定数据集id
            train_type=train_info                       # 指定要训练类型及参数
        )

        trainer = build_trainer(Trainers.speech_kantts_trainer,
                            default_args=kwargs)

        trainer.train()

    except Exception as e:
        print(e)

    return "训练完成"


# 保存模型

import shutil

import datetime

def save_model(worked_dir,dest_dir):
    worked_dir = "/home/user/app/pretrain_work_dir"
    dest_dir = "/home/user/app/trained_model"

    if os.listdir(worked_dir): 

        now = datetime.datetime.now()
        
        date_str = now.strftime("%Y%m%d%H%M%S")
        
        dest_folder = os.path.join(dest_dir, date_str)
        
        shutil.copytree(worked_dir, dest_folder)

                # List of files and directories to delete
        files_to_delete = [
            "tmp_voc",
            "tmp_am/ckpt/checkpoint_2400000.pth",
            "orig_model/description",
            "orig_model/.mdl",
            "orig_model/.msc",
            "orig_model/README.md",
            "orig_model/resource",
            "orig_model/description",
            "orig_model/basemodel_16k/sambert",
            "orig_model/basemodel_16k/speaker_embedding",
            "data/duration",
            "data/energy",
            "data/f0",
            "data/frame_energy",
            "data/frame_f0",
            "data/frame_uv",
            "data/mel",
            "data/raw_duration",
            "data/wav",
            "data/am_train.lst",
            "data/am_valid.lst",
            "data/badlist.txt",
            "data/raw_metafile.txt",
            "data/Script.xml",
            "data/train.lst",
            "data/valid.lst",
            "data/se/0_*"
        ]

        for item in files_to_delete:
            item_path = os.path.join(dest_folder, item)
            if os.path.exists(item_path):
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
        
        shutil.rmtree("/home/yiho/Personal-TTS-v3/output_training_data")
        shutil.rmtree("/home/yiho/Personal-TTS-v3/pretrain_work_dir")
        shutil.rmtree("/home/yiho/Personal-TTS-v3/test_wavs")
        
        os.mkdir("/home/yiho/Personal-TTS-v3/output_training_data")
        os.mkdir("/home/yiho/Personal-TTS-v3/pretrain_work_dir")
        os.mkdir("/home/yiho/Personal-TTS-v3/test_wavs")
        
        return f"模型已成功保存为 {date_str}"
    else: 
        return "保存失败，模型已保存或已被清除"


import random

def infer(text):

  model_dir = "/home/user/app/pretrain_work_dir/"

  test_infer_abs = {
      'voice_name':
      'F7',
      'am_ckpt':
      os.path.join(model_dir, 'tmp_am', 'ckpt'),
      'am_config':
      os.path.join(model_dir, 'tmp_am', 'config.yaml'),
      'voc_ckpt':
      os.path.join(model_dir, 'orig_model', 'basemodel_16k', 'hifigan', 'ckpt'),
      'voc_config':
      os.path.join(model_dir, 'orig_model', 'basemodel_16k', 'hifigan',
              'config.yaml'),
      'audio_config':
      os.path.join(model_dir, 'data', 'audio_config.yaml'),
      'se_file':
      os.path.join(model_dir, 'data', 'se', 'se.npy')
  }
  kwargs = {'custom_ckpt': test_infer_abs}

  model_id = SambertHifigan(os.path.join(model_dir, "orig_model"), **kwargs)

  inference = pipeline(task=Tasks.text_to_speech, model=model_id)
  output = inference(input=text) 


  now = datetime.datetime.now()
  date_str = now.strftime("%Y%m%d%H%M%S")
  rand_num = random.randint(1000, 9999)
  filename = date_str + str(rand_num)
  

  with open(filename + "0.wav", mode='bx') as f:
      f.write(output["output_wav"])


  y, sr = librosa.load(filename + "0.wav")

  S = librosa.stft(y)

  noise = S[np.abs(S) < np.percentile(S, 95)]
  noise_mean, noise_std = np.mean(noise), np.std(noise)

  filter_ = np.ones_like(S)
  filter_[np.abs(S) < noise_mean + 2 * noise_std] = 0

  filtered_S = filter_ * S

  filtered_y = librosa.istft(filtered_S)

  sf.write(filename + "testfile.wav", filtered_y, sr)


  os.remove(filename + "0.wav")


  return filename + "testfile.wav"


def infer_custom(model_name, text, noise_level): 

  custom_model_dir = os.path.join("/home/user/app/trained_model/", model_name) 

  custom_infer_abs = {
      'voice_name':
      'F7', 
      'am_ckpt':
      os.path.join(custom_model_dir, 'tmp_am', 'ckpt'),
      'am_config':
      os.path.join(custom_model_dir, 'tmp_am', 'config.yaml'),
      'voc_ckpt':
      os.path.join(custom_model_dir, 'orig_model', 'basemodel_16k', 'hifigan', 'ckpt'),
      'voc_config':
      os.path.join(custom_model_dir, 'orig_model', 'basemodel_16k', 'hifigan',
              'config.yaml'),
      'audio_config':
      os.path.join(custom_model_dir, 'data', 'audio_config.yaml'),
      'se_file':
      os.path.join(custom_model_dir, 'data', 'se', 'se.npy')
  }
  kwargs = {'custom_ckpt': custom_infer_abs}

  model_id = SambertHifigan(os.path.join(custom_model_dir, "orig_model"), **kwargs)

  inference = pipeline(task=Tasks.text_to_speech, model=model_id)
  output = inference(input=text)


  now = datetime.datetime.now()
  date_str = now.strftime("%Y%m%d%H%M%S")
  rand_num = random.randint(1000, 9999)
  filename = date_str + str(rand_num)


  with open(filename + ".wav", mode='bx') as f:
      f.write(output["output_wav"])




  y, sr = librosa.load(filename + ".wav")

  S = librosa.stft(y)

  noise = S[np.abs(S) < np.percentile(S, 95)]
  noise_mean, noise_std = np.mean(noise), np.std(noise)

  filter_ = np.ones_like(S)
  filter_[np.abs(S) < noise_mean + noise_level * noise_std] = 0

  filtered_S = filter_ * S

  filtered_y = librosa.istft(filtered_S)

  sf.write(filename + "customfile.wav", filtered_y, sr)

  os.remove(filename + ".wav")

  return filename + "customfile.wav"



trained_model = "/home/user/app/trained_model/"


def update_model_dropdown(inp3):

    model_list = os.listdir(trained_model)

    return gr.Dropdown(choices=model_list, value=inp3)


def rename_model(old_name, new_name):

    if not os.path.isdir(os.path.join(trained_model, old_name)):
        return "模型名称不存在，请重新输入！"
    else:
        try:
            os.rename(os.path.join(trained_model, old_name), os.path.join(trained_model, new_name))
            return "模型重命名成功！"
        except OSError:
            return "新名称已经存在，请重新输入！"


# 清除训练缓存
def clear_cache(a):
    shutil.rmtree("/home/user/app/output_training_data")
    shutil.rmtree("/home/user/app/pretrain_work_dir")
    shutil.rmtree("/home/user/app/test_wavs")

    os.mkdir("/home/user/app/output_training_data")
    os.mkdir("/home/user/app/pretrain_work_dir")
    os.mkdir("/home/user/app/test_wavs")
    return "已清除缓存，请返回训练页面重新训练"


from textwrap import dedent



def FRCRN_De_Noise(noise_wav, noisemic_wav):
  
  if noisemic_wav is not None:
      noise_audio = noisemic_wav
  else:
      noise_audio = noise_wav
    
  ans = pipeline(
    Tasks.acoustic_noise_suppression,
    model='/home/yiho/Personal-TTS-v3/damo/speech_frcrn_ans_cirm_16k')

  now = datetime.datetime.now()
  date_str = now.strftime("%Y%m%d%H%M%S")
  rand_num = random.randint(1000, 9999)
  filename = date_str + str(rand_num)

  result = ans(
    noise_audio,
    output_path= filename + "AIdenoise.wav" )
  
  return filename + "AIdenoise.wav"

def Normal_De_Noise(noise_wav, noisemic_wav, noise_level):
  if noisemic_wav is not None:
      noise_audio = noisemic_wav
  else:
      noise_audio = noise_wav
  
  now = datetime.datetime.now()
  date_str = now.strftime("%Y%m%d%H%M%S")
  rand_num = random.randint(1000, 9999)
  filename = date_str + str(rand_num)


  y, sr = librosa.load(noise_audio)

  S = librosa.stft(y)

  noise = S[np.abs(S) < np.percentile(S, 95)]
  noise_mean, noise_std = np.mean(noise), np.std(noise)

  filter_ = np.ones_like(S)
  filter_[np.abs(S) < noise_mean + noise_level * noise_std] = 0

  filtered_S = filter_ * S

  filtered_y = librosa.istft(filtered_S)

  sf.write(filename + "denoise.wav", filtered_y, sr)

  return filename + "denoise.wav"


app = gr.Blocks()

with app:
    gr.Markdown("# <center>🥳🎶🎡 - Sambert中文声音克隆</center>")
    gr.Markdown("## <center>🌟 - 训练3分钟，推理10秒钟，中英真实拟声 </center>")
    gr.Markdown("### <center>🌊 - 基于SambertHifiGan项目修改而来，添加两种降噪功能、模型管理功能等")

    with gr.Tabs(): 
        with gr.TabItem("一键训练"): 
            with gr.Row():
              with gr.Column():
                inp1 = gr.Audio(type="filepath", sources="upload", label="方案一：请从本地上传一段语音")
                inp_micro = gr.Audio(type="filepath", sources="microphone", label="方案二：请用麦克风录制您的声音")
              with gr.Column():
                out1 = gr.Textbox(label="标注情况", lines=1, interactive=False)
                out2 = gr.Textbox(label="训练情况", lines=1, interactive=False)
                inp2 = gr.Slider(label="训练步数(需要为20的倍数)", minimum=200, maximum=4000, value=400, min_width=40)
                inp3 = gr.Textbox(label="请在这里填写您想合成的文本", placeholder="想说却还没说的 还很多...", lines=3, interactive=True)
              with gr.Column():
                out3 = gr.Audio(type="filepath", label="为您合成的专属音频")
                out4 = gr.Textbox(label="保存情况", lines=1, interactive=False)
            with gr.Row():
              btn1 = gr.Button("1.标注数据")
              btn2 = gr.Button("2.开始训练")
              btn3 = gr.Button("3.一键推理", variant="primary")
              btn4 = gr.Button("4.保存模型", variant="primary") 
          
            btn1.click(auto_label, [inp1, inp_micro], out1)
            btn2.click(train, inp2, out2)
            btn3.click(infer, inp3, out3)
            btn4.click(save_model, out1, out4) 
            with gr.Accordion("📒 训练教程", open=True):
              _ = f""" 如何开始训练: 
                  * 第一步，选择 [方案一] 或 [方案二] 上传一分钟左右的音频，注意要吐字清晰、感情饱满、音色纯净不含杂音
                  * 第二步，点击“标注数据”，等到提示标注成功后，选择合适的训练步数，点击“开始训练”等待训练完成
                  * 第三步，耐心等待训练成功后，在文本框内输入想要生成的文字，点击“一键生成”按钮，生成克隆后的语音
                  * ！！注意！！  不要生成会对个人以及组织造成侵害的内容
                  * 如果您的训练素材比较嘈杂，您可以在[AI降噪]选项卡上传或录制训练音频，降噪后再上传到训练界面
                  * 如果您需要用方案二录制您的声音，以下是一段长度合适的文本，供您朗读并录制：

                  记得春天的时候，小草就转出地面，树上的叶子也抽出来了，大地一片绿色，就像穿上了一件绿衣裳。我就与小孩子一起到田野去捉蜻蜓，玩游戏，比如老鹰合作小鸡或是捉迷藏，又或是跳格子。到了夏天，天气热了，我就会与小孩子到水库里面游泳，那时候水库的安全系数还不是很高，几乎每年都会有事故发生，所以父母都不会让我去游泳的，被发现之后当然就是处罚或是责骂了。可是那时候自己真的很叛逆，也不知道什么是危险，被处罚之后下一次还是回去的。到了秋天，田野一片金黄，山上的野果也成熟了，我就会与自己的伙伴拿着篮子到上山去采，采回来了还要跟自己的好朋友一起分享。
                
                  """
              gr.Markdown(dedent(_))

        with gr.TabItem("声音合成"): 
            with gr.Row():
              with gr.Column():
                inp21 = gr.Dropdown(label="请选择一个模型", choices=os.listdir(trained_model)) 
                inp22 = gr.Slider(label="降噪强度(为0时不降噪)", minimum=0, maximum=3, value=2)
              with gr.Column():
                inp23 = gr.Textbox(label="请在这里填写您想合成的文本", placeholder="想说却还没说的 还很多...", lines=3,  interactive=True)
              with gr.Column():
                out21 = gr.Audio(type="filepath", label="为您合成的专属音频", interactive=False)
            with gr.Row():
              btn21 = gr.Button("刷新模型列表") 
              btn22 = gr.Button("一键推理", variant="primary") 

            btn21.click(update_model_dropdown, inp21, inp21)
            btn22.click(infer_custom, [inp21, inp23, inp22], out21) 
            with gr.Accordion("📒 推理教程", open=True):
              _ = f""" 如何推理声音: 
                  * 第一步，选择一个你想要使用的模型，如果训练后保存的模型无法找到请点击“刷新模型列表”
                  * 第二步，在文本框处输入你想要生成的文本，选择降噪强度，如果无需降噪请将强度设为0
                  * 第三步，点击“一键生成”按钮，生成克隆后的语音
                  * ！！注意！！  不要生成会对个人以及组织造成侵害的内容
                  * 此处使用的降噪算法为机械降噪，非AI降噪，如需AI降噪可以将生成的音频下载后转到“AI降噪”选项卡进行AI降噪

                  """
              gr.Markdown(dedent(_))
        
        with gr.TabItem("模型修改"): 
            with gr.Row():
              with gr.Column():
                inp31 = gr.Dropdown(label="选择重命名的模型", choices=os.listdir(trained_model)) 
              with gr.Column():
                inp32 = gr.Textbox(label="输入模型命名", placeholder="新名称", lines=1,  interactive=True)
              with gr.Column():    
                out31 = gr.Textbox(label="保存情况", lines=1, interactive=False)
            with gr.Row():
              btn31 = gr.Button("刷新模型列表") 
              btn32 = gr.Button("重命名", variant="primary") 

            btn31.click(update_model_dropdown, inp31, inp31)
            btn32.click(rename_model, [inp31, inp32], out31)
            with gr.Accordion("📒 推理教程", open=True):
              _ = f""" 如何修改模型名称: 
                  * 第一步，选择一个你想要修改的模型，如果训练后保存的模型无法找到请点击“刷新模型列表”
                  * 第二步，在文本框处输入你想要修改的模型名称，推荐以“[训练步数]时间-名称”来命名
                  * 第三步，点击“重命名”按钮对模型重命名

                  """
              gr.Markdown(dedent(_))              

        with gr.TabItem("AI降噪"): 
            with gr.Row():
              with gr.Column():
                inp41 = gr.Audio(type="filepath", sources="upload", label="方案一：请从本地上传一段语音")
                inp_micro42 = gr.Audio(type="filepath", sources="microphone", label="方案二：请用麦克风录制您的声音")
              with gr.Column():
                out41 = gr.Audio(type="filepath", label="降噪后的音频", interactive=False)
                inp43 = gr.Slider(label="机械降噪强度(非AI降噪)", minimum=0, maximum=3, value=2)  
                btn41 = gr.Button("机械降噪")
                btn42 = gr.Button("一键AI降噪", variant="primary")
            
            btn41.click(Normal_De_Noise, [inp41, inp_micro42, inp43], out41)
            btn42.click(FRCRN_De_Noise, [inp41, inp_micro42], out41)
            with gr.Accordion("📒 AI降噪", open=True):
              _ = f""" 如何使用AI降噪: 
                  * 第一步，在[方案一]上传你想要降噪的音频，或者在[方案二]录制音频
                  * 第二步，点击“一键AI降噪”进行降噪
                  * 第三步，下载降噪后的音频
                  * 如果您的训练素材比较嘈杂，您可以在此处上传或录制训练音频，降噪后再上传到训练界面
                  * 如果您需要用方案二录制您的声音，以下是一段长度合适的文本，供您朗读并录制：

                  记得春天的时候，小草就转出地面，树上的叶子也抽出来了，大地一片绿色，就像穿上了一件绿衣裳。我就与小孩子一起到田野去捉蜻蜓，玩游戏，比如老鹰合作小鸡或是捉迷藏，又或是跳格子。到了夏天，天气热了，我就会与小孩子到水库里面游泳，那时候水库的安全系数还不是很高，几乎每年都会有事故发生，所以父母都不会让我去游泳的，被发现之后当然就是处罚或是责骂了。可是那时候自己真的很叛逆，也不知道什么是危险，被处罚之后下一次还是回去的。到了秋天，田野一片金黄，山上的野果也成熟了，我就会与自己的伙伴拿着篮子到上山去采，采回来了还要跟自己的好朋友一起分享。
                  
                  * AI降噪与机械降噪的不同：机械降噪主要是移除声音的激波，会对人声造成一定的破坏，而AI降噪主要是移除声音中的非人声部分，可以处理复杂的背景音频环境，但是对人声本身质量问题处理的效果一般                  
                  """
              gr.Markdown(dedent(_))            
        
        with gr.TabItem("缓存清理"): 
            with gr.Row():
              with gr.Column():
                gr.Markdown("### <center>注意，这会清除[一键训练]界面生成的所有数据")
                gr.Markdown("### <center>包括标注数据，训练数据，及最终模型")
                gr.Markdown("### <center>如需保存模型请点击保存当前模型按钮")
              with gr.Column():
                out97 = gr.Textbox(label="", lines=1, interactive=False)
                btn91 = gr.Button("保存当前模型", ) 
                btn92 = gr.Button("清空缓存数据", variant="primary") 
            
            btn91.click(save_model, out1, out97) 
            btn92.click(clear_cache, out1, out97)




            

    with gr.Accordion("📒 使用指南", open=False):
        _ = f""" 如何使用此程序: 
            * [一键训练] ： 上传或录制音频，程序会自动标注音频，一键训练模型，支持训练后推理试听，支持模型保存
            * [声音合成] ： 在这里可以选择已保存的模型进行推理，自带可调机械降噪，可以任意选择已训练的音频进行推理
            * [模型修改] ： 在这里可以选择已保存的模型进行重命名，方便日后推理使用
            * [ AI降噪 ] :  在这里可以上传音频进行AI降噪，一键去除噪音杂声
            * [缓存清理] ： 如果训练时出现报错可以尝试缓存清理，每次保存模型会自动清理缓存，如果未保存就重新开始训练需要清理缓存
            * ！！注意！！  不要生成会对个人以及组织造成侵害的内容
            * 如果您需要录制您的声音，以下是一段长度合适的文本，供您朗读并录制：

            记得春天的时候，小草就转出地面，树上的叶子也抽出来了，大地一片绿色，就像穿上了一件绿衣裳。我就与小孩子一起到田野去捉蜻蜓，玩游戏，比如老鹰合作小鸡或是捉迷藏，又或是跳格子。到了夏天，天气热了，我就会与小孩子到水库里面游泳，那时候水库的安全系数还不是很高，几乎每年都会有事故发生，所以父母都不会让我去游泳的，被发现之后当然就是处罚或是责骂了。可是那时候自己真的很叛逆，也不知道什么是危险，被处罚之后下一次还是回去的。到了秋天，田野一片金黄，山上的野果也成熟了，我就会与自己的伙伴拿着篮子到上山去采，采回来了还要跟自己的好朋友一起分享。
                
            """
        gr.Markdown(dedent(_))


    gr.Markdown("### <center>注意❗：请不要生成会对个人以及组织造成侵害的内容，此程序仅供科研、学习及个人娱乐使用。</center>")
    gr.HTML('''
        <div class="footer">
                    <p>🌊🏞️🎶 - 江水东流急，滔滔无尽声。 明·顾璘
                    </p>
        </div>
    ''')


app.launch(show_error=True, share=False)
