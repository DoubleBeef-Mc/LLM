import torch
import os
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# 环境设置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


model_path = "C:\\Users\\mbr\\Desktop\\LLM\\LLM\\Qwen3-TTS-12Hz-1.7B-CustomVoice"


model = Qwen3TTSModel.from_pretrained(
    model_path,
    device_map="cuda:0",
    dtype=torch.float16,
    low_cpu_mem_usage=True
)


with torch.no_grad():
    text = "美国总统拜登宣布将向乌克兰提供新的军事援助，包括先进的防空系统和无人机，以帮助乌克兰抵御俄罗斯的侵略。"
    language = "Chinese"     
    speaker = "Vivian"       

    audio, sr = model.generate_custom_voice(
        text=text,
        language=language,
        speaker=speaker,
        
    )


sf.write("output1.wav", audio[0], sr)
print("语音生成成功")