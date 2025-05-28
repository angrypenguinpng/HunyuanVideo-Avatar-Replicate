# HunyuanVideo-Avatar on Replicate

This is a Replicate deployment of HunyuanVideo-Avatar, which supports animating avatar images to high-dynamic and emotion-controllable videos with audio conditions.

## Features

- **Multi-style avatars**: Photorealistic, cartoon, 3D-rendered, and anthropomorphic characters
- **Multi-scale generation**: Portrait, upper-body, and full-body animations  
- **High-dynamic content**: Realistic foreground and background generation
- **Emotion control**: Facial emotions controlled by input audio
- **Flexible input**: Any scale and resolution avatar images

## Usage

```python
import replicate

output = replicate.run(
    "your-username/hunyuan-video-avatar",
    input={
        "avatar_image": open("avatar.jpg", "rb"),
        "audio_file": open("speech.wav", "rb"),
        "resolution": "720x1280",
        "num_frames": 129,
        "body_type": "portrait",
        "emotion_control": True
    }
)
```

## Parameters

- `avatar_image`: Input avatar image (any style/resolution)
- `audio_file`: Audio for emotion control and lip sync
- `resolution`: Output video resolution (512x512, 720x1280, 1024x1024)
- `num_frames`: Number of frames (30-200)  
- `body_type`: Framing type (portrait, upper_body, full_body)
- `style`: Avatar style (auto-detected by default)
- `emotion_control`: Enable audio-based emotion control
- `seed`: Random seed for reproducibility

## Hardware Requirements

- Minimum: 24GB GPU memory (slow generation)
- Recommended: 80GB GPU memory for optimal performance

Based on the original [HunyuanVideo-Avatar](https://github.com/Tencent/HunyuanVideo-Avatar) by Tencent.
