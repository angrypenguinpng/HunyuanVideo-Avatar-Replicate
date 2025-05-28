import os
import sys
import torch
import tempfile
import shutil
import argparse
from pathlib import Path
from typing import Optional
from cog import BasePredictor, Input, Path as CogPath
from loguru import logger

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        print("🚀 Setting up HunyuanVideo-Avatar...")
        
        # Find the HunyuanVideo-Avatar installation in the Docker container
        possible_paths = [
            "/workspace/HunyuanVideo-Avatar",
            "/src/HunyuanVideo-Avatar", 
            "/app/HunyuanVideo-Avatar",
            "/HunyuanVideo-Avatar",
            "/root/HunyuanVideo-Avatar"
        ]
        
        self.model_dir = None
        for path in possible_paths:
            if os.path.exists(path):
                self.model_dir = path
                print(f"✅ Found HunyuanVideo-Avatar at: {path}")
                break
        
        if not self.model_dir:
            raise RuntimeError("❌ HunyuanVideo-Avatar directory not found in container")
        
        # Add to Python path and change directory
        sys.path.insert(0, self.model_dir)
        os.chdir(self.model_dir)
        
        # Check CUDA availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Using device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Try to import the required modules
        try:
            from hymm_sp.sample_inference_audio import HunyuanVideoSampler
            self.HunyuanVideoSampler = HunyuanVideoSampler
            print("✅ Successfully imported HunyuanVideoSampler")
        except ImportError as e:
            print(f"❌ Failed to import HunyuanVideoSampler: {e}")
            print("📁 Available Python files in model directory:")
            for root, dirs, files in os.walk(self.model_dir):
                for file in files:
                    if file.endswith('.py'):
                        rel_path = os.path.relpath(os.path.join(root, file), self.model_dir)
                        print(f"   {rel_path}")
            # Don't raise here - we'll use CLI fallback
            self.HunyuanVideoSampler = None
        
        # Set up default arguments for the model
        self.args = self._setup_default_args()
        
        # Find model weights path
        self.model_path = self._find_model_path()
        print(f"📦 Model path: {self.model_path}")
        
        # Set MODEL_BASE environment variable if not set
        if 'MODEL_BASE' not in os.environ:
            os.environ['MODEL_BASE'] = str(Path(self.model_path).parent)
            print(f"🔧 Set MODEL_BASE to: {os.environ['MODEL_BASE']}")
        
        # Don't pre-load the model in setup - it's heavy and we'll load it per-prediction
        self.inference_pipeline = self.HunyuanVideoSampler  # Just store the class
        print("⚠️  Model will be loaded per-prediction to save memory")
        
        print("✅ Setup complete!")

    def _setup_default_args(self):
        """Set up default arguments for the HunyuanVideo-Avatar model"""
        # Create a simple args object with default values
        # You might need to adjust these based on the actual model requirements
        class Args:
            def __init__(self):
                # Model settings
                self.precision = "bf16"
                self.text_encoder_precision = "fp16"
                self.text_encoder_precision_2 = "fp16"
                self.vae_precision = "fp16"
                
                # Model architecture
                self.latent_channels = 16
                self.use_fp8 = False
                self.cpu_offload = False
                
                # Text encoder settings
                self.text_encoder = "llava"
                self.text_encoder_2 = None
                self.tokenizer = "llava"
                self.tokenizer_2 = None
                self.text_len = 256
                self.text_len_2 = 256
                self.use_attention_mask = True
                self.prompt_template_video = None
                self.hidden_state_skip_layer = 2
                self.apply_final_norm = False
                self.reproduce = False
                
                # VAE settings
                self.vae = "884-16ch-128f"
                
                # Loading settings
                self.load_key = "module"
        
        return Args()

    def _find_model_path(self):
        """Find the model weights path in the container"""
        possible_model_paths = [
            "/workspace/models",
            "/workspace/HunyuanVideo-Avatar/models", 
            "/models",
            "/src/models",
            "/app/models",
            f"{self.model_dir}/models",
            f"{self.model_dir}/checkpoints",
            f"{self.model_dir}/weights"
        ]
        
        for path in possible_model_paths:
            if os.path.exists(path):
                print(f"📂 Found models directory: {path}")
                return path
        
        # If no standard path found, look for .pt or .pth files
        for root, dirs, files in os.walk(self.model_dir):
            for file in files:
                if file.endswith(('.pt', '.pth', '.safetensors')):
                    model_file = os.path.join(root, file)
                    print(f"🔍 Found model file: {model_file}")
                    return os.path.dirname(model_file)
        
        # Default fallback
        return f"{self.model_dir}/models"

    def predict(
        self,
        avatar_image: CogPath = Input(
            description="Input avatar image (supports photorealistic, cartoon, 3D-rendered, anthropomorphic characters)"
        ),
        audio_file: CogPath = Input(
            description="Audio file for controlling facial emotions and lip sync"
        ),
        prompt: str = Input(
            description="Text prompt describing the desired video (optional)",
            default="A person speaking with natural expressions"
        ),
        resolution: str = Input(
            description="Output video resolution",
            choices=["512x512", "720x1280", "1024x1024", "1280x720"],
            default="720x1280"
        ),
        num_frames: int = Input(
            description="Number of frames to generate",
            default=129,
            ge=25,
            le=200
        ),
        fps: int = Input(
            description="Output video frame rate",
            default=24,
            ge=15,
            le=60
        ),
        guidance_scale: float = Input(
            description="Guidance scale for generation quality",
            default=7.5,
            ge=1.0,
            le=20.0
        ),
        seed: Optional[int] = Input(
            description="Random seed for reproducibility. Leave empty for random generation.",
            default=None
        )
    ) -> CogPath:
        """Generate avatar video from image and audio inputs"""
        
        print("🎬 Starting avatar video generation...")
        
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            print(f"🌱 Using seed: {seed}")
        
        # Parse resolution
        if 'x' in resolution:
            width, height = map(int, resolution.split('x'))
        else:
            width = height = int(resolution)
        
        print(f"📐 Resolution: {width}x{height}, Frames: {num_frames}, FPS: {fps}")
        print(f"💬 Prompt: {prompt}")
        
        # Create temporary workspace
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup input/output paths
            avatar_input = os.path.join(temp_dir, f"avatar{Path(str(avatar_image)).suffix}")
            audio_input = os.path.join(temp_dir, f"audio{Path(str(audio_file)).suffix}")
            video_output = os.path.join(temp_dir, "output_video.mp4")
            
            # Copy input files
            shutil.copy2(str(avatar_image), avatar_input)
            shutil.copy2(str(audio_file), audio_input)
            
            print(f"📁 Inputs prepared: {os.path.basename(avatar_input)}, {os.path.basename(audio_input)}")
            
            # Try generation methods
            success = False
            
            # Method 1: Use Python API if available
            if self.inference_pipeline is not None:
                try:
                    print("🐍 Using Python API...")
                    success = self._generate_with_api(
                        avatar_input, audio_input, video_output,
                        prompt, width, height, num_frames, fps, guidance_scale, seed
                    )
                except Exception as e:
                    print(f"❌ Python API failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Method 2: Fallback to CLI
            if not success:
                print("🔄 Falling back to CLI method...")
                success = self._generate_with_cli(
                    avatar_input, audio_input, video_output,
                    prompt, width, height, num_frames, fps, guidance_scale, seed
                )
            
            # Final check
            if not success or not os.path.exists(video_output):
                # Debug information
                print("🔍 Debug info - temp directory contents:")
                for item in os.listdir(temp_dir):
                    print(f"   {item}")
                
                raise RuntimeError("❌ All generation methods failed. Check the logs above for details.")
            
            # Verify output file
            file_size = os.path.getsize(video_output)
            print(f"🎉 Generated video: {file_size / (1024*1024):.1f} MB")
            
            return CogPath(video_output)

    def _generate_with_api(self, avatar_path, audio_path, output_path, 
                          prompt, width, height, num_frames, fps, guidance_scale, seed):
        """Generate using the Python API based on the actual HunyuanVideo-Avatar implementation"""
        try:
            import numpy as np
            from einops import rearrange
            from transformers import WhisperModel, AutoFeatureExtractor
            from hymm_sp.sample_inference_audio import HunyuanVideoSampler
            from hymm_sp.data_kits.face_align import AlignImage
            
            print("🔄 Setting up audio processing models...")
            
            # Set up Whisper model for audio processing
            model_base = os.environ.get('MODEL_BASE', f"{self.model_dir}/models")
            whisper_path = f"{model_base}/ckpts/whisper-tiny/"
            
            if not os.path.exists(whisper_path):
                # Try alternative paths
                whisper_alternatives = [
                    f"{self.model_dir}/models/whisper-tiny",
                    f"{self.model_dir}/whisper-tiny",
                    "whisper-tiny"  # Will download from HuggingFace
                ]
                for alt_path in whisper_alternatives:
                    if os.path.exists(alt_path):
                        whisper_path = alt_path
                        break
                else:
                    whisper_path = "openai/whisper-tiny"  # Download from HF
            
            wav2vec = WhisperModel.from_pretrained(whisper_path).to(device=self.device, dtype=torch.float32)
            wav2vec.requires_grad_(False)
            
            feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_path)
            
            # Set up face alignment
            det_path = f"{model_base}/ckpts/det_align/detface.pt"
            if not os.path.exists(det_path):
                print(f"⚠️  Face detection model not found at {det_path}")
                # Try to find it elsewhere
                for root, dirs, files in os.walk(self.model_dir):
                    if "detface.pt" in files:
                        det_path = os.path.join(root, "detface.pt")
                        break
            
            align_instance = AlignImage(self.device, det_path=det_path)
            
            print("🔄 Loading HunyuanVideo sampler...")
            
            # Create the sampler (this replaces our self.inference_pipeline)
            hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(
                self.model_path, 
                args=self.args, 
                device=self.device
            )
            
            print("🔄 Preparing batch data...")
            
            # Create batch data in the format expected by the model
            batch = {
                "fps": torch.tensor([fps], dtype=torch.float32),
                "videoid": [f"generated_{torch.randint(0, 10000, (1,)).item()}"],
                "audio_path": [audio_path],
                "image_path": [avatar_path],
                "audio_len": [torch.tensor(num_frames)],
                "height": torch.tensor([height]),
                "width": torch.tensor([width]),
                "prompt": [prompt] if prompt else ["A person speaking naturally"],
                "guidance_scale": torch.tensor([guidance_scale])
            }
            
            print("🎬 Running generation...")
            
            # Run the actual generation
            samples = hunyuan_video_sampler.predict(
                self.args, 
                batch, 
                wav2vec, 
                feature_extractor, 
                align_instance
            )
            
            print("🔄 Processing output...")
            
            # Extract and process the generated video
            sample = samples['samples'][0].unsqueeze(0)  # denoised latent
            sample = sample[:, :, :num_frames]  # Trim to desired length
            
            # Convert from latent to pixel space
            video = rearrange(sample[0], "c f h w -> f h w c")
            video = (video * 255.).data.cpu().numpy().astype(np.uint8)
            
            # Save video frames
            final_frames = []
            for frame in video:
                final_frames.append(frame)
            final_frames = np.stack(final_frames, axis=0)
            
            # Save the video file
            from hymm_sp.data_kits.ffmpeg_utils import save_video
            temp_video_path = output_path.replace('.mp4', '_temp.mp4')
            save_video(final_frames, temp_video_path, n_rows=len(final_frames), fps=fps)
            
            # Combine video with original audio
            import subprocess
            subprocess.run([
                'ffmpeg', '-i', temp_video_path, '-i', audio_path, 
                '-shortest', output_path, '-y', '-loglevel', 'quiet'
            ], check=True)
            
            # Cleanup temp file
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            
            print("✅ API generation completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ API generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _generate_with_cli(self, avatar_path, audio_path, output_path,
                          prompt, width, height, num_frames, fps, guidance_scale, seed):
        """Generate using CLI interface"""
        import subprocess
        
        # Look for generation scripts
        script_candidates = [
            "generate.py",
            "inference.py", 
            "run_inference.py",
            "main.py",
            "demo.py",
            "generate_video.py",
            "sample.py"
        ]
        
        generation_script = None
        for script in script_candidates:
            script_path = os.path.join(self.model_dir, script)
            if os.path.exists(script_path):
                generation_script = script_path
                break
        
        if not generation_script:
            print("❌ No generation script found")
            return False
        
        print(f"📜 Using script: {os.path.basename(generation_script)}")
        
        # Build command with common argument patterns
        cmd_variants = [
            # Pattern 1: Detailed arguments
            [
                "python", generation_script,
                "--avatar_image", avatar_path,
                "--audio", audio_path,
                "--prompt", prompt,
                "--output", output_path,
                "--width", str(width),
                "--height", str(height),
                "--num_frames", str(num_frames),
                "--fps", str(fps),
                "--guidance_scale", str(guidance_scale)
            ],
            # Pattern 2: Alternative argument names
            [
                "python", generation_script,
                "--input_image", avatar_path,
                "--input_audio", audio_path,
                "--text_prompt", prompt,
                "--output_path", output_path,
                "--resolution", f"{width}x{height}",
                "--frames", str(num_frames)
            ],
            # Pattern 3: Simplified
            [
                "python", generation_script,
                "--image", avatar_path,
                "--audio", audio_path,
                "--output", output_path
            ]
        ]
        
        for i, cmd in enumerate(cmd_variants):
            if seed is not None:
                cmd.extend(["--seed", str(seed)])
            
            print(f"🔄 Trying CLI pattern {i+1}...")
            print(f"Command: {' '.join(cmd)}")
            
            try:
                result = subprocess.run(
                    cmd,
                    cwd=self.model_dir,
                    timeout=900,  # 15 minute timeout
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                if result.returncode == 0 and os.path.exists(output_path):
                    print("✅ CLI generation successful!")
                    return True
                else:
                    print(f"❌ CLI pattern {i+1} failed (return code: {result.returncode})")
                    if result.stdout:
                        print(f"STDOUT: {result.stdout[-1000:]}")
                    if result.stderr:
                        print(f"STDERR: {result.stderr[-1000:]}")
            
            except subprocess.TimeoutExpired:
                print(f"⏰ CLI pattern {i+1} timed out")
            except Exception as e:
                print(f"💥 CLI pattern {i+1} exception: {e}")
        
        return False