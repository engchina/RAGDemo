eval "$(conda shell.bash hook)"
conda activate ragdemo
nohup python app_ai_v3.py --host 0.0.0.0 --port 7861 --gradio-auth-path auth.txt &