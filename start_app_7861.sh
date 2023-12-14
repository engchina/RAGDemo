eval "$(conda shell.bash hook)"
conda activate ragdemo
nohup python app_v2.py --host 0.0.0.0 --port 7861 --gradio-auth-path auth.txt &