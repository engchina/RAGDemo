eval "$(conda shell.bash hook)"
conda activate ragdemo
nohup python app_mfg_v2.py --host 0.0.0.0 --port 7862 --gradio-auth-path auth.txt &