def init_wandb():
    from dotenv import load_dotenv
    import os
    from pathlib import Path
    import wandb

    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path)

    key = os.getenv("WANDB_API_KEY")
    if not key:
        raise RuntimeError("Missing WANDB_API_KEY")

    wandb.login(key=key)