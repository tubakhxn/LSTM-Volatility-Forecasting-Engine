import torch
import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

def save_lstm(state_dict, name="lstm_vol_best"):
    torch.save(state_dict, MODEL_DIR / f"{name}.pt")

def load_lstm(model, name="lstm_vol_best"):
    path = MODEL_DIR / f"{name}.pt"
    if path.exists():
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    return model

def save_hmm(hmm_model, name="hmm_regime"):
    joblib.dump(hmm_model, MODEL_DIR / f"{name}.pkl")

def load_hmm(name="hmm_regime"):
    path = MODEL_DIR / f"{name}.pkl"
    return joblib.load(path) if path.exists() else None

def save_dqn(state_dict, name="dqn_policy"):
    torch.save(state_dict, MODEL_DIR / f"{name}.pt")

def load_dqn(agent, name="dqn_policy"):
    path = MODEL_DIR / f"{name}.pt"
    if path.exists():
        agent.policy_net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
    return agent
