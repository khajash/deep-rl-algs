import importlib

def load_policy_class(policy_name):
    mod = importlib.import_module("playground.policies")
    policy_class = getattr(mod, policy_name)
    return policy_class