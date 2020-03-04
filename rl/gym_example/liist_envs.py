#!/usr/bin/env python
from gym import envs
import gym
envids = [spec.id for spec in envs.registry.all()]
valid_envids = [

]
invalid_envids = [

]
for envid in sorted(envids):
    try:
        env = gym.make(envid)
        valid_envids.append(envid)
    except:
        invalid_envids.append(envid)
print(f"Valid env count: {len(valid_envids)}")
print(valid_envids)