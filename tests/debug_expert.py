from roboimitate.env.manipulation_env import RoboImitatePickAndPlaceEnv
from roboimitate.evaluation.baseline import ScriptedPickAndPlaceExpert

env = RoboImitatePickAndPlaceEnv(config={"physics": {"max_episode_steps": 80}})
expert = ScriptedPickAndPlaceExpert(
    max_delta_pos=0.03,
    hover_height=0.12,
    grasp_ee_z=0.460,
    lift_ee_z=0.540,
    place_ee_z=0.465,
    close_gripper_steps=4,
    open_gripper_steps=4,
)
obs, info = env.reset(seed=42)
expert.reset()

print("Initial Privileged State:")
priv = env.get_privileged_state()
print("EE pos:", priv["ee_pos"])
print("Obj pos:", priv["object_pos"])
print("Target pos:", priv["target_pos"])

for s in range(80):
    priv = env.get_privileged_state()
    act = expert.get_action(priv)
    obs, rew, term, trunc, info = env.step(act)
    ee = priv["ee_pos"]
    obj = priv["object_pos"]
    if s % 5 == 0 or term or trunc:
        print(f"step {s:02d} | phase {expert.phase.name:13s} | ee: [{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}] | obj: [{obj[0]:.3f}, {obj[1]:.3f}, {obj[2]:.3f}] | term={term} | trunc={trunc}")
    if term or trunc:
        break

print("Final Success:", info.get("success", False))
print("Final Info:", info)
