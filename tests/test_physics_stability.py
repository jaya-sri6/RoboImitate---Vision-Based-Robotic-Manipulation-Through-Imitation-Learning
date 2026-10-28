import mujoco


def test_physics_stability():
    xml = """
    <mujoco model="test">
      <option timestep="0.005" gravity="0 0 -9.81"/>
      <worldbody>
        <body name="cart" pos="0 0 0.5">
          <joint name="slider_z" type="slide" axis="0 0 1" range="-0.2 0.2" damping="20"/>
          <geom type="cylinder" size="0.03 0.02" mass="1.0"/>
        </body>
      </worldbody>
      <actuator>
        <position joint="slider_z" kp="400" kv="20" ctrlrange="-0.2 0.2"/>
      </actuator>
    </mujoco>
    """
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    for _ in range(500):
        d.ctrl[0] = -0.1
        mujoco.mj_step(m, d)
    # Steady-state position includes gravity offset: mg / kp = (1.0 * 9.81) / 400 = 0.024525
    expected_pos = -0.1 - (1.0 * 9.81 / 400.0)
    assert abs(d.qpos[0] - expected_pos) < 0.005, f"Cart position deviates from equilibrium: {d.qpos[0]}"
    assert abs(d.qvel[0]) < 0.005, f"Cart velocity is not stabilized: {d.qvel[0]}"


if __name__ == "__main__":
    test_physics_stability()
    print("Success, perfectly stable without any warning!")

