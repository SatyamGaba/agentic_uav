import habitat_sim

def test_sensor_render():
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = "/home/krutik/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
    backend_cfg.gpu_device_id = 0
    
    color_sensor = habitat_sim.CameraSensorSpec()
    color_sensor.uuid = "color_sensor"
    color_sensor.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor.resolution = [480, 640]
    color_sensor.position = [0.0, 1.25, 0.0]
    
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [color_sensor]
    
    sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    try:
        sim = habitat_sim.Simulator(sim_cfg)
        print("Simulator with visual sensor created successfully!")
        obs = sim.get_sensor_observations()
        print("Observation keys:", obs.keys())
        rgb = obs["color_sensor"]
        print("RGB shape:", rgb.shape)
        sim.close()
    except Exception as e:
        print("Exception:", e)

test_sensor_render()

