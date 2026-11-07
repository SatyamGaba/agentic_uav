import habitat_sim

def test_gpu_id(dev_id):
    print(f"\n--- Testing gpu_device_id={dev_id} ---")
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = "/home/krutik/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
    backend_cfg.gpu_device_id = dev_id
    
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    try:
        sim = habitat_sim.Simulator(sim_cfg)
        print(f"SUCCESS with gpu_device_id={dev_id}! sim.gpu_device={sim.gpu_device}")
        sim.close()
        return True
    except Exception as e:
        print(f"FAILED with gpu_device_id={dev_id}: {e}")
        return False

test_gpu_id(0)
test_gpu_id(-1)
test_gpu_id(1)

