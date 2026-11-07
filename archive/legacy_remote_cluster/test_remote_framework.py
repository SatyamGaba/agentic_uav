import exec_cmd

cmd = """python -c "import sys; sys.path.insert(0, '/home/jovyan/scratch'); import framework; print('SUCCESS: Framework successfully loaded on remote L40S server! Registered adapters:', list(framework.ADAPTER_REGISTRY.keys()))" """

res = exec_cmd.run_remote(cmd, wait_sec=8.0)
print(res)
