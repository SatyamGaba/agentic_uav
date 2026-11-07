import sys
import exec_cmd

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

cmd = """python -c "
import urllib.request, json
tools = [
    {
        'type': 'function',
        'function': {
            'name': 'plan_flight',
            'description': 'Execute an aerial flight plan composed of discrete OpenFly action integers.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'action_codes': {
                        'type': 'array',
                        'items': {'type': 'integer'},
                        'description': 'List of OpenFly action integers: 9 (forward 9m), 8 (forward 6m), 1 (forward 3m), 3 (turn left -30 deg), 2 (turn right +30 deg), 4 (climb 3m), 5 (descend 3m), 0 (stop).'
                    }
                },
                'required': ['action_codes']
            }
        }
    }
]

data = json.dumps({
    'model': 'qwen3-vl:8b',
    'messages': [
        {'role': 'user', 'content': 'Instruction: Fly forward towards the building and land.\\nSelect the sequence of action codes using plan_flight.'}
    ],
    'tools': tools,
    'stream': False,
    'options': {'temperature': 0.0}
}).encode('utf-8')

req = urllib.request.Request('http://127.0.0.1:11434/api/chat', data=data, headers={'Content-Type': 'application/json'})
resp = urllib.request.urlopen(req).read().decode('utf-8')
print(resp)
"
"""

print(exec_cmd.run_remote(cmd, wait_sec=25.0))
