from agenticnav_paper_replica.geometry.pinhole import parse_view_index

test_inputs = [0, 1, 2, 3, "0", "1", "2", "3", "Front", "Right", "Back", "Left", "front", "view 1", "VIEW 2", "left (90°)"]
print({inp: parse_view_index(inp) for inp in test_inputs})

