import urllib.request
import re

url = "https://ollama.com/library/qwen3.5"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
html = urllib.request.urlopen(req).read().decode('utf-8')

desc = re.findall(r'<meta name="description" content="(.*?)"', html)
print("Description:", desc)

# Check if vision or multimodal is mentioned
text_lower = html.lower()
print("Contains 'vision':", "vision" in text_lower)
print("Contains 'image':", "image" in text_lower)
print("Contains 'multimodal':", "multimodal" in text_lower)

# Also check qwen3-vl
url_vl = "https://ollama.com/library/qwen3-vl"
try:
    req_vl = urllib.request.Request(url_vl, headers={'User-Agent': 'Mozilla/5.0'})
    html_vl = urllib.request.urlopen(req_vl).read().decode('utf-8')
    desc_vl = re.findall(r'<meta name="description" content="(.*?)"', html_vl)
    print("\nQwen3-VL Description:", desc_vl)
except Exception as e:
    print("\nQwen3-VL:", e)

