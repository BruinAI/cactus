from transformers import MoonshineForConditionalGeneration

model = MoonshineForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-tiny")

for key in model.state_dict():
    print(key)