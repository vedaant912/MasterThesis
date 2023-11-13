from models.fasterrcnn_resnet18 import create_model

model = create_model(2)

print(model.backbone[0].state_dict())