import timm
model = timm.list_models('*vit*')
print(*model, sep='\n')