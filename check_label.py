import os, random
from PIL import Image

for cls in ['dry', 'normal', 'oily']:
    folder = f'dataset/{cls}'
    files = random.sample(os.listdir(folder), 16)
    sheet = Image.new('RGB', (800, 800))
    for i, f in enumerate(files):
        im = Image.open(os.path.join(folder, f)).convert('RGB').resize((200, 200))
        sheet.paste(im, ((i % 4) * 200, (i // 4) * 200))
    sheet.save(f'sheet_{cls}.jpg')
print('done')