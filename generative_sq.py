import random
# import uuid
from PIL import Image, ImageDraw

# run_id = uuid.uuid1()
# print(f"Processing run_id: {run_id}")

image = Image.new('RGB', (128, 128))
width, height = image.size

rectangle_width = 8
rectangle_height = 8

num_squares = random.randint(10, 150)

draw_img = ImageDraw.Draw(image)
for i in range(num_squares):
    rectangle_x = random.randint(0, width)
    rectangle_y = random.randint(0, height)

    rectangle_shape = [
        (rectangle_x, rectangle_y),
        (rectangle_x + rectangle_width, rectangle_y + rectangle_height)
    ]
    draw_img.rectangle(
        rectangle_shape,
        fill=(
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    )

# image.save(f'output/{run_id}.png')
image.save("output/sq_img.png")

