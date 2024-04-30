import asyncio
import time

from aiogram import Bot, types
from aiogram.dispatcher.dispatcher import Dispatcher
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model


def load_image(filename):
    img = load_img(filename, target_size=(224, 224))
    img = img_to_array(img)
    img = img.reshape(1, 224, 224, 3)
    img = img.astype('float32')
    return img


bot = Bot(token="6386381405:AAEU3CNwZ7wzapAq-k7SdqELqt40WBePbJM")
dp = Dispatcher()

model = load_model("cat_or_dog_img_classifier.h5")


@dp.message()
async def handle_docs_photo(message):
    file_id = message.document.file_id
    file = await bot.get_file(file_id)
    file_path = file.file_path
    await bot.download_file(file_path, "test.jpg")
    time.sleep(5)
    test_img = load_image("test.jpg")
    if model.predict(test_img):
        await message.answer("It's a dog!")
    else:
        await message.answer('What a pretty cat!')


async def main():
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())
