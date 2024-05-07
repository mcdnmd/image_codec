# Cжатие изображений при помощи нейронных сетей

## Описание работы кодека
Данный простой кодек сжимает изображения размером 128x128 при помощи свёрточных нейронных сетей, квантования и адаптивного арифметического кодирования.
В папке ./train/ находятся изображения, которые были использованы для обучения сети, в папке ./test/ находятся изображения для демонстрации результатов.


## Как запустить
Установка нужной версии питона
```shell
pyenv install
```

Установка poetry
```shell
pyenv exec python -m pip install poetry
```

Установка зависимостей с помощью poetry
```shell
poetry intall
```


Билд с нуля файла под архитектуру вашего процессора. Полсе команды вы получаете файл `EntropyCodec. ...`. 
```shell
make build_cpp
```
Под `arm64` и `python3.12` у меня собрался файл `EntropyCodec.cpython-312-darwin.so`

Имя этого файла необходимо подставить в файл `EntropyCodec.py`
```py
def __bootstrap__():
    global __bootstrap__, __loader__, __file__

    __file__ = pkg_resources.resource_filename(__name__, 'EntropyCodec.cpython-312-darwin.so')
    __loader__ = None
    del __bootstrap__, __loader__
    imp.load_dynamic(__name__, __file__)


__bootstrap__()
```

Так же для упрощения импорта в файле есть сигнатуры методов из собранных исходников чтобы можно было делать конкретный импорт и линтер не ругался
```py
from EntropyCodec import HiddenLayersEncoder, HiddenLayersDecoder
```


