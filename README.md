## Деконволюция
### Задание №1 в рамках курса «Вариационные методы обработки изображений»

В программе реализован метод обращения свёртки (деконволюции) изображений через минимизацию регуляризирующего функционала. Выбранный стабилизатор — функционал полной вариации.

### Запуск из командной строки:

Программа поддерживает запуск из командной строки со строго определённым форматом команд:

``` bash
python main.py input_image kernel output_image noise_level
```

Аргументы:
* input_image   — входное размытое и зашумлённое изображение (имя файла)
* kernel        — изображение ядра размытия (имя файла)
* output_image  — выходное изображение (имя файла)
* noise_level   — уровень шума на входном изображении

Уровень шума — вещественное число, среднеквадратичное отклонение (корень из дисперсии), для диапазона значений пикселей [0, 255].

---

В `test_data` находятся исходные размытые изображения, ядро размытия, референсные изображения, а также результаты работы алгоритма.

Файл `blur.py` можно использовать для создания размытых и зашумленных изображений.

``` bash
python blur.py input_image kernel output_image noise_level
```

Аргументы:
* input_image   — входное изображение (имя файла)
* kernel        — изображение ядра размытия (имя файла)
* output_image  — выходное размытое и зашумлённое изображение (имя файла)
* noise_level   — уровень шума на выходном изображении

### Примеры:
Размытое и зашумленное изображение       |  Результат
:-------------------------:|:-------------------------:
![](/test_data/test_blurred.bmp)   |  ![](/test_data/test_result.bmp)