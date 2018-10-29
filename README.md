# SumDU Data Science Course
**Welcome to the course homepage**

**Наш курс буде охоплювати актуальні питання машинного навчання, як класичні моделі так і останні досягнення і підходи**

**Структура репозитарію:**
- slides - початкове знайомство
- help - це допомога ;)
- data - датасети, які ми будемо використовувати в курсі та домашніх роботах
- lectures - лекції
- homeworks - домашні завдання, виконання всіх завдань гарантує максимальну оцінку
- module - завдання на модуль

**Зв'язок з викладачем забезпечується через чат та телеграм канал @VisualLanguages2018**

**Будь ласка ознайомтесь з інформацією нижче для  встановлення необхідного програмного забезпечення.**

Наш курс присвячений мові програмування Python 3.6 та сучасним бібліотекам машинного навчання та аналізу даних. Для роботи з курсом можливі декілька варіантів, а саме ми можемо працювати в хмарі і використовувати потужні процесори, відеокарти та більш ніж 16 Гб ОЗУ для наших моделей або встановити необхідне середовище розробки на локальний ноутбук або комп'ютер. Кожен підхід має свої переваги, безумовно, якщо ви маєте потужний ноутбук з 8Гб+ ОЗУ та видеокарту NVidia (для тренування нейромереж) то найкращий варіант це встановити локальне середовище. Якщо, ні то ми будемо користуватися безкоштовною хмарою, але безкоштовні сервіси не завжди надійні і це необхідно враховувати. Можна приєднатися до платних сервісів і отримати потужну систему в хмарі на даний час це не коштує велику суму.

**Розглянемо варіанти безкоштовних хмар:**

### Використання kernels від https://www.kaggle.com/ 
Kaggle дає зручну та безкоштовну хмару (хоча і не завжди стабільну) + велику кількість датасетів та змагань.
Ваші дії:  Реєструємося https://www.kaggle.com/ , далі зайти в розділ Kernels в горі сторінки, потім вибрати кнопку New Kernel (вгорі праворуч), в вікні яке випаде вибрати - Notebook (праворуч), з'явиться новий ноутбук, праворуч є меню для під'єднання даних (на кеглі дуже велика кількість датасетів, ми деякими будемо користуватися). Під'єднайте до вашого нового ноутбуку якійсь датасет (натисніть праворуч - "+Add Data" і виберіть зі списку який випаде необхідний датасет), наприклад: "Transaction from a bakery" . Запуск комірок відбувається, як і в стаціонарному Jupyter Notebook (про це далі) відбувається натисканням shift+enter. Прочитайте уважно опис, якій міститься в першій комірці Вашого Jupyter Notebook на kaggle (якщо необхідно під'єднайте розширення google translate до Вашого браузеру, після можні виділяти невеликі речення і переводити текст). Далі зчитати дані (після завантаження ваші дані будуть в хмарі kaggle в віртуальній папці "../input". Зчитайте дані наступним чином: df=pd.read_csv('../input/BreadBasket_DMS.csv'), де замість BreadBasket_DMS.csv (це датасет "Transaction from a bakery" вкажіть датасет, якій Ви підключили в попередніх кроках, ім'я датасету можна побачити коли Ви виконаєте першу комірку Вашого ноутбуку). Виконайте команду df.head() і подивіться на дані. Хмара на Кегл пропонує нам непогано залізо і безкоштовно: 5,2 Гб під дані, 16Гб ОЗУ (що дуже важливо для нас), та доволі потужні процесори, крім того в бета режимі можна користуватися ГПУ для нейромереж.

### Інший шлях, можна взяти хмару від гугла GOOGLE COLAB. Вона стабільніше. Але трохи треба повозитися для під'єднання даних.

Детально можна прочитати тут: https://albahnsen.com/2018/07/22/how-to-download-kaggle-data-into-google-colab/ . Ваші дії: Зайти на сайт https://colab.research.google.com/ (перед цим увійти в Ваш аккаунт на google.com, якщо не маєте то зареєструватися). Після чого ви побачите стартове вікно в якому необхідно вибрати звідкіля ви хочете завантажити файл (наприклад google drive або GitHub, якщо оберете GitHub то ви можете напряму отримати файли з мого репозитарію нашого курсу - лекції, домашн завдань і т.і.) або створити новий (всі Ваші нові файли та дані можуть зберігатися у Вас google drive). Розглянемо випадок коли Ви створюєте новий робочий файл та хочете до нього приєднати датасет з кегла. Оберіть внизу стартового вікна - NEW PYTHON 3 NOTEBOOK.  Далі перейдіть на сайт Kaggle (зареєструйтесь, якщо це не зробили), далі зайдіть в свій аккаунт або, якщо ви загубилися то йдіть за посиланням  https://www.kaggle.com/{username}/account  (замість {username} вставте свій username) після внизу сторінки знайдіть кнопку: Create New API Token.  Натисніть її, ви скачаєте собі на комп'ютер json file. Відкрийте цей файл будь яким текстовим редактором, в середині скачаного файлу буде всього одна строка, яка нам потрібна наступного формату - {“username”:”{username}”,”key”:”{API key}”} 

Поверніться до GOOGLE COLAB та в першу комірку вставте код нижче (замість строки {"username":"{user}","key":"{API key"} вставьте Вашу строку)

**!pip install -U -q kaggle**

**!mkdir -p ~/.kaggle**

**!echo '{"username":"{user}","key":"{API key"}' > ~/.kaggle/kaggle.json**

**!chmod 600 ~/.kaggle/kaggle.json**

Виконайте комірку натискаючи Shift+Enter

Ви приєдналися до Кегл і тепер можете отримувати датасети безпосередньо з Кегл. 

Наприклад вставте в наступну комірку код нижче:

**!mkdir -p data
!kaggle datasets download -d xvivancos/transactions-from-a-bakery
Виконайте комірку натискаючи Shift+Enter**

Ви скчаєте до себе в хмару тренувальний датасет https://www.kaggle.com/xvivancos/transactions-from-a-bakery
Далі вставте в наступну комірку код нижче:

**import pandas as pd
import os
df = pd.read_csv('transactions-from-a-bakery.zip')
df.head()**

Виконайте комірку натискаючи Shift+Enter

Ви побачите перші строки датасету у Вас в хмарі. Що далі робити з даними і як іх аналізувати ми будемо вивчати в нашому курсі.


### Найбільш кращий шлях це встановити на Ваш локальный ноутбук все необхідне програмне забезпечення: Anaconda Navigator та необхідні бібліотеки, детально цей процес описаний нижче.

Це найкраща ідея але якщо у Вашого ноутбуку є як мінімум 4 Гб ОЗУ, якщо менше, то це біль ))) особливо якщо Ви до цих пір на Віндовс.

Процес установки локального середовища розробки (див. нижче)


**Please find installation guide described below**

## Set-up Your Environment
### Prerequisites:
##### Hardware:
- Laptop or PC :) 
- 2+ Cores, 
- 4+ Gb of RAM (ideally 8+ Gb) 
- Free disc space 3+ Gb

##### Software:
- Recent build of [Anaconda](https://www.anaconda.com/download/) for Python 

### Installations:
* Download and install latest [Anaconda](https://www.anaconda.com/download/) build for **Python 3.6+**
  * [Windows guide](https://medium.com/@GalarnykMichael/install-python-on-windows-anaconda-c63c7c3d1444), works on Windows 8+
  * Linux (Ubuntu) guide: [This](https://medium.com/@GalarnykMichael/install-python-on-ubuntu-anaconda-65623042cb5a) or [this](https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04), works on Ubuntu 12.04 / 14.04 / 16.04
  * [Mac guide](https://medium.com/@GalarnykMichael/install-python-on-mac-anaconda-ccd9f2014072)
* Be sure to have Anaconda on PATH and make it your default Python interpreter
* Install/upgrade required packages via `pip install ...` in your Command Line / Terminal (admin privileges should be granted)
  * ```pip install -U pandas``` (22.0+)
  * ```pip install -U numpy``` (1.14.0+, if not installed as pandas dependency)
  * ```pip install -U scipy``` (1.0+)
  * ```pip install -U matplotlib``` (2.1.12+)
  * ```pip install -U lightgbm``` (2.1.0+) or use [official guide](https://github.com/Microsoft/LightGBM/tree/master/python-package)
  * ```pip install -U xgboost``` (0.7+) or use [official guide](https://github.com/dmlc/xgboost/blob/master/doc/build.md)
  * ```pip install -U scikit-learn``` (0.19.1+)
  * download and install proper version of GraphViz from [here](https://graphviz.gitlab.io/download/)
  * ```pip install -U pydot```
  * ```conda install nb_conda``` (to join it with jupyter notebooks)
  * (optional) install `wordcloud` with ```pip install wordcloud``` or use [this solution](https://github.com/amueller/word_cloud/issues/134) for Windows OS
* Check whether your environment is set up properly or not
  * `cd to-directory-containing-file-check_environment.py`
  * `python check_environment.py`
  * `jupyter notebook`

Profit! You are ready to create/view Jupyter `.ipynb` notebooks :)
